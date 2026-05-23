"""
PenaltyAreaDetector
===================
Erkennt den Strafraum pro Frame aus weissen Linien (HoughLinesP).
Arbeitet mit Stubs genau wie CameraMovementEstimator:

  pa = PenaltyAreaDetector()
  pa_data = pa.get_penalty_area_data(
      video_frames,
      read_from_stub=True,
      stub_path='stubs/penalty_area_stub.pkl'
  )
  pa.add_penalty_area_to_tracks(tracks, pa_data)

Stub-Format  (Liste, ein Eintrag pro Frame):
  {
    "polygon"         : np.ndarray (N,2) int32 | None
                        Konvexes Polygon des erkannten Strafraums
    "side"            : "left" | "right" | None
                        Bildseite des Strafraums (Polygon-Schwerpunkt X vs. Bildmitte)
    "frame_width"     : int
    "players_in_area" : list[player_id]
                        IDs aller Spieler deren Fussposition im Polygon liegt.
                        Wird von add_penalty_area_to_tracks() befuellt.
  }

Spieler-Track-Eintraege (nach add_penalty_area_to_tracks):
  "in_penalty_area"   : bool
  "penalty_area_side" : "left" | "right" | None
"""

import os
import pickle
import numpy as np
import cv2


class PenaltyAreaDetector:

    # ── Strafraum-Erkennung ───────────────────────────────────────────────

    @staticmethod
    def _green_mask(frame: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    @staticmethod
    def detect_penalty_area(frame: np.ndarray) -> np.ndarray | None:
        """
        Erkennt den Strafraum in einem einzelnen Frame.

        Strategie: HoughLinesP findet weisse Linien auf gruenem Gras.
        Es wird nach einer horizontalen Grundlinie + zwei Seitenlinien
        gesucht (U-Form). Die konvexe Hülle aller Linienpunkte bildet
        das Strafraum-Polygon.

        Returns: np.ndarray shape (N, 2) dtype int32  oder  None
        """
        h, w = frame.shape[:2]

        # Weisse Linien auf gruenem Gras
        green = PenaltyAreaDetector._green_mask(frame)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 185), (180, 45, 255))
        white = cv2.bitwise_and(white, green)
        k2    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k2)

        # Hough-Linien (mindestens w/10 lang)
        min_len = max(w // 10, 40)
        lines   = cv2.HoughLinesP(
            white, rho=1, theta=np.pi / 180,
            threshold=35, minLineLength=min_len, maxLineGap=30,
        )
        if lines is None or len(lines) < 3:
            return None

        lines = lines.reshape(-1, 4)

        # Horizontale vs. schraege/vertikale Linien trennen
        horiz, angled = [], []
        for l in lines:
            dx, dy = float(l[2] - l[0]), float(l[3] - l[1])
            angle  = abs(np.degrees(np.arctan2(dy, dx + 1e-6)))
            length = (dx ** 2 + dy ** 2) ** 0.5
            if angle < 25 or angle > 155:
                horiz.append((l, length))
            elif 35 < angle < 145:
                angled.append((l, length))

        horiz.sort(key=lambda x: -x[1])
        angled.sort(key=lambda x: -x[1])

        if not horiz or len(angled) < 2:
            return None

        best_hull, best_area = None, 0.0
        min_area = w * h * 0.015   # mind. 1.5% des Frames

        for (end_l, _) in horiz[:6]:
            x1, y1, x2, y2 = end_l
            cy    = (y1 + y2) / 2.0
            mid_x = (min(x1, x2) + max(x1, x2)) / 2.0

            left_pts, right_pts = [], []
            for (al, _) in angled[:12]:
                ax1, ay1, ax2, ay2 = al
                for ex, ey in [(ax1, ay1), (ax2, ay2)]:
                    if abs(ey - cy) < max(h * 0.06, 45):
                        bucket = left_pts if ex < mid_x else right_pts
                        bucket.append((ax1, ay1))
                        bucket.append((ax2, ay2))
                        break

            if not left_pts or not right_pts:
                continue

            all_pts = [(x1, y1), (x2, y2)] + left_pts + right_pts
            hull    = cv2.convexHull(np.array(all_pts, dtype=np.int32))
            area    = float(cv2.contourArea(hull))

            if area <= min_area:
                continue

            bx, by, bw_, bh_ = cv2.boundingRect(hull)
            aspect = bw_ / max(bh_, 1)
            if not (0.3 < aspect < 8.0):
                continue

            if area > best_area:
                best_area = area
                best_hull = hull.reshape(-1, 2).astype(np.int32)

        return best_hull

    # ── Stub-basierte Hauptmethode ────────────────────────────────────────

    def get_penalty_area_data(
        self,
        video_frames: list,
        read_from_stub: bool = False,
        stub_path: str | None = None,
    ) -> list[dict]:
        """
        Berechnet oder laedt Strafraum-Daten pro Frame.

        Parameters
        ----------
        video_frames  : Liste von BGR-Frames (numpy arrays)
        read_from_stub: Wenn True und stub_path existiert -> aus Stub lesen
        stub_path     : Pfad zur .pkl-Datei

        Returns
        -------
        list[dict] – ein Eintrag pro Frame:
          {
            "polygon"         : np.ndarray (N,2) | None,
            "side"            : "left" | "right" | None,
            "frame_width"     : int,
            "players_in_area" : []          # wird von add_... befuellt
          }
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"[PenaltyArea] Lade Stub: {stub_path}")
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        print("[PenaltyArea] Erkenne Strafraum pro Frame (CV2) ...")
        pa_data     = []
        pa_detected = 0
        n           = len(video_frames)

        for fi, frame in enumerate(video_frames):
            fw   = frame.shape[1]
            poly = self.detect_penalty_area(frame)

            if poly is not None:
                cx   = float(poly[:, 0].mean())
                side = "left" if cx < fw / 2 else "right"
                pa_detected += 1
            else:
                side = None

            pa_data.append({
                "polygon":          poly,
                "side":             side,
                "frame_width":      fw,
                "players_in_area":  [],   # wird von add_penalty_area_to_tracks befuellt
            })

        print(f"[PenaltyArea] Erkannt in {pa_detected}/{n} Frames "
              f"({100 * pa_detected / n:.0f}%)")

        if stub_path:
            os.makedirs(os.path.dirname(stub_path) if os.path.dirname(stub_path) else ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(pa_data, f)
            print(f"[PenaltyArea] Stub gespeichert: {stub_path}")

        return pa_data

    # ── Tracks anreichern ─────────────────────────────────────────────────

    def add_penalty_area_to_tracks(
        self,
        tracks: dict,
        pa_data: list[dict],
    ) -> None:
        """
        Fuegt jedem Spieler-Track-Eintrag hinzu:
          'in_penalty_area'   : bool
          'penalty_area_side' : "left" | "right" | None

        Befuellt ausserdem pa_data[fi]['players_in_area'] mit den
        Player-IDs aller Spieler im Strafraum dieses Frames.

        (Laeuft in O(frames * players), ist fast.)
        """
        for fi, (frame_players, pa_frame) in enumerate(
                zip(tracks["players"], pa_data)):

            poly = pa_frame.get("polygon")
            side = pa_frame.get("side")
            fw   = pa_frame.get("frame_width", 1920)

            # players_in_area bei jedem Aufruf neu befuellen
            # (damit doppelter Aufruf keine Duplikate erzeugt)
            pa_frame["players_in_area"] = []

            for pid, pdata in frame_players.items():
                if poly is None:
                    pdata["in_penalty_area"]   = False
                    pdata["penalty_area_side"] = None
                    continue

                bbox   = pdata["bbox"]
                foot_x = float((bbox[0] + bbox[2]) / 2)
                foot_y = float(bbox[3])

                in_pa = cv2.pointPolygonTest(
                    poly.reshape(-1, 1, 2),
                    (foot_x, foot_y),
                    False,
                ) >= 0

                pdata["in_penalty_area"]   = in_pa
                pdata["penalty_area_side"] = side if in_pa else None

                if in_pa:
                    pa_frame["players_in_area"].append(pid)
