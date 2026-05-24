import os
import json
import numpy as np
import cv2
from sklearn.cluster import KMeans
from utils import save_video


# Farbnamen -> RGB-Referenzwerte
_COLOR_RGB = {
    "white":       (255, 255, 255), "black":       (0,   0,   0  ),
    "gray":        (128, 128, 128), "light-gray":  (200, 200, 200),
    "dark-gray":   (64,  64,  64 ), "charcoal":    (50,  50,  50 ),
    "silver":      (180, 180, 180), "ivory":       (240, 235, 210),
    "red":         (200, 30,  30 ), "dark-red":    (120, 0,   0  ),
    "crimson":     (180, 0,   40 ), "maroon":      (128, 0,   0  ),
    "wine":        (100, 0,   40 ), "coral":       (255, 100, 80 ),
    "salmon":      (250, 128, 114), "blue":        (30,  80,  200),
    "dark-blue":   (10,  20,  100), "light-blue":  (100, 160, 240),
    "navy":        (0,   0,   128), "sky-blue":    (80,  160, 240),
    "indigo":      (60,  0,   130), "green":       (30,  160, 30 ),
    "dark-green":  (0,   80,  0  ), "light-green": (100, 200, 100),
    "lime":        (0,   240, 0  ), "olive":       (128, 128, 0  ),
    "mint":        (160, 240, 200), "yellow":      (240, 220, 0  ),
    "gold":        (210, 170, 0  ), "orange":      (240, 120, 0  ),
    "brown":       (120, 60,  20 ), "tan":         (180, 140, 100),
    "beige":       (220, 200, 160), "khaki":       (180, 160, 100),
    "peach":       (255, 180, 140), "purple":      (120, 0,   160),
    "violet":      (140, 0,   200), "magenta":     (200, 0,   200),
    "lavender":    (180, 160, 240), "pink":        (240, 100, 160),
    "cyan":        (0,   200, 200), "teal":        (0,   128, 128),
    "turquoise":   (0,   180, 180),
}

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "teams_config.json")


def _rgb_to_lab(rgb):
    bgr = np.array([[list(reversed(rgb))]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


_NAME_LAB = {name: _rgb_to_lab(rgb) for name, rgb in _COLOR_RGB.items()}


class Debugger:
    """Erzeugt ein Debugging-Video mit Bounding-Boxes, Teamfarben und Kamerabewegung."""

    def __init__(self, video_path: str):
        self._player_cache = {}
        self._ref_lab      = {}
        self.team_colors   = {}
        self._kmeans       = None
        self._use_config   = False
        self._load_config(video_path)

    def _load_config(self, video_path):
        if not os.path.exists(_CONFIG_FILE):
            return
        try:
            with open(_CONFIG_FILE, encoding="utf-8") as f:
                cfg = json.load(f)
            key   = os.path.basename(video_path)
            entry = cfg.get(key) or next((v for v in cfg.values() if isinstance(v, dict)), None)
            if not entry:
                return
            for tid in (1, 2):
                t = entry.get(f"team_{tid}", {})
                pc = str(t.get("player_color", "")).lower()
                if pc in _NAME_LAB:
                    self._ref_lab[f"team_{tid}"] = _NAME_LAB[pc]
                    self.team_colors[tid] = np.array(_COLOR_RGB[pc], dtype=np.float64)
                gc = str(t.get("goalkeeper_color", "")).lower()
                if gc in _NAME_LAB:
                    self._ref_lab[f"gk_{tid}"] = _NAME_LAB[gc]
            rc = str(entry.get("referee", "")).lower()
            if rc in _NAME_LAB:
                self._ref_lab["referee"] = _NAME_LAB[rc]
            self._use_config = bool(self._ref_lab)
        except Exception:
            pass

    def _sample_color(self, frame, bbox):
        x1, y1, x2, y2 = max(0,int(bbox[0])), max(0,int(bbox[1])), int(bbox[2]), int(bbox[3])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([128., 128., 128.])
        top = crop[: crop.shape[0] // 2]
        if top.size == 0:
            top = crop
        px  = top.reshape(-1, 3).astype(np.float64)
        km  = KMeans(n_clusters=2, n_init=1, random_state=0).fit(px)
        lbl = km.labels_.reshape(top.shape[0], top.shape[1])
        corners = [lbl[0,0], lbl[0,-1], lbl[-1,0], lbl[-1,-1]]
        bg = max(set(corners), key=corners.count)
        return km.cluster_centers_[1 - bg]

    def _classify(self, bgr):
        if self._use_config:
            bgr_u8 = np.clip(bgr, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            lab    = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
            best   = min(self._ref_lab, key=lambda r: np.linalg.norm(lab - self._ref_lab[r]))
            if best == "referee":
                return 0
            return int(best.split("_")[1])
        if self._kmeans is not None:
            return int(self._kmeans.predict(bgr.reshape(1, -1))[0]) + 1
        return 1

    def _assign_teams(self, video_frames, tracks):
        if not self._use_config:
            colors = [self._sample_color(video_frames[0], d["bbox"])
                      for d in tracks["players"][0].values()]
            if colors:
                km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0).fit(colors)
                self._kmeans = km
                self.team_colors = {1: km.cluster_centers_[0], 2: km.cluster_centers_[1]}

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                pid = int(player_id) if hasattr(player_id, "__int__") else player_id
                if pid not in self._player_cache:
                    bgr = self._sample_color(video_frames[frame_num], track["bbox"])
                    team_id = self._classify(bgr)
                    # 0 = Schiedsrichter-Klassifikation für einen Spieler → Teamfarben direkt vergleichen
                    if team_id == 0 and len(self._ref_lab) >= 2:
                        bgr_u8 = np.clip(bgr, 0, 255).astype(np.uint8).reshape(1, 1, 3)
                        lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
                        team_refs = {k: v for k, v in self._ref_lab.items() if k.startswith("team_")}
                        if team_refs:
                            best = min(team_refs, key=lambda r: np.linalg.norm(lab - team_refs[r]))
                            team_id = int(best.split("_")[1])
                        else:
                            team_id = 1
                    self._player_cache[pid] = team_id
                team = self._player_cache[pid]
                tracks["players"][frame_num][player_id]["team"]       = team
                tracks["players"][frame_num][player_id]["team_color"] = (
                    self.team_colors.get(team, (128, 128, 128))
                )

    def generate(self, video_frames, tracks, team_ball_control, tracker, cam, cam_mv,
                 out_path: str, fps: float = 24):
        self._assign_teams(video_frames, tracks)
        frames = [f.copy() for f in video_frames]
        frames = tracker.draw_annotations(frames, tracks, team_ball_control)
        frames = cam.draw_camera_movement(frames, cam_mv)
        save_video(frames, out_path, fps=fps)
