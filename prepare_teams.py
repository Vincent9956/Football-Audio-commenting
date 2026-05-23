# prepare_teams.py – Trikotfarben per CIE-Lab-Distanz klassifizieren und Stub aktualisieren
# Nutzung: python prepare_teams.py

import argparse
import json
import os
import pickle
import numpy as np
import cv2
from collections import defaultdict


DEFAULT_STUB   = "stubs/track_stubs.pkl"
DEFAULT_CONFIG = "teams_config.json"

COLOR_NAMES_RGB = {
    "white":       (255, 255, 255),
    "black":       (0,   0,   0  ),
    "gray":        (128, 128, 128),
    "light-gray":  (200, 200, 200),
    "dark-gray":   (64,  64,  64 ),
    "charcoal":    (50,  50,  50 ),
    "silver":      (180, 180, 180),
    "red":         (200, 30,  30 ),
    "dark-red":    (120, 0,   0  ),
    "crimson":     (180, 0,   40 ),
    "maroon":      (128, 0,   0  ),
    "blue":        (30,  80,  200),
    "dark-blue":   (10,  20,  100),
    "light-blue":  (100, 160, 240),
    "navy":        (0,   0,   128),
    "sky-blue":    (80,  160, 240),
    "green":       (30,  160, 30 ),
    "dark-green":  (0,   80,  0  ),
    "light-green": (100, 200, 100),
    "yellow":      (240, 220, 0  ),
    "gold":        (210, 170, 0  ),
    "orange":      (240, 120, 0  ),
    "brown":       (120, 60,  20 ),
    "purple":      (120, 0,   160),
    "pink":        (240, 100, 160),
    "cyan":        (0,   200, 200),
    "teal":        (0,   128, 128),
    "turquoise":   (0,   180, 180),
}


def _rgb_to_lab(rgb: tuple) -> np.ndarray:
    bgr = np.array([[list(reversed(rgb))]], dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


def _bgr_to_lab(bgr_list: list) -> np.ndarray:
    bgr = np.clip(np.array(bgr_list), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


_NAME_LAB = {name: _rgb_to_lab(rgb) for name, rgb in COLOR_NAMES_RGB.items()}


def load_config(config_path: str, video_key: str) -> dict:
    """Liest den Eintrag für video_key aus teams_config.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"teams_config.json nicht gefunden: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if video_key not in cfg:
        raise KeyError(
            f"Kein Eintrag für '{video_key}' in {config_path}.\n"
            f"Verfügbare Schlüssel: {list(cfg.keys())}"
        )
    return cfg[video_key]


def build_ref_labs(entry: dict) -> dict:
    """
    Erstellt Lab-Referenzvektoren aus dem Config-Eintrag.
    Gibt {role: Lab-Array} zurück, z.B.:
      {"team_1": ..., "gk_1": ..., "team_2": ..., "gk_2": ..., "referee": ...}
    """
    refs = {}
    for tid in (1, 2):
        key = f"team_{tid}"
        t   = entry.get(key, {})

        pc = str(t.get("player_color", "")).lower()
        if pc in _NAME_LAB:
            refs[f"team_{tid}"] = _NAME_LAB[pc]
        else:
            print(f"  WARNUNG: Unbekannte player_color '{pc}' für team_{tid}.")

        gc = str(t.get("goalkeeper_color", "")).lower()
        if gc in _NAME_LAB:
            refs[f"gk_{tid}"] = _NAME_LAB[gc]
        elif gc:
            print(f"  WARNUNG: Unbekannte goalkeeper_color '{gc}' für team_{tid}.")

    rc = str(entry.get("referee", "")).lower()
    if rc in _NAME_LAB:
        refs["referee"] = _NAME_LAB[rc]
    elif rc:
        print(f"  WARNUNG: Unbekannte referee-Farbe '{rc}'.")

    return refs


def classify_player(bgr: list, ref_labs: dict) -> str:
    """Gibt die Rolle mit minimaler Lab-Distanz zurück."""
    lab   = _bgr_to_lab(bgr)
    dists = {role: float(np.linalg.norm(lab - ref)) for role, ref in ref_labs.items()}
    return min(dists, key=dists.get)


def collect_avg_colors(tracks: dict) -> dict:
    """Mittlere jersey_color_bgr pro track_id."""
    accum = defaultdict(list)
    for frame_players in tracks.get("players", []):
        for tid, info in frame_players.items():
            color = info.get("jersey_color_bgr")
            if color is not None:
                accum[tid].append(color)
    return {tid: list(np.mean(colors, axis=0)) for tid, colors in accum.items()}


def print_summary(assignments: dict, ref_labs: dict, entry: dict):
    """Lesbare Ausgabe der Zuordnung."""
    from collections import defaultdict
    groups = defaultdict(list)
    for tid, role in assignments.items():
        groups[role].append(tid)

    # Reihenfolge: team_1, team_2, gk_1, gk_2, referee
    order = ["team_1", "team_2", "gk_1", "gk_2", "referee"]

    print("\n" + "=" * 58)
    print("  TEAM-KLASSIFIKATION  (aus teams_config.json)")
    print("=" * 58)

    for role in order:
        if role not in groups and role not in ref_labs:
            continue
        tids    = sorted(groups.get(role, []))
        ids_str = ", ".join(f"#{t}" for t in tids) if tids else "–"

        # Farbnamen aus Config ermitteln
        if role == "team_1":
            color_name = entry.get("team_1", {}).get("player_color", "?")
            label = "Team 1  (Feldspieler)"
        elif role == "team_2":
            color_name = entry.get("team_2", {}).get("player_color", "?")
            label = "Team 2  (Feldspieler)"
        elif role == "gk_1":
            color_name = entry.get("team_1", {}).get("goalkeeper_color", "?")
            label = "Team 1  (Torwart)"
        elif role == "gk_2":
            color_name = entry.get("team_2", {}).get("goalkeeper_color", "?")
            label = "Team 2  (Torwart)"
        else:
            color_name = entry.get("referee", "?")
            label = "Schiedsrichter"

        print(f"\n  {role:<10}  {label}")
        print(f"    Config-Farbe : {color_name}")
        print(f"    Track-IDs    : {ids_str}  ({len(tids)} Spieler)")

    print("\n" + "=" * 58)
    print("  Nächster Schritt: python readpkl.py")
    print("=" * 58 + "\n")


def write_labels(tracks: dict, assignments: dict):
    """Schreibt team_label in jeden Spieler-Eintrag des Stubs."""
    for frame_players in tracks.get("players", []):
        for tid, info in frame_players.items():
            info["team_label"] = assignments.get(tid, "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub",   default=DEFAULT_STUB)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--video",  default=None,
                        help="Video-Dateiname als Schlüssel in teams_config.json "
                             "(Standard: erster verfügbarer Eintrag)")
    args = parser.parse_args()

    # ── Config laden ─────────────────────────────────────────────────────────
    with open(args.config, encoding="utf-8") as f:
        full_cfg = json.load(f)

    video_key = args.video or next(
        (k for k, v in full_cfg.items() if isinstance(v, dict)), None
    )
    if video_key is None:
        print("[prepare_teams] FEHLER: Kein gültiger Video-Eintrag in teams_config.json gefunden.")
        return
    print(f"[prepare_teams] Config-Schlüssel: '{video_key}'")

    try:
        entry = load_config(args.config, video_key)
    except (FileNotFoundError, KeyError) as e:
        print(f"[prepare_teams] FEHLER: {e}")
        return

    ref_labs = build_ref_labs(entry)
    if not ref_labs:
        print("[prepare_teams] FEHLER: Keine gültigen Farben in teams_config.json.")
        return
    print(f"[prepare_teams] Referenzfarben geladen: {list(ref_labs.keys())}")

    # ── Stub laden ───────────────────────────────────────────────────────────
    print(f"[prepare_teams] Lese Stub: {args.stub}")
    with open(args.stub, "rb") as f:
        tracks = pickle.load(f)

    avg_colors = collect_avg_colors(tracks)
    if not avg_colors:
        print("[prepare_teams] FEHLER: Keine jersey_color_bgr im Stub.")
        print("  → Bitte zuerst main.py ausführen.")
        return
    print(f"[prepare_teams] {len(avg_colors)} Spieler-IDs klassifizieren ...")

    # ── Klassifikation per Lab-Distanz ───────────────────────────────────────
    assignments = {
        tid: classify_player(color, ref_labs)
        for tid, color in avg_colors.items()
    }

    print_summary(assignments, ref_labs, entry)

    # ── Stub speichern ───────────────────────────────────────────────────────
    write_labels(tracks, assignments)
    with open(args.stub, "wb") as f:
        pickle.dump(tracks, f)
    print(f"[prepare_teams] team_label in Stub gespeichert: {args.stub}\n")


if __name__ == "__main__":
    main()
