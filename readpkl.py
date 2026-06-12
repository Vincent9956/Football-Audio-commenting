import os
import sys
import glob
import json
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(__file__))

def _find_video():
    for p in sorted(glob.glob("input_videos/*")):
        if p.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            return p
    return "input_videos/input.mp4"

VIDEO_PATH   = _find_video()
TRACK_STUB   = "stubs/track_stubs.pkl"
CAMERA_STUB  = "stubs/camera_movement_stub.pkl"
PA_STUB      = "stubs/penalty_area_stub.pkl"
MODEL_PATH   = "models/best.pt"
EVENTS_OUT   = "events.json"
TEAMS_CONFIG = "teams_config.json"

FPS              = 24     # default; overridden at runtime from video metadata
VOTE_WINDOW      = 12     # possession smoothing half-window
MIN_HOLD_FRAMES  = 24     # min frames for valid possession (~0.27s @ 30fps)
SHOT_VEL         = 500   # px/frame raw ball speed → Schuss
SHOT_GAP_DIST    = 350   # px Gesamtverschiebung über Tracking-Lücke → Schuss
SHOT_GAP_MIN     = 4     # min Frames Lücke bevor Gap-Detektion anspringt
SHOT_COOLDOWN    = 4.0    # s between SHOT events
PRESS_RADIUS     = 250   # px radius for opponent proximity
PRESS_COOLDOWN   = 4.0    # s between PRESS events
PRESS_BALL_DIST  = 100    # px max ball-to-holder dist at fire time (sanity check)
MIN_EVENT_TS     = 0.0    # s - ignore events in first seconds (tracking warmup)
FRAME_WIDTH      = 1920   # overridden at runtime from video metadata
FRAME_HEIGHT     = 1080   # overridden at runtime from video metadata
DANGER_ZONE_FRAC = 0.48   # innermost fraction of frame = near goal/penalty area

POSSESSION_MIN_SEC    = 1.2   # min seconds of team possession to count
POSSESSION_MIN_PASSES = 1    # min intra-team player changes (passes)
POSSESSION_COOLDOWN   = 0.0   # s between POSSESSION_RUN events per team

PASS_COOLDOWN    = 1.0    # s between PASS events per team

PRIORITY_SPEED = {5: 1.5, 4: 1.25, 3: 1.1, 2: 1.0, 1: 1.0}


# ═══════════════════════════════════════════════════════════════════
#  1. TEAM ASSIGNMENT  (liest team_label aus Stub)
# ═══════════════════════════════════════════════════════════════════
def _apply_team_labels(tracks: dict):

    label_to_id = {"team_1": 1, "gk_1": 1, "team_2": 2, "gk_2": 2}
    found = False
    for frame_players in tracks.get("players", []):
        for info in frame_players.values():
            if info.get("team_label"):
                found = True
            info["team"] = label_to_id.get(info.get("team_label"), 0)

    if not found:
        print("\n[Team] FEHLER: Kein team_label im Stub gefunden.")
        print("  → Bitte zuerst prepare_teams.py ausführen:")
        print("      python prepare_teams.py")
        print("  → Danach teams_config.json prüfen und readpkl.py erneut starten.\n")
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════
#  2. ENRICHMENT  (kein read_video – alle Daten aus Stubs)
# ═══════════════════════════════════════════════════════════════════
def build_tracks():
    from object_tracker import ObjectTracker
    from camera_movement_estimator import CameraMovementEstimator
    from penalty_area_detector import PenaltyAreaDetector

    print("Loading tracking stubs ...")
    tracker = ObjectTracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        frames=None, read_from_stub=True, stub_path=TRACK_STUB)
    tracker.add_position_to_tracks(tracks)

    print("Camera movement (from stub) ...")
    cam = CameraMovementEstimator()
    mv  = cam.get_camera_movement(
        frames=None, read_from_stub=True, stub_path=CAMERA_STUB)
    cam.add_adjust_positions_to_tracks(tracks, mv)

    print("Ball interpolation ...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("Team assignment (from stub team_label) ...")
    _apply_team_labels(tracks)

    print("Penalty area data (from stub) ...")
    pad     = PenaltyAreaDetector()
    pa_data = pad.get_penalty_area_data(
        video_frames=None, read_from_stub=True, stub_path=PA_STUB)
    pad.add_penalty_area_to_tracks(tracks, pa_data)

    print("Enrichment done – " + str(len(tracks.get("players", []))) + " frames.\n")
    return tracks


# ═══════════════════════════════════════════════════════════════════
#  2. RAW BALL VELOCITY
# ═══════════════════════════════════════════════════════════════════
def _raw_ball_velocity(tracks):
    import pickle
    with open(TRACK_STUB, "rb") as f:
        raw_tracks = pickle.load(f)

    raw_ball = raw_tracks.get("ball", [])
    n = len(raw_ball)
    vel     = [None] * n
    vel_x   = [None] * n
    prev_fi, prev_cx, prev_cy = None, None, None

    for fi in range(n):
        b = raw_ball[fi]
        if b and 1 in b:
            bb = b[1]["bbox"]
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            if prev_fi is not None and fi - prev_fi <= 3:
                gap = fi - prev_fi
                dx = (cx - prev_cx) / gap
                dy = (cy - prev_cy) / gap
                vel[fi]   = (dx ** 2 + dy ** 2) ** 0.5
                vel_x[fi] = dx
            prev_fi, prev_cx, prev_cy = fi, cx, cy

    # Gap-Detektion: Ball verschwindet > SHOT_GAP_MIN Frames und taucht weit entfernt
    # wieder auf → typisches Muster bei Schüssen (Tracker verliert den schnellen Ball).
    # Vergleicht RAW-Positionen vor und nach der Lücke (nicht interpolierte Frames).
    vel_gap   = [None] * n
    vel_gap_x = [None] * n
    prev_gfi, prev_gcx, prev_gcy = None, None, None
    for fi in range(n):
        b = raw_ball[fi]
        if b and 1 in b:
            bb = b[1]["bbox"]
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            if prev_gfi is not None:
                gap = fi - prev_gfi
                if gap > SHOT_GAP_MIN:
                    dx    = cx - prev_gcx
                    dy    = cy - prev_gcy
                    dist  = (dx ** 2 + dy ** 2) ** 0.5
                    if dist > SHOT_GAP_DIST:
                        vel_gap[fi]   = dist
                        vel_gap_x[fi] = dx
            prev_gfi, prev_gcx, prev_gcy = fi, cx, cy
        # kein else-Reset – wir behalten prev durch die Lücke

    # Reversal-Filter: Wenn auf eine Gap-Detektion in Richtung A innerhalb
    # von REVERSAL_WIN Frames eine Gap-Detektion in entgegengesetzter Richtung
    # folgt, ist es sehr wahrscheinlich ein Tracking-Artefakt (Ball-ID-Wechsel).
    # Solche Falschdetektionen werden unterdrückt.
    REVERSAL_WIN = 15   # Frames vorausschauen
    MIN_REVERSE_DX = 200  # Mindest-dx der Gegenbewegung (vermeidet Überunterdrückung)
    for fi in range(n):
        if vel_gap[fi] is None:
            continue
        dx = vel_gap_x[fi]
        for k in range(1, REVERSAL_WIN + 1):
            j = fi + k
            if j < n and vel_gap_x[j] is not None:
                counter = (dx > 0 and vel_gap_x[j] < -MIN_REVERSE_DX) or \
                          (dx < 0 and vel_gap_x[j] > MIN_REVERSE_DX)
                if counter:
                    vel_gap[fi]   = None
                    vel_gap_x[fi] = None
                    break

    return vel, vel_x, vel_gap, vel_gap_x


# ═══════════════════════════════════════════════════════════════════
#  3. POSSESSION
# ═══════════════════════════════════════════════════════════════════
def compute_possession(tracks):
    from player_ball_assigner import PlayerBallAssigner
    assigner = PlayerBallAssigner()

    ball_vel = [None] * len(tracks["ball"])
    prev_cx, prev_cy = None, None
    for fi, b in enumerate(tracks["ball"]):
        if 1 in b:
            bb = b[1]["bbox"]
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            if prev_cx is not None:
                ball_vel[fi] = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            prev_cx, prev_cy = cx, cy
        else:
            prev_cx, prev_cy = None, None

    BALL_IN_FLIGHT_VEL = 18  # px/frame - above this the ball is a pass/shot

    raw = []
    for fi in range(len(tracks["players"])):
        # Suppress possession if ball is moving fast (in flight)
        if ball_vel[fi] is not None and ball_vel[fi] > BALL_IN_FLIGHT_VEL:
            raw.append(None)
            continue
        # Ball not detected in this frame (interpolation gap)
        if fi >= len(tracks["ball"]) or 1 not in tracks["ball"][fi]:
            raw.append(None)
            continue
        ball_bbox = tracks["ball"][fi][1]["bbox"]
        pid = assigner.assign_ball_to_player(tracks["players"][fi], ball_bbox)
        raw.append(pid if pid != -1 else None)
    return raw


def smooth_possession(raw):
    n, half, out = len(raw), VOTE_WINDOW, []
    for i in range(n):
        seg = raw[max(0, i - half): min(n, i + half + 1)]
        valid = [p for p in seg if p is not None]
        if not valid:
            out.append(None)
            continue
        top, cnt = Counter(valid).most_common(1)[0]
        out.append(top if cnt > len(seg) // 2 else None)
    return out


def _runs(possession):
    """Return [(start, end_inclusive, player_id|None)]."""
    runs, i, n = [], 0, len(possession)
    while i < n:
        p, j = possession[i], i
        while j < n and possession[j] == p:
            j += 1
        runs.append((i, j - 1, p))
        i = j
    return runs


def _team(tracks, frame, player_id):
    fd = tracks["players"][frame]
    return fd[player_id].get("team") if player_id in fd else None


# ═══════════════════════════════════════════════════════════════════
#  4. TEAM CONFIG
# ═══════════════════════════════════════════════════════════════════
def _load_video_config():
    if not os.path.exists(TEAMS_CONFIG):
        return {}
    with open(TEAMS_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get(os.path.basename(VIDEO_PATH), {})


def load_team_halves():
    vcfg = _load_video_config()
    halves = {}
    for tid in (1, 2):
        h = vcfg.get("team_" + str(tid), {}).get("half", "").lower()
        if h in ("left", "right"):
            halves[tid] = h
            direction = "links" if h == "left" else "rechts"
            print("  Team " + str(tid) + ": greift nach " + direction)
        else:
            print("  Team " + str(tid) + ": kein gueltiges 'half'.")
    return halves


def load_team_context():
    vcfg = _load_video_config()
    context = {}
    for tid in (1, 2):
        key = "team_" + str(tid)
        if key in vcfg:
            tc = vcfg[key]
            context[str(tid)] = {
                "player_color":     tc.get("player_color", ""),
                "goalkeeper_color": tc.get("goalkeeper_color", ""),
                "attacks":          tc.get("half", ""),
            }
    if "referee" in vcfg:
        context["referee"] = {"color": vcfg["referee"]}
    return context


# ═══════════════════════════════════════════════════════════════════
#  5. HELPERS
# ═══════════════════════════════════════════════════════════════════
def _field_info(tracks, frame, player_id, team, team_halves):
    if frame >= len(tracks["players"]):
        return None, None
    fd = tracks["players"][frame]
    if player_id not in fd:
        return None, None
    bbox = fd[player_id].get("bbox", [])
    if not bbox:
        return None, None
    cx = (bbox[0] + bbox[2]) / 2
    field_side = "left" if cx < FRAME_WIDTH / 2 else "right"
    half = team_halves.get(team)
    attacking = (field_side == half) if half else None
    return field_side, attacking


def _ball_field_info(tracks, frame, team, team_halves):
    if frame >= len(tracks["ball"]):
        return None, None
    b = tracks["ball"][frame]
    if 1 not in b:
        return None, None
    bb = b[1]["bbox"]
    cx = (bb[0] + bb[2]) / 2
    field_side = "left" if cx < FRAME_WIDTH / 2 else "right"
    half = team_halves.get(team)
    attacking = (field_side == half) if half else None
    return field_side, attacking


def _compute_flank(y_center, attack_direction):
    if not attack_direction or y_center is None:
        return None
    in_top = y_center < FRAME_HEIGHT / 2
    if attack_direction == "right":
        return "left" if in_top else "right"
    else:   # attacks "left"
        return "right" if in_top else "left"


def _ball_y(tracks, frame):

    if frame >= len(tracks["ball"]):
        return None
    b = tracks["ball"][frame]
    if 1 not in b:
        return None
    bb = b[1]["bbox"]
    return (bb[1] + bb[3]) / 2


def _player_y(tracks, frame, player_id):
    if frame >= len(tracks["players"]):
        return None
    fd = tracks["players"][frame]
    if player_id not in fd:
        return None
    bbox = fd[player_id].get("bbox", [])
    if not bbox:
        return None
    return (bbox[1] + bbox[3]) / 2


# ═══════════════════════════════════════════════════════════════════
#  6. EVENT DETECTORS
# ═══════════════════════════════════════════════════════════════════
def _detect_shots(possession, vel, vel_x, vel_gap, vel_gap_x,
                  tracks, team_halves):
    events = []
    last_shot = -SHOT_COOLDOWN * 2
    had_poss = False
    last_team = None

    def _try_shot(fi, v, vx, ts):
        nonlocal last_shot, had_poss
        half = team_halves.get(last_team) if last_team else None
        if half == "left":
            toward_goal = (vx is not None and vx < 0)
        elif half == "right":
            toward_goal = (vx is not None and vx > 0)
        else:
            toward_goal = True

        if toward_goal and ts - last_shot >= SHOT_COOLDOWN:
            field_side, attacking = _ball_field_info(
                tracks, fi, last_team, team_halves)
            flank = _compute_flank(
                _ball_y(tracks, fi), team_halves.get(last_team))
            events.append({
                "ts":         round(ts, 3),
                "type":       "SHOT",
                "team":       last_team,
                "field_side": field_side,
                "attacking":  attacking,
                "flank":      flank,
                "priority":   5,
            })
            last_shot = ts
            had_poss = False

    for fi in range(len(possession)):
        ts = fi / FPS
        if possession[fi] is not None:
            had_poss = True
            last_team = _team(tracks, fi, possession[fi])
            continue

        # --- Raw velocity (high threshold) ---
        v  = vel[fi]
        vx = vel_x[fi]
        if v is not None and v > SHOT_VEL and had_poss:
            _try_shot(fi, v, vx, ts)

        # --- Gap velocity (Ball taucht nach langer Lücke weit entfernt auf) ---
        elif vel_gap[fi] is not None and had_poss:
            _try_shot(fi, vel_gap[fi], vel_gap_x[fi], ts)

        if possession[fi] is None and fi > 0 and fi % int(1.5 * FPS) == 0:
            had_poss = False

    return events


def _detect_possession_runs(raw_poss, tracks, team_halves):
    n = len(raw_poss)
    TOLERANCE = int(FPS * 0.7)  # max consecutive None frames inside a run
    min_frames = int(POSSESSION_MIN_SEC * FPS)
    events = []

    # Build raw team-per-frame (None when ball free / in flight)
    raw_team = []
    for fi, pid in enumerate(raw_poss):
        if pid is None:
            raw_team.append(None)
        else:
            t = _team(tracks, fi, pid)
            raw_team.append(t if (t and t != 0) else None)

    cooldown = {}
    fi = 0
    while fi < n:
        t = raw_team[fi]
        if t is None:
            fi += 1
            continue
        run_start = fi
        none_streak = 0
        j = fi + 1
        while j < n:
            if raw_team[j] == t:
                none_streak = 0
                j += 1
            elif raw_team[j] is None:
                none_streak += 1
                if none_streak > TOLERANCE:
                    break
                j += 1
            else:            # different team -> hard break
                break

        run_end = j
        while run_end > run_start and raw_team[run_end - 1] != t:
            run_end -= 1

        # Count how many frames this team actually had the ball
        team_frames = sum(1 for k in range(run_start, run_end) if raw_team[k] == t)

        if team_frames >= min_frames:
            # Count player transitions (= passes) within the run
            passes = 0
            last_pid = None
            for k in range(run_start, run_end):
                pid = raw_poss[k]
                if pid is None or raw_team[k] != t:
                    continue
                if last_pid is not None and pid != last_pid:
                    passes += 1
                last_pid = pid

            if passes >= POSSESSION_MIN_PASSES:
                # Fire once we've seen min_frames of this team's possession
                count = 0
                fire_frame = run_start
                for k in range(run_start, run_end):
                    if raw_team[k] == t:
                        count += 1
                    if count >= min_frames:
                        fire_frame = k
                        break

                ts = round(fire_frame / FPS, 3)
                last_ts = cooldown.get(t, -POSSESSION_COOLDOWN * 2)

                if ts - last_ts >= POSSESSION_COOLDOWN:
                    holder = raw_poss[fire_frame]
                    # Scan backwards if fire_frame landed on a None
                    if holder is None:
                        for k in range(fire_frame - 1, run_start - 1, -1):
                            if raw_poss[k] is not None and raw_team[k] == t:
                                holder = raw_poss[k]
                                fire_frame = k
                                break

                    field_side, attacking = None, None
                    flank = None
                    if holder:
                        field_side, attacking = _field_info(
                            tracks, fire_frame, holder, t, team_halves)
                        flank = _compute_flank(
                            _ball_y(tracks, fire_frame), team_halves.get(t))

                    events.append({
                        "ts":         ts,
                        "type":       "POSSESSION_RUN",
                        "team":       t,
                        "passes":     passes,
                        "duration":   round(team_frames / FPS, 1),
                        "field_side": field_side,
                        "attacking":  attacking,
                        "flank":      flank,
                        "priority":   3,
                    })
                    cooldown[t] = ts

        fi = run_end if run_end > fi else fi + 1

    return events


def _detect_danger(possession, tracks, team_halves):
    if not team_halves:
        return []

    events = []
    runs = _runs(possession)
    seen = {}

    for start, end, holder in runs:
        if holder is None:
            continue
        if (end - start + 1) < MIN_HOLD_FRAMES:
            continue

        team = _team(tracks, start, holder)
        if team == 0 or team is None:
            continue

        half = team_halves.get(team)
        if half is None:
            continue

        # Use ball position to check danger zone
        b = tracks["ball"][start] if start < len(tracks["ball"]) else {}
        if 1 not in b:
            continue
        bb = b[1]["bbox"]
        ball_cx = (bb[0] + bb[2]) / 2
        ball_rel = ball_cx / FRAME_WIDTH   # 0.0 = left edge, 1.0 = right edge

        if half == "left":
            in_danger = ball_rel < DANGER_ZONE_FRAC        # near left goal
            pa_side = "left"
        elif half == "right":
            in_danger = ball_rel > (1.0 - DANGER_ZONE_FRAC)  # near right goal
            pa_side = "right"
        else:
            continue

        if not in_danger:
            continue

        ts = round(start / FPS, 3)
        last = seen.get(holder, -30.0)
        if ts - last < 4.0:
            continue

        flank = _compute_flank(_ball_y(tracks, start), team_halves.get(team))
        events.append({
            "ts":         ts,
            "type":       "DANGER",
            "player":     holder,
            "team":       team,
            "field_side": pa_side,
            "attacking":  True,
            "flank":      flank,
            "priority":   4,
        })
        seen[holder] = ts

    return events



def _detect_press(possession, raw_poss, tracks, team_halves):
    events = []
    last_press = -PRESS_COOLDOWN * 2
    run_start = None
    run_holder = None
    RUN_MIN = 12

    n = len(possession)
    for fi in range(n):
        holder = possession[fi]
        # Wenn smooth=None, Fallback auf raw possession
        # (z.B. bei fragmentiertem Besitz an der Zeitstelle 18 s)
        if holder is None:
            holder = raw_poss[fi]
        if holder is None:
            run_start = None
            continue

        # Raw possession must NOT point to a different player
        # (if it does, the smooth assignment is a lag artifact)
        rp = raw_poss[fi]
        if rp is not None and rp != holder:
            run_start = None
            continue

        fd = tracks["players"][fi]
        if holder not in fd:
            run_start = None
            continue

        h_team = fd[holder].get("team")
        if not h_team or h_team == 0:   # Schiedsrichter/Unbekannte ignorieren
            run_start = None
            continue

        h_bbox = fd[holder].get("bbox", [])
        if not h_bbox:
            run_start = None
            continue

        hx = (h_bbox[0] + h_bbox[2]) / 2
        hy = (h_bbox[1] + h_bbox[3]) / 2

        opp_close = 0
        for pid, pdata in fd.items():
            if pid == holder:
                continue
            if pdata.get("team") == h_team or pdata.get("team") == 0:
                continue
            bb = pdata.get("bbox", [])
            if not bb:
                continue
            px = (bb[0] + bb[2]) / 2
            py = (bb[1] + bb[3]) / 2
            if ((px - hx) ** 2 + (py - hy) ** 2) ** 0.5 <= PRESS_RADIUS:
                opp_close += 1

        if opp_close >= 2:
            if run_start is None or run_holder != holder:
                run_start = fi
                run_holder = holder
        else:
            run_start = None

        if run_start is not None and (fi - run_start) >= RUN_MIN:
            ts = fi / FPS
            if ts - last_press >= PRESS_COOLDOWN:
                # Final sanity check: ball must be close to holder right now
                ball_ok = False
                b_frame = tracks["ball"][fi] if fi < len(tracks["ball"]) else {}
                if 1 in b_frame:
                    bbb = b_frame[1]["bbox"]
                    bcx = (bbb[0] + bbb[2]) / 2
                    bcy = (bbb[1] + bbb[3]) / 2
                    ball_ok = ((bcx - hx) ** 2 + (bcy - hy) ** 2) ** 0.5 < PRESS_BALL_DIST

                if ball_ok:
                    field_side, attacking = _field_info(
                        tracks, fi, holder, h_team, team_halves)
                    flank = _compute_flank(
                        _player_y(tracks, fi, holder), team_halves.get(h_team))
                    events.append({
                        "ts":         round(ts, 3),
                        "type":       "PRESS",
                        "player":     holder,
                        "team":       h_team,
                        "field_side": field_side,
                        "attacking":  attacking,
                        "flank":      flank,
                        "priority":   3,
                    })
                    last_press = ts
                run_start = None

    return events


def _detect_passes(raw_poss, possession, tracks, team_halves):
    events = []
    last_pass_ts = {}   # team -> last fired ts

    def _try_fire(fi, pid, team):
        ts = round(fi / FPS, 3)
        prev = last_pass_ts.get(team, -PASS_COOLDOWN * 2)
        if ts - prev >= PASS_COOLDOWN:
            fs, atk = _field_info(tracks, fi, pid, team, team_halves)
            flank = _compute_flank(
                _player_y(tracks, fi, pid), team_halves.get(team))
            events.append({
                "ts":         ts,
                "type":       "PASS",
                "player":     pid,
                "team":       team,
                "field_side": fs,
                "attacking":  atk,
                "flank":      flank,
                "priority":   1,
            })
            last_pass_ts[team] = ts

    # ── Source A: raw_poss player transitions ────────────────────────
    last_raw = {}   # team -> (pid, frame_index)
    for fi, pid in enumerate(raw_poss):
        if pid is None:
            continue
        team = _team(tracks, fi, pid)
        if not team or team == 0:
            continue
        if team in last_raw:
            prev_pid, prev_fi = last_raw[team]
            if prev_pid != pid and (fi - prev_fi) <= int(FPS * 1.5):
                _try_fire(fi, pid, team)
        last_raw[team] = (pid, fi)

    # ── Source B: smooth_poss same-team player transitions ───────────
    last_smooth = {}    # team -> pid
    for fi, pid in enumerate(possession):
        if pid is None:
            last_smooth.clear()     # hard gap resets smooth tracking
            continue
        team = _team(tracks, fi, pid)
        if not team or team == 0:
            continue
        if team in last_smooth and last_smooth[team] != pid:
            _try_fire(fi, pid, team)
        last_smooth[team] = pid

    events.sort(key=lambda e: e["ts"])
    return events


def _detect_ball_won(possession, tracks, team_halves):
    events = []
    runs = _runs(possession)
    filt = [(s, e, p) for s, e, p in runs
            if p is None or (e - s + 1) >= MIN_HOLD_FRAMES]

    for idx in range(1, len(filt)):
        s, e, holder = filt[idx]
        if holder is None:
            continue
        ps, pe, prev_h = filt[idx - 1]
        if prev_h is None:
            continue
        prev_team = _team(tracks, pe, prev_h)
        curr_team = _team(tracks, s, holder)
        if (prev_team is not None and curr_team is not None
                and prev_team != curr_team
                and prev_team != 0 and curr_team != 0):
            field_side, attacking = _field_info(
                tracks, s, holder, curr_team, team_halves)
            flank = _compute_flank(
                _player_y(tracks, s, holder), team_halves.get(curr_team))
            events.append({
                "ts":         round(s / FPS, 3),
                "type":       "BALL_WON",
                "player":     holder,
                "team":       curr_team,
                "field_side": field_side,
                "attacking":  attacking,
                "flank":      flank,
                "priority":   3,
            })

    return events


# ═══════════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    global FPS, FRAME_WIDTH, FRAME_HEIGHT

    # ── Auto-detect FPS and frame size from video ──────────────────
    try:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(VIDEO_PATH)
        _detected_fps = _cap.get(_cv2.CAP_PROP_FPS)
        _detected_w   = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        _detected_h   = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()
        if _detected_fps and _detected_fps > 0:
            FPS = _detected_fps
        if _detected_w > 0 and _detected_h > 0:
            FRAME_WIDTH  = _detected_w
            FRAME_HEIGHT = _detected_h
    except Exception as _e:
        print("WARNING: Video metadata detection failed (" + str(_e) + "), using defaults")

    tracks = build_tracks()

    print("Computing possession ...")
    raw_poss = compute_possession(tracks)
    possession = smooth_possession(raw_poss)

    none_pct = 100 * possession.count(None) / len(possession)
    teams = Counter(
        tracks["players"][fi][h].get("team")
        for fi, h in enumerate(possession)
        if h is not None and h in tracks["players"][fi]
    )
    print("  Ball free : " + str(round(none_pct, 1)) + "%")
    for tm, cnt in sorted(teams.items()):
        total = sum(teams.values())
        print("  Team " + str(tm) + "    : " + str(round(100 * cnt / total, 1)) + "%")
    raw_vel, raw_vel_x, raw_vel_gap, raw_vel_gap_x = _raw_ball_velocity(tracks)
    hi_vel = [(fi, round(v, 1)) for fi, v in enumerate(raw_vel)
              if v and v > SHOT_VEL]

    team_halves = load_team_halves()
    team_context = load_team_context()
    print()

    print("Detecting events ...")
    events = (
        _detect_shots(possession, raw_vel, raw_vel_x, raw_vel_gap, raw_vel_gap_x, tracks, team_halves)
        + _detect_possession_runs(raw_poss, tracks, team_halves)
        + _detect_danger(possession, tracks, team_halves)
        + _detect_press(possession, raw_poss, tracks, team_halves)
        + _detect_ball_won(possession, tracks, team_halves)
        + _detect_passes(raw_poss, possession, tracks, team_halves)
    )
    events.sort(key=lambda e: (e["ts"], -e["priority"]))

    before = len(events)
    events = [e for e in events if e["ts"] >= MIN_EVENT_TS]
    if before != len(events):
        print("  " + str(before - len(events)) + " events before " +
              str(MIN_EVENT_TS) + "s discarded.")

    for e in events:
        e["tts_speed"] = PRIORITY_SPEED.get(e.get("priority", 3), 1.1)

    match_sec = round(len(tracks["players"]) / FPS, 2)

    print("  " + str(len(events)) + " events in " + str(round(match_sec, 1)) + "s:")
    for e in events:
        side = e.get("field_side") or "-"
        att = ("atk" if e.get("attacking") else "def") if e.get("attacking") is not None else "-"
        extra = ""
        if "duration" in e:
            extra += " dur=" + str(e["duration"]) + "s"
        if "passes" in e:
            extra += " passes=" + str(e["passes"])
        print("  [" + str(round(e["ts"], 2)).rjust(6) + "s] "
              "P" + str(e["priority"]) + " "
              + e["type"].ljust(14) +
              " side=" + side + " " + att + extra)

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            return super().default(o)

    output = {
        "match_sec":    match_sec,
        "fps":          FPS,
        "team_context": team_context,
        "events":       events,
    }
    with open(EVENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, cls=_Enc)
    print("Next:  python createvoice.py --video output_videos/output_video.avi")


if __name__ == "__main__":
    main()
