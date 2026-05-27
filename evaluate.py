"""
evaluate.py – Vergleicht erkannte Events (events.json) mit Ground-Truth (ground_truth.json).
Berechnet Precision, Recall und F1 pro Ereignistyp und gibt eine lesbare Auswertung aus.

Verwendung:
    python evaluate.py
    python evaluate.py --events events.json --gt ground_truth.json
"""
import json
import argparse

# ---------------------------------------------------------------------------
# Typ-Normalisierung
# ---------------------------------------------------------------------------
TYPE_ALIASES = {
    "pressing":       "PRESS",
    "press":          "PRESS",
    "danger":         "DANGER",
    "shot":           "SHOT",
    "possession_run": "POSSESSION_RUN",
    "ball_won":       "BALL_WON",
    "pass":           "PASS",
}

def _norm(t: str) -> str:
    return TYPE_ALIASES.get(t.lower(), t.upper())


# ---------------------------------------------------------------------------
# Matching  (greedy nach Zeitstempel: naechstes unbenutztes Pred-Event
# innerhalb der Toleranz wird als TP gewertet)
# ---------------------------------------------------------------------------
def match_events(predicted, ground_truth, tolerance):
    used_pred = set()
    matched   = []   # (gt, pred, delta)
    missed    = []   # gt ohne Treffer

    for gt in sorted(ground_truth, key=lambda e: e["ts"]):
        candidates = sorted(
            [(abs(p["ts"] - gt["ts"]), i, p)
             for i, p in enumerate(predicted)
             if p["type"] == gt["type"] and i not in used_pred]
        )
        if candidates and candidates[0][0] <= tolerance:
            dist, idx, pred = candidates[0]
            used_pred.add(idx)
            matched.append((gt, pred, round(dist, 2)))
        else:
            missed.append(gt)

    extra = [p for i, p in enumerate(predicted) if i not in used_pred]
    return matched, missed, extra


# ---------------------------------------------------------------------------
# Hilfsfunktion: ASCII-Balken
# ---------------------------------------------------------------------------
def _bar(value, width=10):
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Hauptauswertung
# ---------------------------------------------------------------------------
def evaluate(events_path, gt_path):
    with open(events_path, encoding="utf-8") as f:
        pred_all = json.load(f)["events"]
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    tolerance = gt_data.get("tolerance_sec", 2.0)
    gt_all    = gt_data["events"]

    for e in pred_all:
        e["type"] = _norm(e["type"])
    for e in gt_all:
        e["type"] = _norm(e["type"])

    event_types   = sorted(set(e["type"] for e in gt_all + pred_all))
    video_sec     = None
    try:
        video_sec = json.load(open(events_path, encoding="utf-8")).get("match_sec")
    except Exception:
        pass

    W = 72
    print()
    print("=" * W)
    print("  EVENT DETECTION EVALUATION")
    if video_sec:
        print(f"  Video: {video_sec:.1f}s   |   Toleranz: +/- {tolerance}s")
    else:
        print(f"  Toleranz: +/- {tolerance}s")
    print("=" * W)

    # ── 1. Matched pairs + misses pro Typ ──────────────────────────────
    all_matched, all_missed, all_extra = [], [], []
    stats = {}

    for etype in event_types:
        pred = [e for e in pred_all if e["type"] == etype]
        gt   = [e for e in gt_all   if e["type"] == etype]
        matched, missed, extra = match_events(pred, gt, tolerance)
        all_matched += matched
        all_missed  += missed
        all_extra   += extra

        tp = len(matched)
        fp = len(extra)
        fn = len(missed)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        stats[etype] = (tp, fp, fn, prec, recall, f1)

    # ── 2. Matches-Tabelle ─────────────────────────────────────────────
    print()
    print("  ERKANNTE EVENTS (Matches)")
    print(f"  {'Typ':<18}  {'GT':>7}   {'Pred':>7}   {'Delta':>6}   Status")
    print("  " + "-" * 60)

    # Sortiere nach GT-Zeitstempel fuer lesbare Reihenfolge
    all_rows = [(gt["ts"], gt["type"], pred["ts"], delta, "OK")
                for gt, pred, delta in all_matched]
    all_rows += [(gt["ts"], gt["type"], None, None, "MISS")
                 for gt in all_missed]
    all_rows.sort(key=lambda r: (r[0], r[1]))

    for gt_ts, etype, pred_ts, delta, status in all_rows:
        if status == "OK":
            sign = "+" if pred_ts >= gt_ts else "-"
            print(f"  {etype:<18}  {gt_ts:>6.2f}s  ->  {pred_ts:>6.2f}s  "
                  f"  {sign}{delta:.2f}s")
        else:
            print(f"  {etype:<18}  {gt_ts:>6.2f}s  ->  {'---':>7}   "
                  f"  nicht erkannt  <<<")

    # ── 3. Falsch-positive ─────────────────────────────────────────────
    if all_extra:
        print()
        print("  FALSCH-POSITIVE (kein GT-Match)")
        print(f"  {'Typ':<18}  {'Pred':>7}")
        print("  " + "-" * 30)
        for p in sorted(all_extra, key=lambda e: e["ts"]):
            print(f"  {p['type']:<18}  {p['ts']:>6.2f}s")

    # ── 4. Zusammenfassung pro Typ ─────────────────────────────────────
    print()
    print("  ZUSAMMENFASSUNG PRO TYP")
    print(f"  {'Typ':<18}  {'TP':>3} {'FP':>3} {'FN':>3}  "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6}  F1-Balken")
    print("  " + "-" * 66)

    total_tp = total_fp = total_fn = 0
    for etype in event_types:
        tp, fp, fn, prec, recall, f1 = stats[etype]
        total_tp += tp; total_fp += fp; total_fn += fn
        print(f"  {etype:<18}  {tp:>3} {fp:>3} {fn:>3}  "
              f"{prec:>5.0%} {recall:>5.0%} {f1:>5.0%}  {_bar(f1)}")

    print("  " + "-" * 66)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    print(f"  {'GESAMT':<18}  {total_tp:>3} {total_fp:>3} {total_fn:>3}  "
          f"{p:>5.0%} {r:>5.0%} {f:>5.0%}  {_bar(f)}")

    # ── 5. Einzeiler-Fazit ─────────────────────────────────────────────
    print()
    hit  = total_tp
    miss = total_fn
    fp_n = total_fp
    total_gt = hit + miss
    print(f"  Fazit: {hit}/{total_gt} GT-Events erkannt  |  "
          f"{fp_n} Falsch-Positive  |  "
          f"Recall {r:.0%}  F1 {f:.0%}")
    print("=" * W)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="events.json")
    ap.add_argument("--gt",     default="ground_truth.json")
    args = ap.parse_args()
    evaluate(args.events, args.gt)
