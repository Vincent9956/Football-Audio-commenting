import sys
import time

TIMING = True
_timing_log: list = []  # [(label, event_ts, elapsed_s), ...]


class timer:
    def __init__(self, label: str, event_ts: float = 0.0):
        self.label    = label
        self.event_ts = event_ts
        self.elapsed  = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter() if TIMING else 0.0
        return self

    def __exit__(self, *_):
        if TIMING:
            self.elapsed = time.perf_counter() - self._t0
            _timing_log.append((self.label, self.event_ts, self.elapsed))


def print_timing_summary():
    if not TIMING or not _timing_log:
        return
    llm = [r for r in _timing_log if r[0] == "LLM"]
    tts = [r for r in _timing_log if r[0] == "TTS"]
    mux = [r for r in _timing_log if r[0] == "ffmpeg-mux"]

    def _fmt(rows):
        if not rows:
            return "-"
        vals = [r[2] for r in rows]
        return (str(round(sum(vals), 2)) + "s total  |  "
                "O " + str(round(sum(vals) / len(vals), 2)) + "s  |  "
                "min " + str(round(min(vals), 2)) + "s  |  "
                "max " + str(round(max(vals), 2)) + "s")

    modus = "Thinking AUS (--no-think)" if any(
        "--no-think" in str(a) for a in sys.argv) else "Thinking AN"
    print("\n" + "=" * 62)
    print("  TIMING-AUSWERTUNG  [" + modus + "]")
    print("=" * 62)
    print("  LLM-Generierung  : " + _fmt(llm))
    print("  TTS-Synthese     : " + _fmt(tts))
    print("  ffmpeg-Montage   : " + _fmt(mux))
    if llm and tts:
        total = sum(r[2] for r in llm) + sum(r[2] for r in tts)
        print("  Gesamt (LLM+TTS) : " + str(round(total, 2)) + "s")
    print("=" * 62)
    print()
    print("  Pro Event:")
    print("  {:>8}  {:>10}  {:>10}".format("@Sekunde", "Schritt", "Dauer"))
    print("  " + "-" * 38)
    for label, ts, elapsed in _timing_log:
        print("  {:>8}  {:>10}  {:>9}s".format(
            str(round(ts, 2)) + "s", label, str(round(elapsed, 2))))
    print("=" * 62 + "\n")
