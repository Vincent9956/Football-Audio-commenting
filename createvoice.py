import os
import sys
import re
import json
import shutil
import argparse
import subprocess
import tempfile
import wave
import urllib.request as _urllib_req
import urllib.error as _urllib_err
import numpy as np
from debugger import timer as _timer, print_timing_summary as _print_timing_summary, TIMING as _TIMING, _timing_log


# ── TTS speed per priority ────────────────────────────────────────────
PRIORITY_SPEED = {5: 1.25, 4: 1.25, 3: 1.1, 2: 1.0, 1: 1.0}

# ── Ollama config ─────────────────────────────────────────────────────
_OLLAMA_URL = "http://localhost:11434/api/chat"
_DEFAULT_MODEL = "qwen3:8b"      # Thinking-Modell: denkt intern, bessere Kommentarqualitaet

# Human-readable event descriptions passed to the LLM
_EVENT_DESC = {
    "SHOT":           "Schussversuch - Ball mit hohem Tempo abgefeuert (kein Tor, kein Netz. nur Schuss!)",
    "DANGER":         "Angreifer mit Ball frei nahe dem gegnerischen Tor - gefaehrliche Situation",
    "POSSESSION_RUN": "Ballzirkulation - Team kombiniert sich mehrfach durch die eigenen Reihen",
    "PRESS":          "Pressing - Balltraeger wird mehren Gegnern eingeschnuert",
    "BALL_WON":       "Balleroberung - Team entreisst dem Gegner den Ball",
    "PASS":           "Zuspiel - der Ball wird zum Mitspieler gespielt",
}

_BASE_SYSTEM_PROMPT = (
    "Du bist ein emotionaler, spontaner deutscher Fussballkommentator.\n"
    "\n"
    "AUSGABEFORMAT:\n"
    "Genau 1-2 kurze deutsche Saetze, zusammen maximal 10 Woerter.\n"
    "Schreibe NUR den Kommentartext. Beginne mit: AUSGABE: <kommentar>\n"
    "\n"
    "REGELN:\n"
    "1. SHOT ist ein Schussversuch. KEIN TOR. Das Wort 'Tor' (auch 'aufs Tor', 'ins Netz') "
    "ist bei SHOT absolut verboten. Gut: 'Schuss!', 'feuert!', 'zieht ab!', 'Hammer!'.\n"
    "2. Keine Spielernamen, Rollen oder Nummern erfinden.\n"
    "3. Keine Ballfarbe erwaehnen.\n"
    "4. Richtungen (links/rechts) nur wenn 'Fluegel' im Prompt steht.\n"
    "5. Nur Fakten aus dem Prompt. keine Spekulationen.\n"
    "\n"
    "Beispiel SHOT: AUSGABE: Hammer! Der Ball fliegt!"
)


def _build_system_prompt(team_context):
    if not team_context:
        return _BASE_SYSTEM_PROMPT

    lines = [_BASE_SYSTEM_PROMPT, "", "Mannschaften in diesem Spiel:"]
    for tid, info in team_context.items():
        if tid == "referee":
            continue
        color = info.get("player_color", "")
        gk    = info.get("goalkeeper_color", "")
        att   = info.get("attacks", "")
        direction = "links" if att == "left" else "rechts" if att == "right" else att
        lines.append(
            "  Team " + tid + ": Spieler in " + color +
            ", Torwart in " + gk +
            ", greift nach " + direction + " an."
        )
    lines.append("")
    lines.append(
        "Nutze diese Informationen, um offensive und defensive Aktionen korrekt einzuordnen."
    )
    return "\n".join(lines)


def _strip_think(text):
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_from_thinking(thinking_text):
    _BAD = ("vermeide", "bedeutet", "field_side", "attacks=", "constraint",
            "Regel", "keine Information", "du weisst", "AUSGABEFORMAT",
            "STRENGE", "maximal 10", "Prompt", "System", "Inferenz",
            "team_context", "attacks_left", "attacks_right")

    def _is_bad(text):
        lo = text.lower()
        return any(b.lower() in lo for b in _BAD)

    # 1) Explicit marker that the system prompt asks the model to write
    marker = re.search(r'AUSGABE\s*:\s*(.+?)(?:\n|$)', thinking_text, re.IGNORECASE)
    if marker:
        candidate = marker.group(1).strip().strip('"„"«»')
        if len(candidate) >= 6 and not _is_bad(candidate):
            return candidate

    # 2) Quoted strings that look German, bad phrases excluded, last match wins
    candidates = re.findall(
        r'[""„«]([^""„«»\n]{6,100})[""»]',
        thinking_text
    )
    _GERMAN = ("der", "die", "das", "und", "ist", "ein", "Ball",
               "Tor", "Schuss", "Team", "Spiel", "klar", "gut", "stark")
    german = [c for c in candidates
              if any(m.lower() in c.lower() for m in _GERMAN)
              and not _is_bad(c)]
    if german:
        return german[-1].strip()

    # 3) Conservative heuristic: last short sentence that starts with a capital
    #    letter, ends with punctuation, and passes the bad-phrase filter.
    for line in reversed(thinking_text.splitlines()):
        line = line.strip()
        if (6 <= len(line) <= 120
                and re.match(r'^[A-ZÜÄÖ]', line)
                and line[-1] in '.!?'
                and not _is_bad(line)):
            return line

    return ""


def generate_commentary(event, model=_DEFAULT_MODEL, system_prompt=None, think=True):
    if system_prompt is None:
        system_prompt = _BASE_SYSTEM_PROMPT

    t = event["type"]
    desc = _EVENT_DESC.get(t, t)

    # Build user message
    parts = ["Ereignis: " + desc + "."]

    pid = event.get("player")
    if pid is not None:
        try:
            n = int(pid)
            label = ("Spieler " + str(n)) if n <= 99 else "ein Spieler"
            parts.append("Spieler: " + label + ".")
        except (TypeError, ValueError):
            pass

    team = event.get("team")
    if team and str(team) in (system_prompt or ""):
        parts.append("Team: " + str(team) + ".")

    field_side = event.get("field_side")
    attacking  = event.get("attacking")
    flank      = event.get("flank")   # "left" | "right" | None
    if field_side:
        situation = "Angriffsaktion" if attacking else "Defensivaktion"
        parts.append("Seite: " + field_side + " (" + situation + ").")
    if flank and attacking:
        # Only mention flank for attacking plays - irrelevant when defending
        flank_de = "links" if flank == "left" else "rechts"
        parts.append("Fluegel: " + flank_de + ".")

    if "passes" in event:
        parts.append("Paesse: " + str(event["passes"]) + ".")
    if "duration" in event:
        parts.append("Dauer: " + str(round(event["duration"], 1)) + "s.")

    user_msg = " ".join(parts)

    def _call(sys_p, usr_p):
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": usr_p},
            ],
            "stream": False,
            "options": {
                "num_predict": 2048 if think else 120,
                "temperature": 0.75,
            },
            "think": think,
        }).encode("utf-8")
        req = _urllib_req.Request(
            _OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urllib_req.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except _urllib_err.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(
                "Ollama HTTP " + str(exc.code) + " – falsches Modell oder kein Chat-Modell?\n"
                "Modell: " + model + "\nDetails: " + body
            ) from exc
        except _urllib_err.URLError as exc:
            raise RuntimeError(
                "Ollama nicht erreichbar (" + str(exc) + ").\n"
                "Starte Ollama mit:  ollama serve"
            ) from exc

        msg = data.get("message", {})
        raw = msg.get("content", "").strip()
        raw = _strip_think(raw)   # strip any inline <think> blocks

        if len(raw) < 3:
            thinking_text = msg.get("thinking") or ""
            if thinking_text:
                raw = _extract_from_thinking(thinking_text)

        # Strip the AUSGABE: marker if the model included it in content
        raw = re.sub(r'(?i)^AUSGABE\s*:\s*', '', raw).strip()
        for q in ('"', """, """, "„", "«", "»"):
            raw = raw.strip(q)
        return raw.strip()

    text = _call(system_prompt, user_msg)

    # Retry with minimal prompt if empty
    if len(text) < 3:
        print("[retry]", end=" ", flush=True)
        simple = "Deutscher Fussballkommentar, 1 Satz: " + desc
        text = _call("Du bist Fussballkommentator. Antworte nur mit dem Kommentar.", simple)

    if len(text) < 3:
        # Last resort: short fixed text (no audio gap)
        text = desc + "!"

    return text


# ── German number words ───────────────────────────────────────────────
_ONES = [
    "null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben",
    "acht", "neun", "zehn", "elf", "zwoelf", "dreizehn", "vierzehn",
    "fuenfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
]
_TENS = [
    "", "", "zwanzig", "dreissig", "vierzig", "fuenfzig",
    "sechzig", "siebzig", "achtzig", "neunzig",
]


def _num_de(n):
    if n < 0:
        return "minus " + _num_de(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] if o == 0 else (("ein" if o == 1 else _ONES[o]) + "und" + _TENS[t])
    if n < 1000:
        h, rest = divmod(n, 100)
        base = ("ein" if h == 1 else _ONES[h]) + "hundert"
        return base + ("" if rest == 0 else _num_de(rest))
    return str(n)


def normalize_text(text):
    text = re.sub(r"\b(\d+)\b", lambda m: _num_de(int(m.group())), text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Silero TTS ────────────────────────────────────────────────────────
_model_cache = None


def _load_silero(speaker):
    global _model_cache
    if _model_cache is None:
        import torch
        print("Loading Silero TTS model ...")
        model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts",
            language="de", speaker="v3_de", trust_repo=True,
        )
        model.to(torch.device("cpu"))
        _model_cache = (model, speaker)
        print("Model ready.\n")
    return _model_cache


def tts_generate(text, out_path, speaker="karlsson"):
    if not text or not text.strip():
        raise ValueError("tts_generate: empty text")
    model, spk = _load_silero(speaker)
    import torch
    audio = model.apply_tts(text=text, speaker=spk, sample_rate=48_000)
    pcm = np.clip(audio.numpy() * 32_767, -32_768, 32_767).astype(np.int16)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48_000)
        wf.writeframes(pcm.tobytes())


# ── ffprobe duration ──────────────────────────────────────────────────
def measure_duration(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    r = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json",
         "-show_entries", "format=duration", "-show_streams", path],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return None
    data = json.loads(r.stdout)
    # Try format-level duration first (most reliable)
    fmt_dur = data.get("format", {}).get("duration")
    if fmt_dur:
        return float(fmt_dur)
    for s in data.get("streams", []):
        if "duration" in s:
            return float(s["duration"])
    return None


# ── Event selection ───────────────────────────────────────────────────
def select_events(events, min_priority, min_gap):
    """Greedy selection: enforce min_gap. Higher priority bumps last event within gap."""
    candidates = sorted(
        [e for e in events if e.get("priority", 1) >= min_priority],
        key=lambda e: (e["ts"], -e.get("priority", 1)),
    )
    selected, last_ts, last_prio = [], -(min_gap * 2), 0
    for evt in candidates:
        gap = evt["ts"] - last_ts
        prio = evt.get("priority", 1)
        if gap >= min_gap:
            selected.append(evt)
            last_ts, last_prio = evt["ts"], prio
        elif prio > last_prio and selected:
            selected[-1] = evt
            last_ts, last_prio = evt["ts"], prio
    return selected


# ── ffmpeg audio assembly ─────────────────────────────────────────────
def assemble_audio(clips, match_sec, out_path, speed=1.3):
    """
    Build a silent base track and mix in TTS clips at their timestamps.
    clips: list of (place_ts, wav_path, clip_speed)
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found.")

    # Normalize to 3-tuples
    norm = []
    for c in clips:
        norm.append(c if len(c) == 3 else (c[0], c[1], speed))

    n = len(norm)
    cmd = [ffmpeg, "-y"]
    for _, p, _ in norm:
        cmd += ["-i", p]

    parts = ["aevalsrc=0:c=mono:s=44100:d=" + str(round(match_sec, 3)) + "[sil]"]
    for i, (ts, _, spd) in enumerate(norm):
        delay_ms = ts * 1000.0
        tempo = (",atempo=" + str(round(spd, 4))) if abs(spd - 1.0) > 0.001 else ""
        parts.append(
            "[" + str(i) + "]asetpts=PTS-STARTPTS" + tempo + ","
            "adelay=" + str(round(delay_ms, 1)) + "|" + str(round(delay_ms, 1)) +
            "[c" + str(i) + "]"
        )
    all_in = "[sil]" + "".join("[c" + str(i) + "]" for i in range(n))
    parts.append(
        all_in + "amix=inputs=" + str(n + 1) +
        ":normalize=0:dropout_transition=0:duration=first[out]"
    )
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]", "-ar", "44100", "-ac", "1", "-b:a", "128k",
        out_path,
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-2000:])


def merge_with_video(audio, video, out):
    ffmpeg = shutil.which("ffmpeg")
    # libx264 + aac = universell browserkompatibel
    r = subprocess.run([
        ffmpeg, "-y", "-i", video, "-i", audio,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-movflags", "+faststart",
        out,
    ], capture_output=True, text=True)
    if r.returncode == 0:
        mb = os.path.getsize(out) / 1_048_576
        print("  Video saved -> " + out + "  (" + str(round(mb, 1)) + " MB)")
    else:
        print("  Video merge failed:\n" + r.stderr[-500:])


# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events",       default="events.json")
    ap.add_argument("--output-audio", default="commentary_audio.mp3")
    ap.add_argument("--video",        default=None)
    ap.add_argument("--output-video", default="commentary_video.mp4")
    ap.add_argument("--speaker",      default="karlsson",
                    choices=["friedrich", "eva_k", "bernd_ungerer",
                             "karlsson", "hokuspokus", "random"])
    ap.add_argument("--speed",        default=1.3, type=float,
                    help="Global fallback TTS speed")
    ap.add_argument("--min-priority", default=3, type=int,
                    help="Min event priority to voice (default 3)")
    ap.add_argument("--min-gap",      default=1.5, type=float,
                    help="Min seconds between clips (default 1.5)")
    ap.add_argument("--lead-time",    default=0.0, type=float,
                    help="Offset in seconds from event to clip START (default 0.0). "
                         "0=clip starts at event, -1.0=clip starts 1s before event.")
    ap.add_argument("--model",        default=_DEFAULT_MODEL,
                    help="Ollama model name (default: " + _DEFAULT_MODEL + ")")
    ap.add_argument("--no-think",     action="store_true",
                    help="Thinking-Modus deaktivieren (schneller, fuer Vergleichsmessungen)")
    args = ap.parse_args()
    use_think = not args.no_think
    if _TIMING:
        print("Modus: " + ("Thinking AN" if use_think else "Thinking AUS") + "\n")

    if not os.path.exists(args.events):
        sys.exit("'" + args.events + "' not found - run 'python readpkl.py' first.")

    with open(args.events, encoding="utf-8") as f:
        data = json.load(f)

    match_sec = data["match_sec"]
    all_evts = data["events"]
    team_context = data.get("team_context", {})


    system_prompt = _build_system_prompt(team_context)
    if team_context:
        print("Team context loaded: " + str(list(team_context.keys())) + "\n")

    if args.video and os.path.exists(args.video):
        video_dur = measure_duration(args.video)
        if video_dur and abs(video_dur - match_sec) > 1.0:
            print("Hinweis: Tracking-Dauer " + str(round(match_sec, 1)) + "s, "
                  "Video-Dauer " + str(round(video_dur, 1)) + "s -> verwende Video-Dauer.")
            match_sec = video_dur
            # Events ausserhalb des Videos verwerfen
            all_evts = [e for e in all_evts if e["ts"] < match_sec - 0.5]

    print("Loaded " + str(len(all_evts)) + " events, match = " + str(round(match_sec, 1)) + "s\n")

    selected = select_events(all_evts, args.min_priority, args.min_gap)
    print("Selected " + str(len(selected)) + " events "
          "(min_priority=" + str(args.min_priority) + ", "
          "min_gap=" + str(args.min_gap) + "s)\n")

    if not selected:
        sys.exit("No events selected. Lower --min-priority or --min-gap.")

    placed, last_end = [], -999.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for evt in selected:
            ts = evt["ts"]
            prio = evt.get("priority", 3)

            print("  LLM  [" + str(round(ts, 2)) + "s] P" + str(prio) +
                  " " + evt["type"].ljust(12), end=" ", flush=True)
            try:
                with _timer("LLM", ts) as t_llm:
                    raw_text = generate_commentary(evt, model=args.model,
                                                   system_prompt=system_prompt,
                                                   think=use_think)
            except Exception as exc:
                print("FEHLER (" + str(exc) + ") - uebersprungen")
                continue
            text = normalize_text(raw_text)
            llm_info = ("  [T LLM " + str(round(t_llm.elapsed, 2)) + "s]") if _TIMING else ""
            print("-> " + text + llm_info)

            path = os.path.join(tmpdir, "clip_" + str(round(ts, 3)) + ".wav")
            clip_speed = evt.get("tts_speed", PRIORITY_SPEED.get(prio, args.speed))

            try:
                with _timer("TTS", ts) as t_tts:
                    tts_generate(text, path, args.speaker)
                if _TIMING:
                    print("  TTS  [T " + str(round(t_tts.elapsed, 2)) + "s]")
            except Exception as exc:
                print("  TTS FEHLER: " + str(exc) + " - uebersprungen")
                continue

            dur = measure_duration(path)
            if dur is None:
                dur = len(text.split()) / 4.0
            play_dur = dur / clip_speed if clip_speed > 1.0 else dur
            place_ts = max(0.0, ts + args.lead_time)

            if place_ts < last_end + 0.1:
                print("  SKIP [" + str(round(place_ts, 2)) + "s] Ueberlappung")
                continue

            placed.append((place_ts, path, clip_speed))
            last_end = place_ts + play_dur
            print("  OK   place=" + str(round(place_ts, 2)) + "s "
                  "(event@" + str(round(ts, 2)) + "s, endet@" +
                  str(round(place_ts + play_dur, 2)) + "s) "
                  "dur=" + str(round(play_dur, 1)) + "s "
                  "spd=" + str(round(clip_speed, 2)) + "x")

        print("\n" + str(len(placed)) + " clips placed. Assembling ...")
        if not placed:
            sys.exit("No clips placed.")

        with _timer("ffmpeg-mux", 0.0) as t_mux:
            assemble_audio(placed, match_sec, args.output_audio, speed=args.speed)
        if _TIMING:
            print("  ffmpeg-Montage  [T " + str(round(t_mux.elapsed, 2)) + "s]")

    kb = os.path.getsize(args.output_audio) // 1024
    print("Audio saved -> " + args.output_audio +
          "  (" + str(kb) + " KB, " + str(round(match_sec, 1)) + "s)\n")

    if args.video:
        if os.path.exists(args.video):
            print("Merging commentary into video ...")
            merge_with_video(args.output_audio, args.video, args.output_video)
        else:
            print("Video not found: " + args.video)

    # _print_timing_summary()

    # Timing-Daten als JSON speichern für die Web-UI
    if _timing_log:
        llm_times = [e[2] for e in _timing_log if e[0] == "LLM"]
        tts_times = [e[2] for e in _timing_log if e[0] == "TTS"]
        timing_data = {
            "model": args.model,
            "think": use_think,
            "llm_avg":   round(sum(llm_times) / len(llm_times), 2) if llm_times else 0,
            "llm_total": round(sum(llm_times), 2),
            "tts_avg":   round(sum(tts_times) / len(tts_times), 2) if tts_times else 0,
            "entries": [
                {"label": e[0], "ts": round(e[1], 2), "elapsed": round(e[2], 2)}
                for e in _timing_log
            ],
        }
        with open("timing.json", "w", encoding="utf-8") as f:
            json.dump(timing_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
