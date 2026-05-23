# Fussball-Kommentierung

Automatische Sprachkommentierung von Fußballvideos.  
Das System erkennt Spielereignisse per Computer Vision und generiert mit einem lokalen LLM deutsche Kommentarsätze, die per Text-to-Speech in einen Audiotrack umgewandelt und mit dem Video zusammengeführt werden.


> Aufgrund der parallelen Entwicklungsstruktur liegt zu Beginn keine einheitliche Datenstruktur oder Videoeingabe vor. Dieser Schritt erfolgt daher in redundanter Form über eine separate Implementierung. Das Skript main.py verarbeitet ein Eingabevideo und erzeugt die erforderlichen Datenstrukturen. Der Fokus dieser Arbeit liegt auf den Komponenten readpkl.py, prepare_teams.py und createvoice.py. 

---

## Voraussetzungen

| Anforderung | Details |
|---|---|
| Python | 3.10 oder neuer |
| Ollama | lokal installiert und gestartet (`ollama serve`) |
| LLM-Modell | `ollama pull qwen3:8b` |
| ffmpeg | Im System-PATH verfügbar |
| GPU | Optional – CPU reicht, ist aber deutlich langsamer |

### Python-Pakete installieren

```bash
pip install ultralytics supervision scikit-learn opencv-python numpy pandas torch
```

---

## Bedienung (Weboberfläche)

```bash
python server.py
```

Browser öffnen: **http://localhost:8080**

**Schritt 1 – Video hochladen**  
Videodatei auswählen und auf *Hochladen* klicken. Das Video wird automatisch in `input_videos/` gespeichert.

**Schritt 2 – Teamfarben einstellen**  
Spielerfarbe, Torwartfarbe und Angriffsrichtung für beide Teams sowie die Schiedsrichterfarbe auswählen.

**Schritt 3 – Analyse starten**  
Auf *Analyse starten* klicken. Die Pipeline läuft vollständig automatisch durch:
1. Objekterkennung & Tracking (YOLO + ByteTrack)
2. Team-Klassifikation per Trikotfarbe
3. Ereigniserkennung (Schuss, Pressing, Balleroberung …)
4. Kommentargenerierung (LLM) + Sprachsynthese (TTS) + Videomontage

Das fertige Video mit Kommentarspur wird danach direkt im Browser abgespielt.

> **Hinweis:** Schritt 1 (Objekterkennung) dauert je nach Hardware mehrere Minuten.  
> Für ein neues Video werden die gecachten Tracking-Daten (Stubs) automatisch gelöscht.

---

## Bedienung (Kommandozeile)

```bash
# 1. Video in input_videos/ ablegen, dann Tracking und Annotation
python main.py

# 2. Teamfarben in teams_config.json eintragen, dann Trikotfarben klassifizieren
python prepare_teams.py

# 3. Ereignisse erkennen -> events.json
python readpkl.py

# 4. Kommentarspur generieren (ohne Thinking-Modus = schneller)
python createvoice.py --no-think --video output_videos/output_video.avi --output-video commentary_video.mp4
```

---

## Projektstruktur

```
input_videos/          Eingabevideo hierher ablegen
output_videos/         Annotiertes Video (wird von main.py erzeugt)
models/best.pt         YOLO-Modell (vortrainiert)
stubs/                 Gecachte Tracking-Daten (werden automatisch verwaltet)
ui/index.html          Weboberfläche
server.py              Webserver (startet die Oberfläche)
main.py                Schritt 1: Tracking und Annotation
prepare_teams.py       Schritt 2: Trikotfarben-Klassifikation
readpkl.py             Schritt 3: Ereigniserkennung
createvoice.py         Schritt 4: LLM-Kommentar + TTS + ffmpeg-Montage
teams_config.json      Teamfarben-Konfiguration
events.json            Erkannte Ereignisse (wird von readpkl.py erzeugt)
commentary_video.mp4   Fertiges Ausgabevideo mit Kommentarspur
```



## Häufige Probleme

| Problem | Lösung |
|---|---|
| `Ollama nicht erreichbar` | `ollama serve` in einem separaten Terminal starten |
| `ffmpeg not found` | ffmpeg installieren und sicherstellen, dass es im PATH liegt |
| Kein Video gefunden | Videodatei muss in `input_videos/` liegen (mp4/avi/mkv/mov) |
| Pipeline sehr langsam | Normales Verhalten auf CPU – Schritt 1 dauert am längsten |
