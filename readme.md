# TheCatBouncer 🐱🚫

> **AI‑powered pet access control** – let your own cat in, keep unwelcome felines out.

    

---

TheCatBouncer is a vision‑based, two‑phase monitoring system that **recognises your own cat by colour & shape and automatically repels intruders** with lights and sound – perfect for cat‑flap cameras, garden doors or patio windows.

---

## Table of Contents

1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Hardware & Software Requirements](#requirements)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running](#running)
7. [CLI Reference](#cli-reference)
8. [Folder Layout](#folder-layout)
9. [Roadmap](#roadmap)
10. [Contributing](#contributing)
11. [License](#license)

---

## Features

| Category                        | Details                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| **High‑accuracy cat detection** | YOLOv11/yolo11l_openvino models (user‑selectable) with automatic fallback CPU → CUDA → OpenVINO       |
| **Owner vs. intruder logic**    | Simple colour‑histogram analysis lets you whitelist *your* cat with just a few HSV ranges |
| **Two‑phase efficiency**        | Low‑res passive monitoring → High‑res active analysis when motion/light is detected       |
| **Smart‑home integration**      | Native **Philips Hue** support (lights on/off, colour changes, flash patterns)            |
| **Automatic deterrents**        | Plays custom scare sounds (`.wav` / `.mp3`) + light show for unknown cats                 |
| **Scheduled operation**         | Active hours / quiet hours & daily maintenance cron (backup + cleanup)                    |
| **Data management**             | Incremental backups to NAS, automatic disk‑space pruning                                  |
| **Cross‑platform**              | Runs on Windows, macOS & Linux; works with any UVC webcam                                 |
| **Fully configurable**          | Single `config.ini` controls *everything*; no code changes required                       |

---

## How it Works

```mermaid
flowchart TD
  subgraph Passive[Passive Phase]
    A[Low‑res webcam<br/>@ 640×480] -->|Motion/light trigger| Switch{Trigger met?}
  end
  Switch -->|Yes| Active
  Switch -->|No| A
  subgraph Active[Active Phase]
    B[High‑res webcam<br/>@ 1920×1080] --> C[YOLO inference]
    C --> D{Cat confidence ≥ threshold?}
    D -->|No| EndAll[Sleep for short time + goes back to Passiv Phase]
    D -->|Yes| ColourCheck[Colour histogram<br/>matches OWN cat?]
    ColourCheck -->|Yes| Welcome[Do nothing + goes back to Passiv Phase after short time]
    ColourCheck -->|No| Repel[Scare sound + Hue lights]
  end
  Active --> EndAll
```

---

## Requirements

### Hardware

- USB / IP webcam (720p+ recommended)
- **Optional:** Philips Hue bridge + lights
- **Optional:** NAS with SMB share for backups

### Software

- Python **3.10**
- [Conda ≥ 23] or [Mamba] for environment management
- Dependencies listed in `environment.yml`

---

## Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/JojiAce/TheCatBouncer.git
$ cd TheCatBouncer

# 2. Create environment (recommended)
$ conda env create -f environment.yml
$ conda activate catbouncer-env

# 3. (Optional) Install Ultralytics if you want YOLOv8
$ pip install ultralytics

# 4. Add your scare sounds
$ mkdir -p cat_scare_sound && cp ~/Downloads/bark.mp3 cat_scare_sound/
```

---

## Configuration

Copy the template and fill in **your** values:

```bash
cp config.sample.ini config.ini
```

```ini
[Zeitsteuerung]
start_zeit = 01:00   # Monitoring starts 6 PM
end_zeit   = 07:00   # Monitoring ends 7 AM

[PhilipsHue]
bridge_ip = YOUR_BRIDGE_IP_HEAR
app_key   = YOUR_APP_KEY_HERE
licht_ids = YOUR_LICHT_IDS_HEAR

[Kamera]
kamera_index      = 0
niedrige_aufloesung = 640,480
hohe_aufloesung     = 1920,1080
fps_niedrig = 10
fps_hoch    = 30

[Bilderkennung]
yolo_modell_pfad         = models/yolov8n.pt
katzen_klassen_id        = 15
katzen_sicherheit_schwelle = 0.8

[Farbanalyse]
untere_schwarz_hsv = 0,0,0
obere_schwarz_hsv  = 180,255,50
schwarz_pixel_schwelle = 0.5
```

> **Tip :** Calibrate colour ranges with the built‑in live debug window (`--live-preview`).

---

## Running

```bash
# Basic run (uses config.ini)
python TheCatBouncer.py

# Force live preview window
python TheCatBouncer.py --live-preview

# Override inference device
python TheCatBouncer.py --device cuda:0
```

### CLI Reference

| Flag                                 | Description                       | Default      |
| ------------------------------------ | --------------------------------- | ------------ |
| `--config PATH`                      | Path to configuration file        | `config.ini` |
| `--live-preview / --no-live-preview` | Show/hide OpenCV window           | From config  |
| `--device {cpu,cuda:0,gpu}`          | Force inference device            | Auto‑detect  |
| `--debug`                            | Verbose logging & visual overlays | Off          |

---

## Folder Layout

```
TheCatBouncer/
├── cat_scare_sound/        # Your audio deterrents (*.wav / *.mp3)
├── models/                 # YOLO weights (.pt) or OpenVINO (.xml+.bin)
├── backups/                # NAS snapshots (optional)
├── VideoRecordings_Debug/  # Saved camera clips (optional)
├── TheCatBouncer.py        # Main entry‑point
├── config.sample.ini       # Template config – **commit me**
└── environment.yml         # Conda env definition
```

---

## Roadmap

-Not really Sure yet

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.\
Make sure to run `ruff` and `black` before committing.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

