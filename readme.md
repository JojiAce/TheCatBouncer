# TheCatBouncer

TheCatBouncer is a two-stage computer-vision pipeline that keeps the entrance to your home cat-friendly. It waits for a light-triggered event, performs a high-resolution object detection pass, analyses the colour of the detected animal and reacts by notifying you, switching Philips Hue lights and starting deterrent recordings.

This repository contains a production-ready restructuring of the project with clearly separated modules, validated configuration loading and a documented setup.

## Key Features

- **Passive + active camera pipeline** – energy-efficient low-resolution polling switches to high-resolution inference only when a light trigger fires.
- **Pluggable inference backends** – ONNX Runtime, OpenVINO, PyTorch, Safetensors and CoreML wrappers share the same interface.
- **Colour verification** – HSV or optional LLM-based analysis decides whether the detected cat matches the expected colour.
- **Smart-home integration** – Philips Hue support with graceful error handling.
- **Intruder response** – configurable notifications, scare sounds and recording of evidence.
- **Daily maintenance** – scheduled disk clean-up and NAS backups keep the system tidy.

## Requirements

- Python 3.10 or newer.
- Webcam or RTSP compatible camera.
- Optional Philips Hue bridge and lights.
- Optional NAS reachable via rsync (Linux/macOS) or robocopy (Windows).

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** GPU specific packages (e.g. `torch`, `openvino-dev`, `onnxruntime-gpu`) may require platform-specific installation instructions. Adjust the requirements file to match your hardware.

## Repository Structure

```
TheCatBouncer/
├── main-file_and_active_analysis_pipeline.py   # Entry point that orchestrates the workflow
├── passive_analyzer.py                         # Passive light trigger logic
├── color_analyzer.py                           # HSV based colour verification
├── llm_color_analyzer.py                       # Optional Ollama powered colour verification
├── inference_factory.py                        # Backend independent inference loader
├── NewPickerEnigneforActive.py                 # Helper to resolve model file paths
├── hue_controller.py                           # Philips Hue integration
├── notifier.py                                 # Email/Telegram notifications
├── utility_recorder.py                         # Intruder recording & scare sounds
├── data_manager.py                             # Disk cleanup and NAS backup
├── config.ini                                  # Main configuration (edit this)
├── config.sample.ini                           # Template configuration
├── requirements.txt                            # Python dependencies
└── readme.md                                   # This document
```

## Configuration

1. Copy the sample configuration and adjust it to your environment:

   ```bash
   cp config.sample.ini config.ini
   ```

2. Provide the correct paths for your YOLO models in `[Backend]`. The repository expects the folder layout described in `NewPickerEnigneforActive.py`.
3. Fill in Philips Hue credentials, notification details and NAS settings if you use those integrations.
4. Adjust the `[TimeManagement]` window so the system runs during the hours you need.

## Running the System

Ensure your models are available and the configuration is complete, then run:

```bash
python main-file_and_active_analysis_pipeline.py
```

The main loop will:

1. Run the passive analyser until a stable light trigger occurs.
2. Switch to the active inference pipeline with multiprocessing.
3. Store successful detections including bounding boxes and a video recording.
4. Perform colour analysis and trigger notifications or deterrents for intruders.
5. Continue monitoring within the configured schedule and perform daily maintenance once per day.

## Maintenance & Logs

- Successful detections are written to the folder configured via `success_frame_folder`.
- Intruder recordings are stored in `intruder_video_folder`.
- Per-run logs live under `logs/<YYYY-MM-DD>/analysis.log`.
- Daily NAS backups use rsync on Unix-like systems and robocopy on Windows with verbose logging of failures.

## Troubleshooting

- Set `[Debug] window = false` if you run the pipeline on a headless system.
- If you disable Philips Hue or notifications, leave the related sections blank – the modules degrade gracefully.
- When using the LLM-based colour analysis ensure `ollama` and the configured model are available locally; otherwise the HSV method is recommended.
- The revamped `llm_color_analyzer` can also talk to an Ollama server via pure HTTP, which is handy on minimalist deployments where installing the Python package is not possible. Use `ollama_temperature` and `ollama_timeout` in the config to fine tune requests.

## Development Notes

- The codebase follows modular logging via `logging.getLogger(__name__)` in each module.
- Multiprocessing queues are size-bound to keep memory usage predictable.
- The repository ships with a requirements file; pin versions according to your deployment environment for reproducibility.

Feel free to open issues or submit pull requests if you extend the project (e.g. by adding support for other smart home systems or inference engines).

## Repository Hygiene

- The project-level `.gitignore` excludes generated artefacts such as `__pycache__/`, build outputs, and local data directories like `CatDetectorData/`.
- Local interpreter selections stored in `.python-version` stay untracked so contributors can use Pyenv without polluting the repository.
- Keep dependency manifests (`requirements.txt`, `setup.py`) ending with a newline to ensure packaging tools read the full metadata.

