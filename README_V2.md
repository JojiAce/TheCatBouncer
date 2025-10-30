# TheCatBouncer 🐱🚫 - Version 2.0

> **AI‑powered pet access control** – let your own cat in, keep unwelcome felines out.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

TheCatBouncer is a vision‑based, two‑phase monitoring system that **recognises your own cat by colour & shape and automatically repels intruders** with lights and sound – perfect for cat‑flap cameras, garden doors or patio windows.

### Key Features
- **High-accuracy cat detection**: YOLO11 and OpenVINO models natively supported
- **Smart hardware backend picker**: Automatically selects the best available backend (CUDA, OpenVINO, CPU)
- **Owner vs. intruder logic**: Simple colour‑histogram analysis to whitelist your cat
- **Two-phase efficiency**: Low‑res passive monitoring → High‑res active analysis
- **Smart-home integration**: Native Philips Hue support
- **Automatic deterrents**: Plays custom scare sounds and activates lights for unknown cats
- **Scheduled operation**: Configurable active/quiet hours & daily maintenance
- **Robust NAS backup**: OS-aware snapshots via rsync (Linux/macOS) or robocopy (Windows)
- **Data management**: Incremental backups, disk-space pruning, rolling rotation
- **Disk-space watchdog**: Continuously monitors free space and purges old data
- **CLI overrides & threaded I/O**: Command‑line flags for live preview & inference device
- **Comprehensive logging**: Multi‑level logging with performance metrics
- **Cross-platform**: Runs on Windows, macOS & Linux; works with any webcam

## Architecture

TheCatBouncer 2.0 follows a clean, modular architecture with distinct components:

```
src/
├── components/          # Component implementations
├── config/             # Configuration management
├── engines/            # AI inference engines
├── interfaces/         # Abstract base classes
├── managers/           # High-level managers
└── utils/              # Utility functions
```

### Core Components
- **ConfigManager**: Type-safe configuration with validation
- **CameraManager**: Cross-platform camera access
- **InferenceEngines**: ONNX, OpenVINO, PyTorch, SafeTensors, CoreML backends
- **PassiveMonitor**: Low-res motion/brightness detection
- **ActiveAnalyzer**: High-performance multi-threaded detection pipeline
- **ColorAnalyzer**: HSV and LLM-based color analysis
- **HueController**: Philips Hue integration
- **AudioManager**: Cross-platform sound playback
- **NotificationService**: Email and Telegram notifications
- **StorageManager**: Organized file storage with retention
- **ScheduleManager**: Time-based operation scheduling
- **PerformanceMonitor**: Real-time optimization and metrics
- **LoggingService**: Structured logging with rotation

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/JojiAce/TheCatBouncer.git
cd TheCatBouncer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your scare sounds directory
mkdir -p cat_scare_sound
```

### Hardware Requirements
- Webcam (720p+ recommended)
- **Optional:** Philips Hue bridge + lights
- **Optional:** NAS for backups

### Software Requirements
- Python 3.10
- Dependencies listed in `requirements.txt`

## Configuration

Copy and customize the configuration:

```bash
cp config.ini config.ini.bak  # Backup original
cp config.ini config.ini       # Edit your copy
```

Key configuration sections:

### Time Management
```ini
[TimeManagement]
start_time            = 18:00   # Active hours start
end_time              = 07:00   # Active hours end
data_management_time  = 07:10   # Daily maintenance window
```

### Camera Settings
```ini
[Camera]
camera_index          = 0
low_resolution        = 640,480   # Passive monitoring resolution
high_resolution       = 1920,1080 # Active analysis resolution
fps_low               = 5         # Passive FPS
fps_high              = 30        # Active FPS
```

### AI Model Configuration
```ini
[ImageRecognition]
yolo_model_path            = yolo11_openvino_model_paths/yolo11s_openvino_model
inference_device           = gpu      # cpu, gpu, or cuda:0
cat_class_id               = 15       # COCO dataset cat class ID
cat_confidence_threshold   = 0.8      # Detection confidence threshold
```

### Color Analysis
```ini
[ColorAnalysis]
# HSV range for black color detection
lower_black_hsv            = 0,0,0
upper_black_hsv            = 180,255,60
black_pixel_threshold      = 0.5      # Fraction of pixels that must be black

# Alternative: LLM-based analysis
analysis_method            = hsv      # hsv or llm
ollama_host                = http://localhost:11434
ollama_model               = llava
ollama_prompt              = Is the cat in this image black? Answer with only "yes" or "no".
```

## Usage

### Basic Usage
```bash
# Run with default configuration
python -m src.main

# Force live preview window
python -m src.main --live-preview

# Use specific inference device
python -m src.main --device cuda:0

# Enable verbose logging
python -m src.main --verbose
```

### Command Line Options
```
-h, --help            show this help message and exit
-c CONFIG, --config CONFIG
                      Path to configuration file (default: config.ini)
--live-preview        Force enable live preview window
--no-live-preview     Force disable live preview window
-d {cpu,gpu,cuda:0,cuda:1}, --device {cpu,gpu,cuda:0,cuda:1}
                      Force inference device (overrides config)
-v, --verbose         Enable verbose logging
--version             show program's version number and exit
```

## Components

### Inference Engine Selection
The system automatically selects the best available backend based on your hardware:
- **NVIDIA GPU**: CUDA with ONNX or PyTorch
- **AMD/Intel GPU**: OpenVINO with GPU plugin
- **Integrated GPU**: OpenVINO optimized for iGPU
- **Apple Silicon**: CoreML for Neural Engine
- **CPU**: OpenVINO CPU plugin

### Hardware Detection
```python
from src.engines.base_engine import HardwareDetector

detector = HardwareDetector()
hardware_info = detector.detect_hardware()
print(hardware_info)
```

### Performance Optimization
The system includes dynamic performance optimization:

- **Adaptive Threading**: Adjusts worker count based on CPU usage
- **Queue Optimization**: Dynamically adjusts queue sizes
- **Load Balancing**: Distributes work across available resources
- **Real-time Monitoring**: Tracks FPS, latency, and throughput

## Performance

### Benchmark Results
The optimized pipeline achieves:
- **Passive monitoring**: < 10% CPU at 5 FPS
- **Active analysis**: 15-30 FPS depending on hardware
- **Inference latency**: 10-50ms on modern GPUs
- **Memory usage**: 200-800MB depending on model size

### Optimization Tips
1. Use the appropriate model size for your hardware
2. Select the optimal inference backend for your GPU
3. Adjust camera resolution based on your performance needs
4. Use OpenVINO IR models for CPU inference
5. Enable GPU acceleration when available

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes (`pytest`)
5. Format your code (`black . && ruff .`)
6. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black ruff mypy pytest pytest-mock

# Code quality checks
ruff check .
black --check .
mypy src/

# Run tests
pytest
```

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---
*TheCatBouncer - Keeping your home purr-fectly secure since 2023 🐾*