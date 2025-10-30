# Technology Stack

## Core Technologies

### Programming Language
- **Python 3.10** - Required version for compatibility with all AI frameworks

### AI/ML Frameworks
- **YOLO11** - Primary object detection model
- **ONNX Runtime** - Cross-platform inference (NVIDIA CUDA, AMD/Intel GPU)
- **OpenVINO** - Intel optimization (CPU, iGPU - only format supporting integrated GPU)
- **PyTorch** - Native .pt model support with NVIDIA GPU acceleration
- **SafeTensors** - Safer, faster model loading than PyTorch
- **CoreML** - Apple Silicon optimization (Neural Engine + integrated GPU)
- **Ollama** - Local LLM for advanced color analysis

### Computer Vision & Media
- **OpenCV** - Video capture, image processing, frame manipulation
- **NumPy** - Numerical operations and array processing
- **Pygame** - Cross-platform audio playback (.wav, .mp3)

### Hardware Integration
- **Philips Hue API** - Smart lighting control via local bridge
- **Requests** - HTTP communication with Hue bridge and APIs

### Environment Management
- **Conda** - Primary environment manager (recommended)
- **pip** - Package installation for conda-unavailable packages

## Hardware Optimization Strategy

### Automatic Backend Selection Priority
1. **NVIDIA GPU**: ONNX Runtime with CUDA or SafeTensors
2. **AMD/Intel GPU**: ONNX Runtime with DirectML/OpenCL
3. **Intel iGPU**: OpenVINO (only format supporting integrated GPU)
4. **Apple Silicon**: CoreML for Neural Engine + integrated GPU
5. **CPU Fallback**: OpenVINO with multi-threading

### Performance Architecture
- **Multi-threaded Pipeline**: Separate threads for capture, preprocessing, inference, postprocessing
- **Inter-process Communication**: Optimized queues for maximum throughput
- **Dynamic Worker Allocation**: Configurable threads based on hardware capabilities

## Build System & Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate cat-analysis-env

# Alternative pip installation
pip install -r requirements.txt
```

### Development Commands
```bash
# Run application
python main.py

# Run with CLI overrides
python main.py --device cuda:0 --live-preview
python main.py -d cpu --no-live-preview

# Configuration validation
python main.py --validate-config

# Performance testing
python main.py --benchmark
```

### Testing Commands
```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Performance benchmarks
python -m pytest tests/performance/ -v
```

### Model Management
```bash
# Convert models (when implemented)
python tools/convert_models.py --input yolo11s.pt --output-formats onnx,openvino

# Validate model formats
python tools/validate_models.py --model-dir models/
```

## Configuration Management

### Single Configuration File
- **config.ini** - All system configuration in INI format
- **Environment Variables** - Override sensitive values (API keys, passwords)
- **CLI Arguments** - Runtime overrides for development/debugging

### Cross-Platform Considerations
- **pathlib** - Cross-platform path handling
- **Platform detection** - Automatic OS-specific feature selection
- **Backup tools**: rsync (Linux/macOS) vs robocopy (Windows)
- **Camera interfaces** - OS-specific camera access handling