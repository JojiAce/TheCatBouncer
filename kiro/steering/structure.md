# Project Structure and Organization

## Directory Layout

### Current Legacy Structure (to be refactored)
```
TheCatBouncer/
├── main-file_and_active_analysis_pipeline.py  # Legacy main entry point
├── passive_analyzer.py                        # Passive monitoring implementation
├── color_analyzer.py                          # HSV color analysis
├── LLM_COLOR_DETECTOR.py                     # LLM-based color analysis
├── inference_factory.py                       # AI model loading factory
├── hue_controller.py                          # Philips Hue integration
├── data_manager.py                            # Storage and backup management
├── utility_recorder.py                        # Event recording utilities
├── notifer.py                                 # Notification system
├── NewPickerEnigneforActive.py               # Model path selection
├── config.ini                                # Main configuration file
├── environment.yml                           # Conda environment definition
└── readme.md                                 # Project documentation
```

### Target Clean Structure (rebuild goal)
```
src/
├── __init__.py
├── main.py                                   # New main entry point
├── config/
│   ├── __init__.py
│   ├── manager.py                           # Configuration management
│   └── validation.py                       # Config validation
├── camera/
│   ├── __init__.py
│   ├── manager.py                           # Camera abstraction
│   └── capture.py                           # Frame capture utilities
├── ai/
│   ├── __init__.py
│   ├── inference_engine.py                 # Abstract base class
│   ├── onnx_engine.py                      # ONNX Runtime implementation
│   ├── openvino_engine.py                  # OpenVINO implementation
│   ├── pytorch_engine.py                   # PyTorch implementation
│   ├── safetensors_engine.py               # SafeTensors implementation
│   ├── coreml_engine.py                    # CoreML implementation
│   └── factory.py                          # Engine factory with hardware detection
├── monitoring/
│   ├── __init__.py
│   ├── passive.py                          # Passive monitoring system
│   ├── active.py                           # Active analysis pipeline
│   └── scheduler.py                        # Time-based scheduling
├── analysis/
│   ├── __init__.py
│   ├── color_hsv.py                        # HSV color analysis
│   ├── color_llm.py                        # LLM color analysis
│   └── detector.py                         # Detection result processing
├── integrations/
│   ├── __init__.py
│   ├── hue.py                              # Philips Hue controller
│   ├── audio.py                            # Audio deterrent system
│   └── notifications.py                    # Email/Telegram notifications
├── storage/
│   ├── __init__.py
│   ├── manager.py                          # Storage management
│   ├── backup.py                           # Cross-platform backup service
│   └── cleanup.py                          # Automated cleanup
├── performance/
│   ├── __init__.py
│   ├── monitor.py                          # Performance monitoring
│   └── optimizer.py                        # Dynamic optimization
└── utils/
    ├── __init__.py
    ├── logging.py                          # Structured logging
    ├── hardware.py                         # Hardware detection
    └── platform.py                         # Cross-platform utilities

tests/
├── __init__.py
├── unit/                                   # Unit tests
├── integration/                            # Integration tests
├── performance/                            # Performance benchmarks
└── fixtures/                              # Test data and mocks

tools/
├── convert_models.py                       # Model format conversion
├── validate_models.py                      # Model validation
└── setup_hue.py                           # Hue bridge setup utility

data/                                       # Runtime data (auto-created)
├── detections/                             # Successful detections
├── intruders/                              # Intruder recordings
├── logs/                                   # Application logs
└── backups/                               # Local backup staging

models/                                     # AI model storage
├── yolo11/
│   ├── onnx/                              # ONNX format models
│   ├── openvino/                          # OpenVINO format models
│   ├── pytorch/                           # PyTorch .pt models
│   ├── safetensors/                       # SafeTensors format
│   └── coreml/                            # CoreML format (macOS)
└── conversion/                            # Temporary conversion workspace

sounds/                                     # Audio deterrent files
├── scare1.wav
├── scare2.mp3
└── ...

config/
├── config.ini                             # Main configuration
├── config.sample.ini                      # Template configuration
└── .env.example                           # Environment variables template
```

## Code Organization Principles

### Module Structure
- **Single Responsibility**: Each module handles one specific aspect of the system
- **Abstract Interfaces**: Use ABC classes for pluggable components (inference engines, notification services)
- **Factory Pattern**: Hardware-aware component selection (AI backends, backup tools)
- **Dependency Injection**: Pass dependencies explicitly for testability

### Naming Conventions
- **Files**: snake_case for Python files
- **Classes**: PascalCase (e.g., `InferenceEngine`, `HueController`)
- **Functions/Variables**: snake_case (e.g., `analyze_color`, `detection_confidence`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_CONFIDENCE_THRESHOLD`)
- **Private Methods**: Leading underscore (e.g., `_validate_config`)

### Import Organization
```python
# Standard library imports
import os
import time
from pathlib import Path
from typing import Optional, Dict, List

# Third-party imports
import cv2
import numpy as np
import onnxruntime

# Local imports
from src.config.manager import ConfigManager
from src.ai.factory import InferenceEngineFactory
```

### Configuration Architecture
- **Centralized Config**: Single `config.ini` file for all settings
- **Type Safety**: Use dataclasses or Pydantic for configuration objects
- **Environment Overrides**: Support env vars for sensitive data
- **Validation**: Comprehensive validation with helpful error messages

### Error Handling Strategy
- **Graceful Degradation**: System continues operating when non-critical components fail
- **Comprehensive Logging**: All errors logged with context and stack traces
- **Recovery Mechanisms**: Automatic retry logic for transient failures
- **User-Friendly Messages**: Clear error messages for configuration issues

### Performance Considerations
- **Multi-threading**: Separate threads for I/O-bound operations (camera, file operations)
- **Multi-processing**: Separate processes for CPU-intensive operations (AI inference)
- **Queue Management**: Optimized inter-process communication
- **Resource Cleanup**: Proper cleanup of cameras, models, and system resources