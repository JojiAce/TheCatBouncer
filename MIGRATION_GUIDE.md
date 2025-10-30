# TheCatBouncer v2.0 Migration Guide

This guide explains how to migrate from the legacy TheCatBouncer codebase to the new v2.0 architecture.

## Overview

TheCatBouncer v2.0 represents a complete architectural redesign with:
- Clean, modular code organization
- Proper separation of concerns
- Type-safe configuration management
- Improved performance and maintainability
- Better component isolation

## Key Changes

### 1. Directory Structure
**Old Structure:**
```
TheCatBouncer/
├── main-file_and_active_analysis_pipeline.py
├── color_analyzer.py
├── passive_analyzer.py
├── hue_controller.py
├── data_manager.py
├── utility_recorder.py
├── inference_factory.py
├── NewPickerEnigneforActive.py
└── ...
```

**New Structure:**
```
TheCatBouncer/
├── src/
│   ├── components/
│   ├── config/
│   ├── engines/
│   ├── interfaces/
│   ├── managers/
│   └── utils/
├── src/main.py
├── requirements.txt
├── setup.py
└── README_V2.md
```

### 2. Component Architecture
The new architecture uses a clean separation pattern:

- **`interfaces/`**: Abstract base classes defining component contracts
- **`engines/`**: AI inference backend implementations
- **`managers/`**: High-level component orchestrators
- **`config/`**: Configuration management system
- **`utils/`**: Helper functions and utilities

### 3. Configuration Management
**Old**: Inline configuration in code or basic INI files
**New**: Type-safe configuration with validation and dataclasses

```python
# Old style (in multiple files)
config = configparser.ConfigParser()
config.read('config.ini')

# New style
from src.config.config_manager import ConfigManager
config_manager = ConfigManager('config.ini')
config_data = config_manager.config_data
```

### 4. Inference Engine System
**Old**: Mixed implementation with limited backend support
**New**: Factory pattern with comprehensive backend detection and selection

```python
# New inference engine factory
from src.engines.engine_factory import create_inference_engine

model_paths = ['/path/to/model.onnx', '/path/to/model.xml']
engine, class_names = create_inference_engine(
    model_paths, 
    device='auto'  # Automatically select best device
)
```

## Migration Steps

### 1. Update Configuration
The configuration format remains INI-based but with new sections:

| Old Section | New Section | Notes |
|-------------|-------------|-------|
| `[Input]` | `[Camera]` | Renamed for consistency |
| `[Backend]` | `[ImageRecognition]` | More descriptive |
| `[PassivAnalyzerCamera]` | `[Camera]` | Consolidated |
| `[PassivAnalyzerTrigger]` | `[Trigger]` | Separate section |

Update your `config.ini` file according to the new schema.

### 2. Code Migration

#### Legacy Pipeline → New Architecture
```python
# Old approach (monolithic)
run_passive_analysis()
if trigger_detected:
    event_path = run_active_analysis()
    if event_path:
        is_cat_black = analyze_cat_color(event_path)

# New approach (component-based)
passive_monitor = PassiveMonitor(config)
if passive_monitor.is_triggered():
    active_analyzer = ActiveAnalyzer(config, engine)
    detection_path = active_analyzer.start_analysis()
    if detection_path:
        color_analyzer = ColorAnalyzer(config)
        is_cat_black = color_analyzer.analyze_color(image_path, bbox)
```

### 3. Dependency Updates
Update your requirements file to match the new dependencies:

```txt
# New requirements.txt
numpy>=1.21.0
opencv-python>=4.5.0
onnxruntime>=1.14.0
openvino-dev>=2023.0.0
torch>=1.12.0
safetensors>=0.3.0
coremltools>=6.0.0
pygame>=2.0.0
phue>=1.1
pyTelegramBotAPI>=4.0.0
psutil>=5.8.0
GPUtil>=1.4.0
```

### 4. Entry Point
The main entry point has changed:

```python
# Old: Run main-file_and_active_analysis_pipeline.py
# New: Run via module
python -m src.main
```

## Breaking Changes

### Removed Components
- `utility_recorder.py`: Split into `audio_manager.py` and logging system
- `NewPickerEnigneforActive.py`: Replaced by `engine_factory.py`
- `LLM_COLOR_DETECTOR.py`: Integrated into `color_analyzer.py`

### Renamed Methods
- `run_passive_analysis()` → `PassiveMonitor.start_monitoring()`
- `run_active_analysis()` → `ActiveAnalyzer.start_analysis()`
- `analyze_cat_color()` → `ColorAnalyzer.analyze_color()`

### New Configuration Options
The new system adds these configuration options:
- `storage_management` section
- `time_management` section
- Hardware optimization settings
- Advanced logging options

## Performance Improvements

| Aspect | Old | New |
|--------|-----|-----|
| Code Organization | Monolithic | Modular |
| Configuration | Basic validation | Type-safe with validation |
| Inference | Limited backends | Auto-backend selection |
| Threading | Basic multiprocessing | Optimized pipeline |
| Logging | Print statements | Structured logging |
| Error Handling | Basic | Comprehensive |

## Best Practices for Migration

1. **Start with Configuration**: Update your config.ini first
2. **Component by Component**: Migrate one component at a time
3. **Test Thoroughly**: The new system has extensive unit tests
4. **Use the Factory**: Leverage the inference engine factory for backend selection
5. **Check Logging**: The new logging system provides better diagnostics

## Support

If you encounter issues during migration:

1. Check the [v2.0 README](README_V2.md) for updated usage instructions
2. Review the unit tests in `src/managers/test_*.py` for usage examples  
3. Open an issue in the GitHub repository with specific migration questions

## Rollback

If you need to temporarily revert to the old version:

```bash
# Backup your config
cp config.ini config_v2_backup.ini

# Restore old config
cp config.ini.bak config.ini  # If you kept the original

# Use old entry point
python main-file_and_active_analysis_pipeline.py
```

---

*Migration complete? Update your documentation and enjoy the improved architecture!*