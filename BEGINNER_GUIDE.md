# TheCatBouncer v2.0 - Complete Beginner's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Understanding the Components](#understanding-the-components)
7. [Hardware Setup](#hardware-setup)
8. [Troubleshooting](#troubleshooting)
9. [Customization](#customization)
10. [Advanced Usage](#advanced-usage)

## Introduction

TheCatBouncer v2.0 is an AI-powered pet access control system designed to detect cats and automatically activate deterrents to keep them away from restricted areas. This advanced system uses computer vision, artificial intelligence, and smart home integration to provide a humane and effective solution for pet management.

The system features:
- Real-time cat detection using AI
- Multiple deterrent methods (lighting, audio)
- Smart home integration (Philips Hue)
- Audio deterrent system
- Advanced scheduling and maintenance
- Performance monitoring and logging
- Cross-platform compatibility

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 7+, macOS 10.12+, or Linux (Ubuntu 18.04+)
- **CPU**: Intel/AMD dual-core processor (4th generation or newer)
- **RAM**: 4 GB available system memory
- **Python**: Python 3.10 or higher
- **Storage**: 500 MB available space
- **Camera**: USB webcam or built-in laptop camera
- **Internet Connection**: Required for initial setup and updates

### Recommended Requirements
- **Operating System**: Windows 10+, macOS 11+, or Linux (Ubuntu 20.04+)
- **CPU**: Intel/AMD quad-core processor (8th generation or newer)
- **RAM**: 8 GB available system memory
- **GPU**: NVIDIA GPU with CUDA support (for faster AI processing) or Intel integrated graphics
- **Python**: Python 3.10 or higher
- **Storage**: 1 GB available space
- **Camera**: 720p or higher resolution webcam
- **Internet Connection**: Broadband for optimal AI model performance

### Optional Hardware
- Philips Hue smart lighting system for light-based deterrents
- External speakers for audio deterrents

## Installation

### Method 1: Manual Installation (Recommended)

1. **Download the Source Code**
   - Navigate to the project directory: `cd /path/to/TheCatBouncer`
   - If you haven't already, download or clone the repository

2. **Verify Python Installation**
   ```bash
   python3 --version
   ```
   Make sure you have Python 3.10 or higher installed.

3. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   ```
   
   On Windows:
   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**
   
   On Linux/macOS:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

5. **Upgrade pip**
   ```bash
   pip install --upgrade pip
   ```

6. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Verify Installation**
   ```bash
   python3 -c "import cv2; import numpy; print('Dependencies installed successfully')"
   ```

### Method 2: Using Environment Manager (For Advanced Users)

If you're using conda or another environment manager:

1. **Create a new environment**
   ```bash
   conda create -n thecatbouncer python=3.10
   conda activate thecatbouncer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or use conda install for some packages if available
   ```

## Configuration

### Understanding the Configuration File

The main configuration file is `config.ini` located in the root directory. This file contains all the settings for the application:

```ini
[SYSTEM]
# System-wide settings
debug = False
log_level = INFO
data_directory = /path/to/data
backup_directory = /path/to/backup

[AI]
# AI model settings
model_path = models/cat_detection.onnx
confidence_threshold = 0.7
max_detection_attempts = 10

[DETECTION]
# Detection parameters
active_analysis_interval = 10
passive_monitoring_interval = 2
brightness_threshold = 0.5

[SMART_HOME]
# Philips Hue settings
hue_bridge_ip = 
hue_username = 
hue_deterrent_group = Living Room
hue_brightness = 254
hue_transition_time = 1

[AUDIO]
# Audio settings
audio_enabled = True
audio_file_path = audio/deterrent.wav
audio_volume = 0.8

[NOTIFICATION]
# Notification settings
email_enabled = False
email_smtp_server = smtp.gmail.com
email_smtp_port = 587
email_address = your_email@gmail.com
email_password = your_app_password
telegram_enabled = False
telegram_token = 
telegram_chat_id = 

[PERFORMANCE]
# Performance settings
camera_buffer_size = 4
max_concurrent_analyses = 2
cpu_affinity = 0-3
```

### Required Configuration Steps

1. **Set up the Data Directory**
   - Modify `data_directory` in the `[SYSTEM]` section to point to where you want to store logs and captured images
   - Create this directory if it doesn't exist

2. **Configure AI Model Path**
   - Set `model_path` in the `[AI]` section to the location of your trained model
   - The default path assumes the model is in a `models/` directory

3. **Adjust Detection Sensitivity**
   - Modify `confidence_threshold` based on your camera's quality and lighting conditions
   - Lower values (0.5) = more detections, higher chance of false positives
   - Higher values (0.9) = fewer detections, more accurate but might miss cats

### Optional Configuration Steps

1. **Setup Smart Home Integration (Philips Hue)**
   - Find your Hue Bridge IP address (usually shown in the Hue app)
   - Generate a username by following Philips Hue API documentation
   - Update `hue_bridge_ip` and `hue_username` in the `[SMART_HOME]` section

2. **Configure Notifications**
   - For email notifications, set `email_enabled = True` and provide your SMTP details
   - For Telegram, set `telegram_enabled = True` and provide your bot token and chat ID

3. **Adjust Audio Settings**
   - Place your deterrent audio files in the audio directory
   - Update `audio_file_path` to point to your chosen audio file

## Running the Application

### Basic Usage

1. **Make sure you're in the correct directory**
   ```bash
   cd /path/to/TheCatBouncer
   ```

2. **Activate your virtual environment**
   ```bash
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Run the main application**
   ```bash
   python -m src.main
   ```

### Command Line Options

The application supports several command-line options:

```bash
# Run in debug mode
python -m src.main --debug

# Run with a specific configuration file
python -m src.main --config /path/to/custom/config.ini

# Run without audio deterrents
python -m src.main --no-audio

# Run without smart home integration
python -m src.main --no-smart-home

# Run with verbose logging
python -m src.main --verbose
```

### Using the ConfigManager Class

If you want to programmatically access configuration:

```python
from src.config.config_manager import ConfigManager

# Initialize configuration
config = ConfigManager()

# Access configuration values
camera_interval = config.get_detection_config().active_analysis_interval
ai_threshold = config.get_ai_config().confidence_threshold

# Update configuration at runtime
config.update_config({"AI": {"confidence_threshold": 0.8}})
```

## Understanding the Components

### Core Components Overview

TheCatBouncer v2.0 is built with a modular architecture. Here's what each component does:

#### 1. ConfigManager
- Manages all application settings
- Validates configuration values
- Handles environment variable overrides
- Provides type-safe access to configuration

#### 2. CameraManager
- Controls camera hardware
- Captures real-time video frames
- Manages camera buffer and settings
- Handles multiple camera formats

#### 3. PassiveMonitor
- Monitors for basic triggers (light, movement)
- Uses a state machine approach
- Low-power operation when possible
- Triggers active analysis when needed

#### 4. ActiveAnalyzer
- Performs real-time AI analysis
- Uses multiprocessing for speed
- Handles multiple concurrent analyses
- Integrates with various AI backends

#### 5. ColorAnalyzer
- Uses HSV color analysis for detection
- Implements LLM-based color detection
- Provides backup detection methods
- Supports different color spaces

#### 6. HueController
- Manages Philips Hue smart lights
- Implements deterrent lighting sequences
- Handles connection and authentication
- Provides graceful degradation

#### 7. AudioManager
- Plays audio deterrents
- Manages volume levels
- Handles different audio formats
- Provides cross-platform support

#### 8. NotificationService
- Sends email and Telegram notifications
- Manages notification queues
- Handles delivery failures
- Supports multiple notification types

#### 9. StorageManager
- Manages data storage and backup
- Handles NAS synchronization
- Implements data retention policies
- Logs all system events

#### 10. ScheduleManager
- Manages operation schedules
- Implements maintenance windows
- Handles time-based operations
- Supports recurring tasks

#### 11. PerformanceMonitor
- Tracks system performance
- Monitors resource usage
- Provides real-time metrics
- Generates performance reports

#### 12. LoggingService
- Manages application logging
- Implements log rotation
- Provides structured logging
- Supports different log levels

### How Components Work Together

1. **Passive Monitoring**: The system runs in a low-power state, monitoring brightness and basic triggers
2. **Trigger Detection**: When a trigger is detected, the ActiveAnalyzer is activated
3. **AI Analysis**: The ActiveAnalyzer processes camera frames using AI models
4. **Decision Making**: If a cat is detected, deterrents are activated
5. **Smart Home Integration**: Lights are controlled via HueController
6. **Audio Deterrents**: AudioManager plays deterrent sounds
7. **Notifications**: Relevant events are sent via NotificationService
8. **Logging**: All events are recorded by LoggingService
9. **Storage**: Data is managed by StorageManager
10. **Performance**: PerformanceMonitor tracks system health

## Hardware Setup

### Camera Setup

1. **Connect your camera**: USB webcam or built-in laptop camera
2. **Position the camera**: Point it toward the area you want to monitor
3. **Adjust the angle**: Ensure the detection area is fully visible
4. **Test the camera**: Verify it works with the system

### Smart Home Integration (Philips Hue)

1. **Set up your Hue Bridge**: Connect to your network
2. **Add lights to your system**: Follow Hue app instructions
3. **Find the Bridge IP**: Check your router or Hue app
4. **Create a username**: Use the Hue API to generate a username
5. **Configure the system**: Update your config.ini with the details

### Audio Setup

1. **Connect speakers**: External speakers or built-in laptop speakers
2. **Test audio output**: Verify volume levels are appropriate
3. **Place speakers correctly**: Ensure audio deterrents reach the target area
4. **Test deterrent sounds**: Make sure they're effective but not too loud

## Troubleshooting

### Common Issues and Solutions

#### Camera Issues
**Problem**: Camera not detected
- **Solution**: 
  - Check physical connection
  - Verify no other application is using the camera
  - Try different USB ports
  - Update camera drivers

**Problem**: Poor image quality
- **Solution**:
  - Adjust camera positioning
  - Improve lighting conditions
  - Clean camera lens
  - Check for obstructions

#### AI Model Issues
**Problem**: No detections or false positives
- **Solution**:
  - Adjust `confidence_threshold` in config.ini
  - Verify model file path is correct
  - Check lighting conditions
  - Update to a better trained model

#### Smart Home Issues
**Problem**: Hue lights not responding
- **Solution**:
  - Verify Bridge IP and username
  - Check network connectivity
  - Ensure Hue Bridge is powered on
  - Test with Hue app first

#### Audio Issues
**Problem**: No audio output
- **Solution**:
  - Check speaker connections
  - Verify volume levels
  - Ensure audio file exists and is accessible
  - Test with system audio settings

#### Performance Issues
**Problem**: Slow detection or high CPU usage
- **Solution**:
  - Reduce `max_concurrent_analyses` in config
  - Lower camera resolution
  - Adjust analysis intervals
  - Close other applications to free up resources

### Log Analysis

Check the log files for errors:

1. **Locate log files**: Usually in the data directory specified in config.ini
2. **Look for ERROR or CRITICAL messages**
3. **Check timestamps** around when issues occurred
4. **Search for specific error messages** to find solutions

### Environment Issues

**Problem**: Dependencies not installing
- **Solution**:
  - Use a virtual environment
  - Check Python version requirements
  - Try installing packages individually
  - Use `--user` flag if system permissions are an issue

## Customization

### Modifying Detection Sensitivity

1. **In config.ini**:
   - Increase `confidence_threshold` to reduce false positives
   - Decrease `confidence_threshold` to catch more detections

2. **In code**:
   ```python
   from src.config.config_manager import DetectionConfig
   
   detection_config = DetectionConfig(
       confidence_threshold=0.75,
       active_analysis_interval=5
   )
   ```

### Adding Custom AI Models

1. **Place your model in the models directory**
2. **Update the `model_path` in config.ini**
3. **Ensure your model is compatible** with the inference engines
4. **Test the new model** with sample images

### Custom Audio Deterrents

1. **Place audio files in the audio directory**
2. **Update `audio_file_path` in config.ini**
3. **Test the new audio file** to ensure it's effective
4. **Consider the volume level** to avoid disturbing neighbors

### Custom Scheduling

1. **Use the ScheduleManager** to create custom schedules
2. **Set up maintenance windows** when deterrence should be disabled
3. **Configure time-based behavior** for different times of day

## Advanced Usage

### Performance Tuning

1. **CPU Optimization**:
   - Adjust `max_concurrent_analyses` based on your CPU cores
   - Set `cpu_affinity` to dedicate cores to the application
   - Use appropriate camera resolution (720p is often sufficient)

2. **GPU Acceleration**:
   - Install CUDA-enabled packages if using NVIDIA GPU
   - Use OpenVINO for Intel hardware optimization
   - Enable hardware acceleration in AI engines

3. **Memory Management**:
   - Monitor RAM usage with PerformanceMonitor
   - Adjust camera buffer size
   - Implement proper data retention policies

### Integration with Other Systems

1. **Home Automation**:
   - Integrate with Home Assistant, SmartThings, etc.
   - Use NotificationService to trigger other smart devices
   - Create custom notification handlers

2. **Data Analysis**:
   - Export detection logs for analysis
   - Create custom reporting tools
   - Integrate with monitoring systems

3. **Web Interface**:
   - Consider adding a web UI for remote monitoring
   - Create REST API endpoints for external control
   - Implement real-time dashboard

### Development and Testing

1. **Unit Testing**:
   - Run tests regularly: `python -m pytest tests/`
   - Add tests for new functionality
   - Mock hardware dependencies for testing

2. **Performance Testing**:
   - Use PerformanceMonitor to identify bottlenecks
   - Test with different camera feeds
   - Monitor resource usage under load

3. **Integration Testing**:
   - Test all components together
   - Verify behavior with missing hardware
   - Test error handling and recovery

## Maintenance

### Regular Maintenance Tasks

1. **Check Logs**: Review logs regularly for issues
2. **Update Models**: Keep AI models up to date
3. **Backup Data**: Regular backups of configuration and logs
4. **Update Dependencies**: Keep packages updated
5. **Hardware Checks**: Verify camera, lights, and audio are working

### Data Management

1. **Log Rotation**: Automatic log rotation is implemented
2. **Data Retention**: Configure retention policies in StorageManager
3. **Backup Schedules**: Automated NAS backup is available

## Security Considerations

1. **Configuration Security**: Store API keys and passwords securely
2. **Network Security**: Use strong passwords and encrypted connections
3. **Physical Security**: Protect the device and camera from physical access
4. **Software Updates**: Keep the system and dependencies updated

## Conclusion

TheCatBouncer v2.0 provides a comprehensive, modular solution for pet access control. With proper setup and configuration, it offers reliable, AI-powered detection and deterrent capabilities. The system is designed to be stable, maintainable, and extensible, with components that can be modified or extended as needed.

Follow this guide for installation and configuration, and refer to the documentation in the code for more detailed information about specific components. The system is designed to be robust and handle various failure scenarios gracefully, making it suitable for long-term deployment.

Remember to regularly check logs, perform maintenance, and adjust settings based on your specific environment and needs. With proper care, TheCatBouncer v2.0 will provide reliable service for keeping cats away from restricted areas.