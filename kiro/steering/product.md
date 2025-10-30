# Product Overview

## TheCatBouncer - AI-Powered Pet Access Control

TheCatBouncer is an intelligent pet access control system that uses computer vision and AI to distinguish between the owner's cat and intruder cats. The system automatically deters unwelcome felines with lights and sounds while allowing the owner's cat free access.

### Core Functionality

- **Two-Phase Monitoring**: Efficient passive monitoring (low-res, 5 FPS) switches to active analysis (high-res, 30 FPS) when motion is detected
- **AI-Powered Detection**: Uses YOLO11 models with automatic hardware optimization for accurate cat detection
- **Owner Recognition**: Identifies the owner's cat through HSV color analysis or local LLM processing
- **Smart Deterrents**: Activates Philips Hue lights and plays random scare sounds for intruders
- **Cross-Platform**: Runs on Windows, macOS, and Linux with automatic hardware optimization

### Key Features

- Automatic hardware backend selection (CUDA → OpenVINO → CPU)
- Scheduled operation with configurable active hours
- Smart home integration (Philips Hue lights)
- Notification system (email, Telegram)
- Automated data management and NAS backups
- Comprehensive logging and debugging capabilities

### Target Users

Pet owners who want to:
- Control access through cat flaps or doors
- Monitor garden areas or patios
- Deter stray cats while allowing their own cat access
- Integrate pet monitoring with existing smart home systems