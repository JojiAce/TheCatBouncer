# Requirements Document

## Introduction

TheCatBouncer is an AI-powered pet access control system that uses computer vision to distinguish between the owner's cat and intruder cats, automatically deterring unwelcome felines with lights and sounds while allowing the owner's cat free access. The system operates in two phases: a low-resource passive monitoring phase that detects motion/light changes, followed by a high-resolution active analysis phase that performs AI inference and color analysis to identify cats and determine if they belong to the owner.

## Requirements

### Requirement 1: Core System Architecture

**User Story:** As a pet owner, I want a reliable two-phase monitoring system that efficiently uses system resources while providing accurate cat detection and identification.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL initialize all components (camera, AI model, configuration) without errors
2. WHEN in passive monitoring mode THEN the system SHALL use low-resolution video (640x480) at 5 FPS to minimize resource usage
3. WHEN a brightness trigger is detected THEN the system SHALL automatically switch to active analysis mode
4. WHEN in active analysis mode THEN the system SHALL use high-resolution video (1920x1080) at 30 FPS for accurate detection
5. WHEN active analysis completes or times out THEN the system SHALL return to passive monitoring mode
6. WHEN any component fails THEN the system SHALL log the error and attempt graceful recovery

### Requirement 2: High-Performance AI-Powered Cat Detection

**User Story:** As a pet owner, I want the system to accurately detect cats using AI models with maximum throughput performance, automatic hardware optimization, and configurable confidence thresholds.

#### Acceptance Criteria

1. WHEN a frame is processed THEN the system SHALL use YOLO11 or equivalent models with multi-threaded pipeline for maximum throughput
2. WHEN multiple AI backends are available THEN the system SHALL automatically select optimal format: NVIDIA GPU (ONNX/SafeTensors) > AMD/Intel GPU (ONNX) > iGPU (OpenVINO) > CPU (OpenVINO/SafeTensors)
3. WHEN running on Apple Silicon THEN the system SHALL prefer CoreML format for optimal performance
4. WHEN GPU is selected but unavailable THEN the system SHALL fall back to OpenVINO iGPU if available, otherwise multi-threaded CPU with OpenVINO
5. WHEN processing pipeline operates THEN it SHALL use separate threads for capture, preprocessing, inference, and postprocessing to maximize throughput
6. WHEN inference engine runs THEN it SHALL support configurable worker threads based on hardware capabilities
7. WHEN a cat is detected with confidence >= threshold THEN the system SHALL proceed to color analysis
8. WHEN no cat is detected within timeout period THEN the system SHALL return to passive mode
9. WHEN detection confidence is below threshold THEN the system SHALL ignore the detection
10. WHEN processing frames THEN the system SHALL maintain target FPS through optimized multi-threaded pipeline architecture
11. WHEN system operates THEN it SHALL maximize throughput using parallel processing for all pipeline stages
12. WHEN performance bottlenecks occur THEN the system SHALL log detailed metrics and automatically optimize thread allocation

### Requirement 3: Owner Cat Identification

**User Story:** As a pet owner, I want the system to distinguish my cat from other cats using color analysis so only intruders are deterred.

#### Acceptance Criteria

1. WHEN a cat is detected THEN the system SHALL perform color analysis on the bounding box region
2. WHEN using HSV color analysis THEN the system SHALL compare pixel colors against configured HSV ranges
3. WHEN using LLM color analysis THEN the system SHALL send the image to a local Ollama instance for analysis
4. WHEN the cat matches owner criteria THEN the system SHALL take no action and return to passive mode
5. WHEN the cat does not match owner criteria THEN the system SHALL classify it as an intruder
6. WHEN color analysis fails THEN the system SHALL log the error and default to intruder classification for safety

### Requirement 4: Smart Home Integration

**User Story:** As a smart home user, I want the system to integrate with my Philips Hue lights for illumination and deterrent effects.

#### Acceptance Criteria

1. WHEN entering active analysis mode THEN the system SHALL turn on configured Hue lights for better visibility
2. WHEN an intruder is detected THEN the system SHALL activate deterrent lighting patterns
3. WHEN the owner's cat is identified THEN the system SHALL turn off the lights
4. WHEN Hue bridge is unavailable THEN the system SHALL continue operating without light control
5. WHEN light control fails THEN the system SHALL log the error and continue with other deterrent methods
6. WHEN system shuts down THEN the system SHALL turn off all controlled lights

### Requirement 5: Audio Deterrent System

**User Story:** As a pet owner, I want the system to play deterrent sounds to scare away intruder cats without harming them.

#### Acceptance Criteria

1. WHEN an intruder is detected THEN the system SHALL play a random sound file from the configured directory
2. WHEN multiple sound files exist THEN the system SHALL randomly select one for variety
3. WHEN sound files are in different formats THEN the system SHALL support both .wav and .mp3 formats
4. WHEN no sound files exist THEN the system SHALL log a warning and continue with other deterrent methods
5. WHEN audio playback fails THEN the system SHALL log the error and continue operation
6. WHEN deterrent duration expires THEN the system SHALL stop audio playback

### Requirement 6: Notification System

**User Story:** As a pet owner, I want to receive notifications when intruders are detected so I can monitor my property remotely.

#### Acceptance Criteria

1. WHEN an intruder is detected THEN the system SHALL send a notification with timestamp and image
2. WHEN email notifications are configured THEN the system SHALL send email with attached detection image
3. WHEN Telegram notifications are configured THEN the system SHALL send message with embedded image
4. WHEN notification service is unavailable THEN the system SHALL log the failure and continue operation
5. WHEN notifications are disabled THEN the system SHALL not attempt to send any notifications
6. WHEN notification fails THEN the system SHALL retry once before logging failure

### Requirement 7: Scheduled Operation

**User Story:** As a pet owner, I want to configure active hours for monitoring so the system only operates when needed.

#### Acceptance Criteria

1. WHEN current time is within configured schedule THEN the system SHALL operate normally
2. WHEN current time is outside configured schedule THEN the system SHALL enter sleep mode
3. WHEN schedule spans midnight THEN the system SHALL handle time ranges correctly (e.g., 18:00-07:00)
4. WHEN in sleep mode THEN the system SHALL check schedule every 5 minutes
5. WHEN schedule configuration is invalid THEN the system SHALL log error and operate 24/7
6. WHEN transitioning between modes THEN the system SHALL complete current operations before changing state

### Requirement 8: Data Management and Storage

**User Story:** As a system administrator, I want automated data management to prevent storage overflow and maintain system performance.

#### Acceptance Criteria

1. WHEN successful detections occur THEN the system SHALL save detection images and metadata to organized folders
2. WHEN disk space falls below threshold THEN the system SHALL automatically delete oldest files
3. WHEN daily maintenance time arrives THEN the system SHALL perform cleanup and backup operations
4. WHEN NAS backup is configured THEN the system SHALL create incremental backups using appropriate tools (rsync/robocopy)
5. WHEN files exceed maximum age THEN the system SHALL delete them during maintenance
6. WHEN backup operations fail THEN the system SHALL log errors and continue local operation

### Requirement 9: Cross-Platform Compatibility

**User Story:** As a user on different operating systems, I want the system to work consistently across Windows, macOS, and Linux.

#### Acceptance Criteria

1. WHEN running on Windows THEN the system SHALL use robocopy for NAS backups and support NVIDIA CUDA, AMD/Intel GPU via ONNX, and iGPU via OpenVINO
2. WHEN running on macOS THEN the system SHALL use rsync for NAS backups and prefer CoreML for Apple Silicon, ONNX for Intel Macs
3. WHEN running on Linux THEN the system SHALL use rsync for NAS backups and support NVIDIA CUDA, AMD/Intel GPU via ONNX, and iGPU via OpenVINO
4. WHEN Apple Silicon is detected THEN the system SHALL use CoreML format for optimal Neural Engine and GPU utilization
5. WHEN iGPU is available THEN the system SHALL utilize integrated GPU acceleration through OpenVINO (the only format supporting iGPU)
6. WHEN camera access varies by OS THEN the system SHALL handle different camera interfaces appropriately
7. WHEN file paths differ by OS THEN the system SHALL use cross-platform path handling with pathlib
8. WHEN dependencies vary by OS THEN the system SHALL provide OS-specific installation instructions
9. WHEN OS-specific features are unavailable THEN the system SHALL gracefully disable those features and log the limitation

### Requirement 10: Configuration Management

**User Story:** As a user, I want a single configuration file that controls all system behavior without requiring code changes.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL load all configuration from a single config.ini file
2. WHEN configuration values are missing THEN the system SHALL use sensible defaults
3. WHEN configuration values are invalid THEN the system SHALL log errors and use defaults
4. WHEN configuration is updated THEN the system SHALL support hot-reloading where safe
5. WHEN sensitive configuration is required THEN the system SHALL provide clear setup instructions
6. WHEN configuration validation fails THEN the system SHALL provide helpful error messages

### Requirement 11: Logging and Debugging

**User Story:** As a system administrator, I want comprehensive logging and debugging capabilities to troubleshoot issues and monitor performance.

#### Acceptance Criteria

1. WHEN any operation occurs THEN the system SHALL log appropriate level messages (DEBUG, INFO, WARNING, ERROR)
2. WHEN detections occur THEN the system SHALL log detection details with timestamps and confidence scores
3. WHEN performance metrics are available THEN the system SHALL log FPS and latency information
4. WHEN debug mode is enabled THEN the system SHALL display live video window with detection overlays
5. WHEN errors occur THEN the system SHALL log full stack traces for debugging
6. WHEN log files grow large THEN the system SHALL implement log rotation to manage disk space