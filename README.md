TheCatBouncer: AI-Powered Pet Access Control
An intelligent, vision-based system to distinguish between your own cat and unwanted four-legged visitors. Using a camera, object detection (YOLO), and smart home integration, TheCatBouncer acts as an automated doorman, granting peace to your pet while deterring intruders.

The system operates in two phases: it passively monitors for activity using a motion-light trigger and then launches an active analysis with a high-resolution camera stream to identify the cat. Based on the cat's color, it decides whether to welcome it silently or scare off an intruder with sound and light.

🌟 Key Features
Intelligent Intruder Detection: Uses a YOLO model to specifically detect cats with high accuracy.

Owner vs. Intruder Logic: Differentiates between a pre-defined "own" cat (based on color analysis) and other "intruder" cats.

Automated Deterrents: Triggers actions like playing a scare sound or controlling smart lights to deter intruders.

Smart Home Integration: Natively controls Philips Hue lights.

Efficient Two-Phase System: Saves resources by using a low-resolution passive mode and only switching to high-resolution active analysis when triggered.

Flexible AI Backend: Supports PyTorch (CPU/NVIDIA CUDA) and OpenVINO (CPU/Intel iGPU) for maximum hardware compatibility.

Automated Data Management: Includes features for daily backups to a NAS and local disk space cleanup.

Highly Configurable: All major parameters can be adjusted in a simple config.ini file.

🔧 Setup & Installation
Requirements
Hardware:

A compatible webcam.

A computer to run the Python script (Windows/macOS/Linux).

Philips Hue Bridge and at least one light.

(Optional) A NAS for automated backups.

Software:

Python 3.10

The dependencies listed in environment.yml.

Installation Steps
Clone the repository:

git clone https://github.com/JojiAce/TheCatBouncer.git
cd TheCatBouncer

Create a Conda environment:
This is the recommended way to install the dependencies.

conda env create -f environment.yml
conda activate savebounce-env

Prepare Sound Files:
Place your scare sounds (e.g., bark.mp3, hiss.wav) into the cat_scare_sound directory.

Configure the system:
Rename config.ini.example to config.ini and edit it according to the detailed guide below. This is the most important step!

⚙️ Configuration Guide (config.ini)
This file is the heart of the system. All settings are managed here.

[Zeitsteuerung]
Controls when the system is active.

start_zeit: The time when the monitoring begins (e.g., 18:00 for 6 PM).

end_zeit: The time when the monitoring stops (e.g., 07:00 for 7 AM). The script correctly handles overnight periods.

datenmanagement_zeit: The time for daily backup and cleanup tasks (e.g., 07:10).

[PhilipsHue]
Your smart light settings.

bridge_ip: The local IP address of your Philips Hue Bridge.

app_key: Your Hue Application Key. Follow the official guide to get your key.

licht_ids: The unique IDs of the lights you want to control. You can list multiple IDs separated by commas.

[Kamera]
Camera settings for both monitoring phases.

kamera_index: The index of your camera. 0 is usually the default built-in webcam.

niedrige_aufloesung: Resolution for the passive monitoring phase (e.g., 640,480). A lower resolution saves CPU.

hohe_aufloesung: Resolution for the active analysis phase (e.g., 1920,1080). A higher resolution improves detection accuracy.

fps_niedrig / fps_hoch: Frames per second for each phase.

[Trigger]
Defines what activates the system.

helligkeit_schwelle: A value from 0-255. A pixel is considered "bright" if its grayscale value is above this. 220 is a good starting point for a bright outdoor light.

helligkeit_pixel_prozent: The percentage of pixels that must be "bright" to trigger the system (e.g., 0.01 for 1%). This prevents triggers from small reflections.

[Bilderkennung]
The core of the AI detection.

yolo_modell_pfad: The path to the YOLO model you want to use. The project is structured to use either .pt (PyTorch) or OpenVINO model formats.

inference_device: Crucial setting!

cpu: For using the CPU. Works with both model types.

gpu: A smart setting. It will try to use a NVIDIA GPU via CUDA first. If not available, it will try to use an Intel iGPU via OpenVINO. If neither is found, it falls back to CPU.

cuda:0: To explicitly use a specific NVIDIA GPU.

katzen_klassen_id: The class ID for "cat" in your YOLO model. This can vary depending on the model's training data.

katzen_sicherheit_schwelle: The minimum confidence required to classify an object as a cat (e.g., 0.8 for 80%).

[Farbanalyse]
Defines how your own cat is identified.

untere_schwarz_hsv / obere_schwarz_hsv: The lower and upper HSV color values to define the color of your cat. You will need to calibrate these for your specific lighting and cat.

schwarz_pixel_schwelle: The percentage of pixels within the detected cat's bounding box that must match the color range to be identified as your own (e.g., 0.5 for 50%).

[Aktionen]
Defines the system's reactions.

eindringling_licht_minuten: How long the lights should stay on after an intruder is detected.

zeige_live_fenster: Set to true to show a live debug window with detection boxes. Set to false to run headlessly. This can also be overridden by command-line arguments.

[NAS] & [Speicherverwaltung]
For data backup and cleanup.

nas_ip, nas_benutzer, etc.: Your NAS connection details for automated backups.

min_freier_speicher_gb: The minimum amount of free space to maintain on the local drive.

max_alter_tage: The maximum age of local files before they are deleted by the cleanup task.

▶️ Usage
To run the script, activate your environment and execute the main file (replace YourScriptName.py with your actual filename):

python YourScriptName.py

You can use command-line arguments to override settings from config.ini:

Force Live Preview:

python YourScriptName.py --live-preview

Disable Live Preview:

python YourScriptName.py --no-live-preview

Override Inference Device:

python YourScriptName.py --device cpu
