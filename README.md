# Adaptive-HCI

A hardware-software system for real-time gesture recognition using LED-encoded glove signals. The user wears an LED glove whose lights are modulated with OOK (On-Off Keying) encoding; a phone camera running the Xamera app detects the glove, tracks its position, and classifies the drawn gesture as a letter or digit using on-device deep learning models.

Built as a contribution to Dr. Xiao Zhang's RoFin research (ACM MobiSys 2023).

## Repository Structure

| Directory | Description |
|---|---|
| `Arduino/` | Microcontroller firmware for the LED glove. Two versions: `original_6_nodes_v2` and `optimized_glove`, tuned for different camera shutter speeds. |
| `Documents & Other/` | Research documents, system design write-ups, planning material, and project pitches. |
| `Letter & Digit Inference/` | Python ML pipeline for training and evaluating letter and digit classifiers on MNIST, DIDA, and custom datasets. Includes a model converter for exporting PyTorch models to TFLite. |
| `Proof of Concept/` | Early Python prototypes used to validate core design assumptions before the Android app was built. |
| `Xamera/` | Android application. Detects the LED glove via YOLOv11nano, draws a Kalman-smoothed spline trace of its path, and runs on-device TFLite inference to classify the gesture. |
| `YOLO/` | Object detection pipeline: training configs, datasets, auto-labeling tools, model converters, and benchmarking scripts for PyTorch and TFLite YOLOv11nano models. |

## System Overview

**LED Glove (Arduino)**
The glove encodes signals using OOK modulation, making the LED pulses detectable and distinguishable in standard video recordings.

**Object Detection (YOLO)**
A YOLOv11nano model, trained on a custom dataset, locates the LED glove in each camera frame. The `YOLO/Tools/` directory contains an auto-labeler (OpenCV-based), a PyTorch-to-TFLite converter, and separate benchmarking suites for both runtimes.

**Gesture Tracing and Classification (Xamera)**
The Android app tracks the bounding box center across frames, applies a Kalman filter for noise reduction, and fits a spline to produce a clean motion trace. That trace is exported as a 28x28 bitmap and run through a TFLite CNN classifier to produce a letter or digit prediction.

**Inference Models (Letter & Digit Inference)**
CNN classifiers trained on MNIST, DIDA, and a custom handwritten dataset. Exported to TFLite for on-device inference in Xamera.

## Getting Started

### Prerequisites

- Android Studio (Hedgehog or later) with NDK installed
- Python 3.9+ for the ML pipeline components
- Arduino IDE for flashing the glove firmware
- JDK 17 (shipped with Android Studio)

### Clone

```bash
git clone https://github.com/your-org/Adaptive-HCI.git
cd Adaptive-HCI
```

### Xamera (Android App)

The OpenCV native libraries are not committed to the repository due to their size (~300 MB). You must download and place them before building.

```bash
cd /tmp
curl -L -o opencv-sdk.zip https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-android-sdk.zip
unzip -q opencv-sdk.zip
cp -r OpenCV-android-sdk/sdk/native/libs /path/to/Adaptive-HCI/Xamera/OpenCV-4.10.0/native/libs
rm -rf opencv-sdk.zip OpenCV-android-sdk
```

Replace `/path/to/Adaptive-HCI` with your actual clone path. Once the libraries are in place, open the `Xamera/` directory in Android Studio and run the app on a device or emulator (API 26+, arm64-v8a or x86_64).

### Python Environments

```bash
# Letter and digit classifiers
cd "Letter & Digit Inference"
pip install -r requirements.txt

# YOLO training and tooling
cd YOLO
pip install -r requirements.txt
```

### Arduino

Open the `.ino` files in `Arduino/optimized_glove/` or `Arduino/original_6_nodes_v2/` with Arduino IDE and upload to the glove's microcontroller. Choose the version that matches your camera's shutter speed.

## Tech Stack

| Component | Technology |
|---|---|
| Android app | Kotlin, CameraX, OpenCV 4.10.0, LiteRT (TFLite), Apache Commons Math |
| Object detection | YOLOv11nano (PyTorch training, TFLite inference) |
| Gesture smoothing | Kalman filter + cubic spline interpolation |
| Inference models | PyTorch (training), TFLite (on-device) |
| Glove firmware | Arduino C++ |
| ML pipeline | Python, PyTorch |
