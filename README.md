# Adaptive-HCI: LED-Based Gesture Recognition System

**Adaptive-HCI** is a hardware-software system designed for human-computer interaction using real-time video-based detection and inference of LED-encoded gestures and characters. The system leverages deep learning, real-time image processing, and embedded electronics to track and classify user inputs via LED glove signals. Built on Dr. Xiao Zhang's RoFin paper (ACM MobiSys 2023) as a contribution to his research.

---

## Repository Structure

| Folder | Description |
|--------|-------------|
| `Arduino/` | Microcontroller code for LED glove control and OOK signal encoding (multiple versions, tuned to different camera shutter speeds). |
| `Documents & Other/` | Research documents, system design, planning material, team workflow files, and project pitches. |
| `Letter & Digit Inference/` | Python-based ML pipeline for training and testing digit/letter classifiers (MNIST, DIDA, and custom datasets). Includes scripts to convert PyTorch models to TFLite. |
| `Proof of Concept/` | Early Python-based prototypes and conceptual demos to validate design hypotheses. |
| `Xamera/` | Final Android mobile app. Records and processes video input from users in real-time using deep learning backends. |
| `YOLO/` | Object detection pipeline: training configs, datasets, auto-labeling tools, model format converters, and benchmarking tools for PyTorch and TFLite YOLOv11nano models. |

---

## Features

- **LED Glove with OOK Encoding**  
  Glove LEDs are modulated using OOK (On-Off Keying) to encode signals recognizable in video recordings.

- **Android App (Xamera)**  
  Captures user gestures, streams video, detects glove position via YOLO, classifies gestures/characters using on-device TFLite models, and displays outputs in AR.

- **Letter & Digit Inference Models**  
  CNN classifiers trained on MNIST, DIDA, and custom datasets. TFLite conversion enables mobile deployment.

- **YOLOv11nano Object Detection**  
  Detects LED glove in frames using custom-trained YOLO models. Benchmarked and optimized for both PyTorch and TFLite.

- **Tooling**  
  - `Autolabeler`: Automatically generates YOLO labels using OpenCV.  
  - `ModelConverter`: Converts YOLOv11nano `.pt` models to `.tflite`.  
  - `Benchmarking`: Latency and accuracy evaluation scripts for PyTorch vs TFLite on image and video inputs.

---

## Getting Started

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-org/Adaptive-HCI.git
   cd Adaptive-HCI

2. **Setup Python Environments:**
    Letter & Digit Inference/requirements.txt
    YOLO/requirements.txt

3. **Build & Run the Android App (Xamera):**
    Open Xamera/ in Android Studio
    Configure emulator or device
    Build & run

4. **Arduino Setup:**
    Open .ino files inside Arduino/ with Arduino IDE
    Upload the code to your glove's microcontroller
