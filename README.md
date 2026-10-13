# Gaze Keyboard 👁️⌨️

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An assistive computer vision on-screen virtual keyboard that enables hands-free text entry using webcam-based eye gaze tracking and intentional blink detection.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Experimental Results & Benchmark](#-experimental-results--benchmark)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Keyboard Layout & Controls](#-keyboard-layout--controls)
- [Known Issues & Future Work](#-known-issues--future-work)
- [Attribution & References](#-attribution--references)
- [License](#-license)

---

## 🌟 Overview

The **Gaze Keyboard** is designed as an accessibility (AAC) tool for individuals with limited motor capabilities. Using a standard consumer webcam without requiring proprietary or expensive eye-tracking hardware, the system:
1. Tracks facial orientation and pupil position in real-time.
2. Projects the user's estimated gaze direction onto an interactive virtual keyboard.
3. Detects intentional eye blinks via the **Eye Aspect Ratio (EAR)** to select and type keys.

---

## 🧠 System Architecture

The pipeline processes input through four core modules:

```
 Webcam Feed (OpenCV)
        │
        ▼
 MediaPipe FaceMesh (478 Landmarks with Refined Irises)
        │
        ├───> 3D Head Pose & Gaze Vector (gaze.py)
        │         ├── cv2.solvePnP (Anthropometric 3D Face Model)
        │         ├── cv2.estimateAffine3D (Pupil-to-Model Space)
        │         └── Exponential Moving Average (EMA) Filter
        │
        └───> Blink Detection (helpers.py)
                  ├── Dual Eye Aspect Ratio (EAR) computation
                  └── Debounce Guard (1.8s cooldown)
                            │
                            ▼
           Hit-Testing & Layout Engine (keyboard.py)
                            │
                            ▼
           Real-Time OpenCV Virtual Display & HUD
```

### 1. Gaze Vector Estimation (`gaze.py`)
- Extracts 6 anchor landmarks on the face (nose tip, chin, eye corners, mouth corners) and fits them to canonical anthropometric 3D coordinates using `cv2.solvePnP`.
- Maps the 2D pupil center (Landmark `468`) into 3D head space via `cv2.estimateAffine3D`.
- Extrapolates a 3D gaze ray from the eyeball center outwards and re-projects it back to 2D image coordinates using `cv2.projectPoints`.
- Stabilizes tracking jitter using an Exponential Moving Average (EMA) filter.

### 2. Blink Classification (`helpers.py`)
- Employs the formula proposed by Soukupová and Čech (2016):
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
- Eye closure is recognized when both eyes drop below `EAR_THRESHOLD = 0.20`.
- A debounce cooldown (`BLINK_INTERVAL = 1.8s`) prevents accidental double-keystrokes.

---

## ✨ Key Features

- **Hands-Free Text Entry:** Type complete phrases without physical contact.
- **Accurate Hit-Testing:** 1-to-1 matching between visible key boundaries and gaze collision detection.
- **Dedicated Text Display Header:** Top-level banner displays current typed buffer with a cursor, preventing text from overlapping keyboard keys.
- **Utility Keys:** Full support for `SPACE`, `BACKSPACE`, and `CLEAR`.
- **Heads-Up Display (HUD):** Real-time monitoring of EAR values, blink status, target key, and debounce cooldown bar.
- **No Specialized Hardware Required:** Operates on standard RGB webcams (720p or 1080p).

---

## 📂 Repository Structure

```
.
├── main.py              # Main execution loop, event handling, and HUD
├── gaze.py              # 3D head pose and gaze vector estimation + EMA filter
├── helpers.py           # Coordinate conversions and Eye Aspect Ratio (EAR) math
├── keyboard.py          # VirtualKeyboard class with hit-testing and OpenCV rendering
├── requirements.txt     # Python package dependencies
├── .gitignore           # Standard git ignore rules
├── Gaze Report.pdf      # Experimental evaluation report
├── GAZE_DEMO.mp4        # Screen recording demo of the keyboard in action
└── README.md            # Comprehensive project documentation
```

---

## 📊 Experimental Results & Benchmark

Based on the study documented in [`Gaze Report.pdf`](./Gaze%20Report.pdf) by **Youssef Ahmed Hany** (ID: 220427):

| Parameter | Physical Keyboard Typing | Gaze Keyboard Typing |
| :--- | :--- | :--- |
| **Target Sentence** | `"You Are Nice"` | `"You Are Nice"` |
| **Output Produced** | `"You Are Nice"` | `"You Arre Nicre"` |
| **Completion Time** | **4.2 seconds** | **61.0 seconds** (~14.5x slower) |
| **Typing Accuracy** | **99.50%** | **83.33%** |

### Benchmark Analysis:
1. **Speed Penalty:** The 1.8–2.0s blink debounce cooldown creates an inherent ceiling on typing speed (~20–30 characters per minute max).
2. **Double-Strike Errors:** Intentionally closing the eyelids triggers involuntary micro-saccades and pupil movement right before full closure (eye-closure drift), causing occasional double hits (e.g. `rr` in `"Arre"`).
3. **Editing Requirement:** The addition of `BACKSPACE` in the current release resolves the inability to fix miss-hits during typing.

---

## 🚀 Getting Started

### Prerequisites

- **Python:** Version `3.9`, `3.10`, `3.11`, or `3.12` (64-bit).
  *(Note: MediaPipe official wheels are available up to Python 3.12).*
- A working webcam connected to your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/gaze-keyboard.git
   cd gaze-keyboard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main application:
```bash
python main.py
```

* Two windows will appear:
  1. **Gaze Tracking Feed:** Shows the webcam video, facial landmarks, gaze ray vector, and HUD telemetry.
  2. **Virtual Keyboard:** Shows the interactive keyboard with your typed text.
* To exit the application, press **`ESC`** or **`q`** in either window.

---

## ⌨️ Keyboard Layout & Controls

- **QWERTY Alphabet:** Rows for all letters `A`–`Z`.
- **`SPACE`:** Inserts a whitespace character into the buffer.
- **`BACKSPACE`:** Deletes the most recently typed character.
- **`CLEAR`:** Clears the entire text buffer.

### Interaction Flow:
1. Move your eyes to look at your desired key on the screen.
2. A **green border** and **red target indicator** will highlight the targeted key.
3. Perform a deliberate blink (holding closed for ~150–250ms).
4. The key will register, and the cooldown bar on the HUD will reset.

---

## 🔮 Known Issues & Future Work

While the core pipeline provides a functional proof-of-concept, the following areas represent critical future enhancements:

### 1. Screen Calibration & Homography Mapping
- **Issue:** Currently, the projected gaze point resides within the camera's frame coordinate system ($640 \times 480$), requiring empirical window positioning.
- **Future Work:** Implement a 5-point or 9-point calibration routine at session startup. By recording eye landmarks while the user fixates on known screen coordinates, an affine or homography transformation matrix $H$ can map gaze directly to absolute display coordinates.

### 2. Dwell-Time Selection Paradigm
- **Issue:** Relying on blinks causes two usability issues:
  - Involuntary blinks (natural eye lubrication) can trigger unwanted selections.
  - Eyelid closure causes the pupil to rapidly move upwards (Bell's phenomenon), shifting the gaze vector immediately prior to registration.
- **Future Work:** Transition to (or offer an option for) **Dwell Selection**: fixating on a key for a user-defined threshold (e.g., 600ms–900ms) with a visual radial progress indicator triggers the keypress without blinking.

### 3. Binocular Gaze Integration
- **Issue:** The current gaze estimation relies exclusively on the left eye pupil (Landmark `468`).
- **Future Work:** Incorporate both left and right pupil centers (Landmarks `468` and `473`) and compute a weighted binocular average, reducing tracking noise by ~30–40%.

### 4. Advanced Jitter Filtering (1€ Filter / Kalman Filter)
- **Issue:** Static Exponential Moving Average (EMA) filters trade responsiveness for stability: low alpha values eliminate jitter but introduce cursor lag during rapid eye movements (saccades).
- **Future Work:** Implement the **One-Euro (1€) Filter** or an adaptive Kalman filter, which dynamically scales cutoff frequencies based on movement velocity (high smoothing during fixations, low smoothing during saccades).

### 5. Predictive Text & Word Completion (n-gram / T9)
- **Issue:** Single-character typing at ~1 character per 2 seconds results in typing speeds under 10 WPM.
- **Future Work:** Integrate an autocomplete dictionary (e.g., SymSpell or a lightweight n-gram model) presenting the top 3 word completions on an extra top row, cutting required keystrokes by over 50%.

### 6. OS-Level Keystroke Injection
- **Issue:** Keystrokes are currently confined to the internal OpenCV window buffer.
- **Future Work:** Use `pyautogui` or `pynput` to simulate native OS keyboard events, enabling users to type into any active desktop application (Notepad, web browser, word processor).

### 7. Modern GUI Framework Integration
- **Issue:** OpenCV windows lack DPI scaling, accessibility theming, and smooth anti-aliased font rendering.
- **Future Work:** Migrate the user interface to a modern framework such as **PyQt6** or an **Electron / Web UI** with customizable color themes, high-contrast modes, and audio feedback on keypress.

---

## 👥 Attribution & References

- **Project Author:** Youssef Ahmed Hany (ID: 220427) — [Gaze Report.pdf](./Gaze%20Report.pdf)
- **Blink Detection Paper:** Tereza Soukupová and Jan Čech. *"Real-Time Eye Blink Detection using Facial Landmarks."* Computer Vision Winter Workshop (CVWW), 2016.
- **Original Gaze Estimation Reference:** [Eye Gaze Estimation Using a Webcam in 100 Lines of Code](https://medium.com/mlearning-ai/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23) by rancicne.
- **Facial Landmark Framework:** [Google MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and build upon this work for academic, personal, or assistive applications.
