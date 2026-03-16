# Clean Gesture-Controlled Shapes (Resizable Edition)

A simple **computer vision demo** that allows users to control and resize 3D shapes using **hand gestures detected via webcam**.

The system uses **MediaPipe Hand Tracking** and **OpenCV** to recognize gestures in real-time and trigger shape rendering actions.

---

# Overview

This project demonstrates how **gesture recognition can be used as a natural user interface**.

Using only a webcam, the program detects hand gestures and maps them to different shapes and interactions.

Users can:

- Spawn shapes using gestures
- Resize shapes using pinch gestures
- Control the interface without any keyboard or mouse interaction

---

# Installation

Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

# Running the Program

Run the script:

```bash
python gesture_shapes.py
```

The webcam will activate and begin detecting gestures in real-time.

---

# ✋ Supported Gestures

| Gesture | Action |
|------|------|
| ✋ Open Hand (5 fingers) | Create a **Sphere** |
| ✌️ Peace Sign (2 fingers) | Create a **Cube** |
| 👆 Pointing (1 finger) | Create a **Pyramid** |
| 🤏 Pinch Gesture | **Resize current shape** (closer = smaller, farther = larger) |

---

# Controls

| Key | Action |
|---|---|
| **Q** | Quit application |
| **ESC** | Quit application |
| **H** | Toggle help text |

---

# Technologies Used

- **Python**
- **OpenCV** – Video processing and rendering
- **MediaPipe Hands** – Real-time hand tracking
- **NumPy** – Mathematical operations

---

# Features

- Real-time gesture recognition
- Dynamic shape creation
- Pinch-based resizing
- Webcam-based interaction
- Minimal and clean interface

---

# Future Improvements

- Rotation gestures
- Multi-hand interaction
- More 3D shapes
- Gesture-based color control
- Integration with 3D engines
