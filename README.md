# Leaf Image Acquisition System

A modular computer vision system for automatic leaf localization, tracking, quality assessment, and image acquisition.

Developed as part of research in Controlled Environment Agriculture (CEA) and plant phenotyping workflows, the system automatically identifies a leaf, evaluates image quality, adjusts framing, and captures leaf images with minimal user intervention.

---

## Features

### Leaf Localization
- Excess Green (ExG) vegetation segmentation
- HSV-based color filtering
- Contour analysis and scoring
- Multi-stage candidate selection
- False-positive rejection mechanisms

### Tracking & Target Locking
- Stable leaf tracking
- Target persistence and hysteresis
- Bounding box smoothing
- Automatic target reacquisition

### Auto Zoom
- Automatic leaf sizing
- Framing adjustment
- Zoom convergence logic
- Target-centered acquisition workflow

### Quality Control
- Brightness evaluation
- Sharpness estimation
- Vein visibility scoring
- Stability assessment
- Capture readiness evaluation

### Exposure Management
- Adaptive exposure control
- Brightness normalization
- Gain management
- Camera parameter monitoring

### Image Capture
- Automated capture decision pipeline
- Cooldown management
- Multi-condition validation
- Optional background refinement

---

## Project Structure

```text
leaf_capture/
│
├── main.py
├── config.py
│
├── localization/
│   └── contour_localizer.py
│
├── quality/
│   └── quality_control.py
│
├── tracking/
│   ├── leaf_tracker.py
│   └── auto_zoom.py
│
├── capture/
│   ├── capture_decision.py
│   └── save.py
│
├── camera/
│   ├── exposure_control.py
│   └── camera_io.py
│
└── ui/
    ├── overlay.py
    └── controls_panel.py
```

---

## System Workflow

```text
Camera Feed
      │
      ▼
Leaf Localization
      │
      ▼
Target Tracking
      │
      ▼
Auto Zoom
      │
      ▼
Quality Assessment
      │
      ▼
Capture Decision
      │
      ▼
Image Saving
```

---

## Detection Pipeline

```text
Input Frame
    │
    ▼
Gray World White Balance
    │
    ▼
Excess Green (ExG)
    │
    ▼
HSV Vegetation Filtering
    │
    ▼
Thresholding & Morphology
    │
    ▼
Contour Extraction
    │
    ▼
Candidate Scoring
    │
    ▼
Leaf Selection
```

---

## Quality Evaluation

The system evaluates multiple quality metrics before image acquisition:

- Exposure quality
- Sharpness
- Vein visibility
- Leaf size
- Stability
- Framing quality

An image is captured only when all required acquisition conditions are satisfied.

---



## Installation

Clone the repository:

```bash
git clone https://github.com/japmanyakaur/Leaf-Image-Acquisition-System
cd Leaf-Image-Acquisition-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the System

```bash
python leaf_capture/main.py
```

---

## Controls

| Key | Action |
|------|----------|
| q | Quit |
| m | Toggle debug mode |
| s | Save current frame |
| r | Reset tracking |


## Current Capabilities

- Automatic leaf localization
- Real-time tracking
- Exposure management
- Quality-based image acquisition
- Modular architecture for future expansion

---

## Future Improvements

- Enhanced lighting robustness
- Improved acquisition speed

---

## Author

**Japmanya Kaur**

B.Tech Computer Science & Engineering  

Research Internship – IIT Mandi

---

## License

This project is intended for academic and research purposes.
