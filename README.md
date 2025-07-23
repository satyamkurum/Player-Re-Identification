# Soccer Player Re-Identification (Computer Vision)

This repository contains a full pipeline for soccer player cross-camera ReID using YOLOv11 + ByteTrack + Torchreid.

## Pipeline Overview
1. Player detection and tracking using YOLOv11 + ByteTrack
2. Cropping each player from each frame
3. Feature extraction using pretrained ReID model (Torchreid)
4. Appearance-based matching between camera views

## Instructions
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place your videos and YOLO model weights in the working directory.
3. Run the python script: `SoccerPlayerReIdentification`

## Sample Folder Structure
- `TrackedPlayer/` - Contain Two Videos from different Camera Angles
- `crops/` – Contains cropped player images
- `runs/detect/track/labels/` – Label outputs from ByteTrack (Tacticam)
- `runs/detect/predict/labels/` – Label outputs from ByteTrack (Broadcast)

## About Me
  Satyam Kurum
- Data Scientist | ML Developer | NITK Surathkal Graduate 2025
- Passionate about GenAI, NLP, and creative machine learning apps

- You are free to use, modify, and distribute it with attribution.
