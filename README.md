# Soccer Player Re-Identification (Cross-Camera)

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
2. Place your videos and YOLO model weights (https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view) in the working directory.
3. Run the python script: `SoccerPlayerReIdentification`

## Sample Folder Structure
- `TrackedPlayer/` - Contain Two Videos from different Camera Angles
- `crops/` – Contains cropped player images
- `runs/detect/track/labels/` – Label outputs from ByteTrack (Tacticam)
- `runs/detect/predict/labels/` – Label outputs from ByteTrack (Broadcast)
