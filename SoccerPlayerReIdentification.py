#Import all the Libraries
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cdist

# Step 1: Track Players using YOLO + ByteTrack 
from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # or a smaller model if needed

for video_path in ['broadcast.mp4', 'tacticam.mp4']:
    model.track(source=video_path,
                tracker='bytetrack.yaml',
                persist=True, save=True, save_txt=True)

# Step 2: Crop Players from Frames using Tracking Labels 
def extract_crops(label_dir, video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_dict = {}
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_dict[frame_idx] = frame
        frame_idx += 1

    for label_file in Path(label_dir).glob('*.txt'):
        frame_id = int(label_file.stem)
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # skip malformed lines

            cls, track_id, x, y, w, h = map(float, parts)
            track_id = int(track_id)

            frame = frame_dict.get(frame_id)
            if frame is None:
                continue

            h_frame, w_frame = frame.shape[:2]
            x1 = int((x - w/2) * w_frame)
            y1 = int((y - h/2) * h_frame)
            x2 = int((x + w/2) * w_frame)
            y2 = int((y + h/2) * h_frame)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # skip invalid crops

            out_dir = Path(output_dir) / str(track_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f'{frame_id}.jpg'), crop)

extract_crops('runs/detect/predict/labels', 'broadcast.mp4', 'crops/broadcast')
extract_crops('runs/detect/track/labels', 'tacticam.mp4', 'crops/tacticam')

# Step 3: Extract Appearance Features using Torchreid
extractor = FeatureExtractor(model_name='osnet_x1_0', device='cuda' if torch.cuda.is_available() else 'cpu')

def get_features(crop_dir):
    features, ids = [], []
    for track_id_dir in Path(crop_dir).iterdir():
        if not track_id_dir.is_dir():
            continue
        for img_path in track_id_dir.glob('*.jpg'):
            img = Image.open(img_path).convert('RGB')
            feat = extractor(img)
            features.append(feat.cpu().numpy())
            ids.append(track_id_dir.name)
    if features:
        return np.vstack(features), np.array(ids)
    else:
        return np.empty((0, 512)), np.array([])  # handle empty crop case

broadcast_feats, broadcast_ids = get_features('crops/broadcast')
tacticam_feats, tacticam_ids = get_features('crops/tacticam')

# Step 4: Match Players Across Views using Cosine Distance
dist = cdist(broadcast_feats, tacticam_feats, metric='cosine')
matches = {}
for i, b_id in enumerate(broadcast_ids):
    if dist.shape[1] == 0:
        continue
    match_idx = np.argmin(dist[i])
    matches[b_id] = tacticam_ids[match_idx]

print("\n Player ID Mapping (Broadcast -> Tactim):")
for k, v in matches.items():
    print(f"Player {k} - Player {v}")
