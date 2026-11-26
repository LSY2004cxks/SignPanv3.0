import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import torch
import torchvision

from src.data.sampling import sample_indices

VIDEO_EXTS = ('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV')


def load_frame_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def load_frame_from_dir(frames_dir, frame_idx):
    path = Path(frames_dir) / f"{frame_idx:06d}.jpg"
    if not path.exists():
        return None
    return cv2.imread(str(path))


def compute_bbox(landmarks, w, h, margin=0.2):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy))


def resolve_video_path(data_root: Path, event: dict):
    signer = str(event.get('signer_id', '')).strip()
    video_id = str(event.get('video_id', '')).strip()
    video_path = str(event.get('video_path', '')).strip()

    candidates = []
    if video_path:
        p = Path(video_path)
        candidates.append(p)
        candidates.append(data_root / p)
        if signer:
            candidates.append(data_root / signer / p.name)
    if signer and video_id:
        for ext in VIDEO_EXTS:
            candidates.append(data_root / signer / f"{video_id}{ext}")
    if video_id:
        for ext in VIDEO_EXTS:
            candidates.append(data_root / 'videos' / f"{video_id}{ext}")

    for c in candidates:
        if c.exists():
            return c

    if signer:
        signer_dir = data_root / signer
        if signer_dir.exists():
            for p in signer_dir.iterdir():
                if p.is_file() and p.suffix in VIDEO_EXTS and video_id and video_id in p.stem:
                    return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--T', type=int, default=8)
    ap.add_argument('--sampling', default='biased_start')
    ap.add_argument('--sampling_bias', type=float, default=0.7)
    ap.add_argument('--use_frames', action='store_true')
    ap.add_argument('--extract_feat', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_events = out_dir / 'events'
    out_events.mkdir(parents=True, exist_ok=True)

    events = [json.loads(l) for l in Path(args.split).read_text().splitlines() if l.strip()]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1)

    roi_encoder = None
    if args.extract_feat:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        roi_encoder = torch.nn.Sequential(*list(m.children())[:-1]).to(args.device).eval()

    for e in events:
        vid = e['video_id']
        hand = e['hand']
        signer = str(e.get('signer_id', 'UNK'))
        start = int(e['span']['start_frame'])
        end = int(e['span']['end_frame'])
        idxs = sample_indices(start, end, args.T, args.sampling, args.sampling_bias)
        resolved_video_path = resolve_video_path(data_root, e)

        roi_imgs = []
        roi_feats = []
        landmarks_seq = []
        quality = []

        for fidx in idxs:
            if args.use_frames:
                frame = load_frame_from_dir(data_root / 'frames' / vid, fidx)
            else:
                if resolved_video_path is None:
                    frame = None
                else:
                    frame = load_frame_from_video(resolved_video_path, fidx)
            if frame is None:
                quality.append('FRAME_MISSING')
                roi_imgs.append(np.zeros((224,224,3), dtype=np.uint8))
                landmarks_seq.append(np.zeros((21,4), dtype=np.float32))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                quality.append('NO_HAND')
                roi_imgs.append(np.zeros((224,224,3), dtype=np.uint8))
                landmarks_seq.append(np.zeros((21,4), dtype=np.float32))
                continue

            # select hand by handedness
            selected = None
            for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handed.classification[0].label.lower()
                if label == hand.lower():
                    selected = lms
                    break
            if selected is None:
                selected = res.multi_hand_landmarks[0]

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = compute_bbox(selected.landmark, w, h, margin=0.2)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                quality.append('ROI_TOO_SMALL')
            else:
                quality.append('OK')

            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (224,224))
            roi_imgs.append(roi)

            lms = []
            for lm in selected.landmark:
                lms.append([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
            landmarks_seq.append(np.array(lms, dtype=np.float32))

        roi_imgs = np.stack(roi_imgs, axis=0)
        landmarks_seq = np.stack(landmarks_seq, axis=0)

        if args.extract_feat:
            x = torch.from_numpy(roi_imgs.astype(np.float32)).permute(0,3,1,2) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
            x = (x - mean) / std
            x = x.to(args.device)
            with torch.no_grad():
                feat = roi_encoder(x).flatten(1).cpu().numpy().astype(np.float32)
            roi_feats = feat

        stem = f"{signer}_{vid}_{e['event_id']}_{hand}"
        out_path = out_events / f"{stem}.npz"
        np.savez(out_path,
                 roi_feat=roi_feats if args.extract_feat else None,
                 roi_img=roi_imgs if not args.extract_feat else None,
                 landmarks=landmarks_seq,
                 quality=np.array(quality))

    print('Done ->', out_events)


if __name__ == '__main__':
    main()
