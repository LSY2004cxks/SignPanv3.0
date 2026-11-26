import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segment_video import build_pose_features, _load_boundary, load_boundary_model

video_path = str(ROOT / "data28" / "S142" / "slr-4-万岁-142-6326.mp4")

# 复用 segment_from_video_file 里的抽帧逻辑（简化版）
import cv2
import mediapipe as mp
cap = cv2.VideoCapture(video_path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
all_xyz = []
for _ in range(n):
    ok, frame = cap.read()
    if not ok: break
    res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        all_xyz.append(np.array([[lm.x,lm.y,lm.z] for lm in res.multi_hand_landmarks[0].landmark], dtype=np.float32))
    else:
        all_xyz.append(np.zeros((21,3), dtype=np.float32))
cap.release(); mp_hands.close()
xyz = np.stack(all_xyz)

# 推理
BoundaryTCN, decode = _load_boundary()
model, _ = load_boundary_model(
    str(ROOT / "boundary_prefill_B_final" / "logs" / "run_20260228_210024" / "best.pt"),
    "cpu",
)
feat = build_pose_features(xyz)
x = np.expand_dims(feat, 0)
with torch.no_grad():
    logits = model(torch.from_numpy(x)).cpu().numpy()
p = 1/(1+np.exp(-logits[0]))

print(f"T={len(p)}")
print(f"p_start: min={p[:,0].min():.4f} max={p[:,0].max():.4f} mean={p[:,0].mean():.4f}")
print(f"p_end:   min={p[:,1].min():.4f} max={p[:,1].max():.4f} mean={p[:,1].mean():.4f}")
print(f"p_inside:min={p[:,2].min():.4f} max={p[:,2].max():.4f} mean={p[:,2].mean():.4f}")
print(f"\ntau=0.5 时 >0.5 的帧数: start={int((p[:,0]>0.5).sum())} end={int((p[:,1]>0.5).sum())}")

# 保存概率供后续分析
output_path = ROOT / "output" / "debug_probs.npy"
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, p)