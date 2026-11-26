import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kpts.geometry import root_scale_normalize, compute_frame_features


def test_root_scale_normalize():
    # simple synthetic: shoulders at (0,0) and (2,0)
    pose = np.zeros((2, 17, 2), dtype=np.float32)
    pose[:, 5, :] = np.array([0, 0])  # left_shoulder
    pose[:, 6, :] = np.array([2, 0])  # right_shoulder
    pose[:, 11, :] = np.array([0, -2])  # left_hip
    pose[:, 12, :] = np.array([2, -2])  # right_hip

    pose_norm, root, scale = root_scale_normalize(
        pose, None,
        joint_names=[
            'nose','left_eye','right_eye','left_ear','right_ear',
            'left_shoulder','right_shoulder','left_elbow','right_elbow',
            'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'
        ],
        left_shoulder='left_shoulder',
        right_shoulder='right_shoulder',
        left_hip='left_hip',
        right_hip='right_hip'
    )
    assert np.allclose(root[0], [1,0])
    assert np.allclose(scale[0], [2])


def test_compute_frame_features():
    T = 3
    hand = np.zeros((T,2), dtype=np.float32)
    pose = np.zeros((T,17,2), dtype=np.float32)
    feats = compute_frame_features(hand, pose, ['nose'] + ['j']*16, ['nose'])
    assert feats.shape[0] == T

if __name__ == '__main__':
    test_root_scale_normalize()
    test_compute_frame_features()
    print('OK')
