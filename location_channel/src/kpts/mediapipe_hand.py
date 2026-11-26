from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np


class MediaPipeHandDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 hand_center: str = 'wrist'):
        try:
            import mediapipe as mp
        except Exception as e:
            raise ImportError("mediapipe is required for hand detection. Install mediapipe.") from e
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_center = hand_center

    def _hand_point(self, kpts: np.ndarray) -> np.ndarray:
        if self.hand_center == 'mean':
            return kpts.mean(axis=0)
        # wrist
        return kpts[0]

    def infer(self, image_bgr: np.ndarray, expected_hand: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        import cv2

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None
        h, w = image_bgr.shape[:2]

        hands = []
        for idx, hand_lms in enumerate(res.multi_hand_landmarks):
            kpts = np.array([[lm.x * w, lm.y * h] for lm in hand_lms.landmark], dtype=np.float32)
            # handedness
            hand_type = None
            score = 0.0
            if res.multi_handedness and idx < len(res.multi_handedness):
                hd = res.multi_handedness[idx].classification[0]
                hand_type = hd.label  # 'Left' or 'Right'
                score = float(hd.score)
            hands.append({'kpts': kpts, 'handedness': hand_type, 'score': score})

        # expected_hand: 'L' or 'R' or None
        if expected_hand in ('L', 'R'):
            target = 'Left' if expected_hand == 'L' else 'Right'
            candidates = [h for h in hands if h['handedness'] == target]
            if not candidates:
                candidates = hands
        else:
            candidates = hands

        # pick highest score (fallback bbox area)
        best = None
        best_score = -1.0
        for h in candidates:
            s = h['score']
            if s > best_score:
                best = h
                best_score = s
        if best is None:
            return None

        hand_point = self._hand_point(best['kpts'])
        return {
            'hand_point': hand_point,
            'hand_score': np.array(best_score, dtype=np.float32),
            'hand_kpts': best['kpts'],
            'handedness': best['handedness']
        }
