from __future__ import annotations

from typing import Optional, Dict

import numpy as np


class MMPoseWrapper:
    def __init__(self, config: str, checkpoint: str, device: str = 'cpu'):
        try:
            from mmpose.apis import MMPoseInferencer
        except Exception as e:
            raise ImportError("mmpose is required for pose estimation. Please install mmpose + mmcv.") from e
        if config is None or checkpoint is None:
            raise ValueError("MMpose config/checkpoint must be set in config.")
        self.inferencer = MMPoseInferencer(pose2d=config, pose2d_weights=checkpoint, device=device)

    def infer(self, image_bgr: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        # mmpose expects BGR or RGB depending on backend; inferencer handles np.ndarray
        results = self.inferencer(image_bgr, return_datasamples=True)
        # results is a generator
        res = next(results)
        data_sample = res['predictions'][0] if isinstance(res, dict) and 'predictions' in res else res

        # Try common mmpose structures
        if hasattr(data_sample, 'pred_instances'):
            inst = data_sample.pred_instances
            kpts = inst.keypoints
            scores = inst.keypoint_scores if hasattr(inst, 'keypoint_scores') else None
        else:
            # fallback
            return None

        if kpts is None or len(kpts) == 0:
            return None

        kpts = np.array(kpts[0], dtype=np.float32)  # (J,2)
        if scores is None:
            scores = np.ones((kpts.shape[0],), dtype=np.float32)
        else:
            scores = np.array(scores[0], dtype=np.float32)

        return {'pose_kpts': kpts, 'pose_score': scores}
