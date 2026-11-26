from typing import Any, Dict, List, Optional, Tuple


COCO_BODY_23 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

COCO_WHOLEBODY_133_NAMES = (
    COCO_BODY_23
    + [f"face_{i}" for i in range(68)]
    + [f"left_hand_{i}" for i in range(21)]
    + [f"right_hand_{i}" for i in range(21)]
)


def _recursive_find_keypoint_names(obj: Any, out: List[List[str]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"keypoint_names", "keypoints_names"} and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out.append(v)
            _recursive_find_keypoint_names(v, out)
    elif isinstance(obj, list):
        for x in obj[:5]:
            _recursive_find_keypoint_names(x, out)


def extract_keypoint_names(pose_obj: Any, expected_k: Optional[int] = None) -> Optional[List[str]]:
    found: List[List[str]] = []
    _recursive_find_keypoint_names(pose_obj, found)
    if not found:
        return None
    if expected_k is None:
        return found[0]
    for names in found:
        if len(names) == expected_k:
            return names
    return found[0]


def build_name_to_index(
    pose_obj: Any,
    kpt_count: int,
) -> Tuple[Dict[str, int], str]:
    names = extract_keypoint_names(pose_obj, expected_k=kpt_count)
    if names and len(names) == kpt_count:
        return {str(n): i for i, n in enumerate(names)}, "pose_meta"

    if kpt_count == 133:
        return {n: i for i, n in enumerate(COCO_WHOLEBODY_133_NAMES)}, "fallback_coco_wholebody_133"

    # Last-resort generic fallback.
    return {f"kpt_{i}": i for i in range(kpt_count)}, "fallback_generic"


def _pick_idx(name_to_idx: Dict[str, int], keys: List[str]) -> Optional[int]:
    lower = {k.lower(): v for k, v in name_to_idx.items()}
    for k in keys:
        if k in name_to_idx:
            return name_to_idx[k]
        if k.lower() in lower:
            return lower[k.lower()]
    return None


def resolve_required_body_indices(name_to_idx: Dict[str, int]) -> Dict[str, Optional[int]]:
    return {
        "left_shoulder": _pick_idx(name_to_idx, ["left_shoulder", "l_shoulder"]),
        "right_shoulder": _pick_idx(name_to_idx, ["right_shoulder", "r_shoulder"]),
        "left_elbow": _pick_idx(name_to_idx, ["left_elbow", "l_elbow"]),
        "right_elbow": _pick_idx(name_to_idx, ["right_elbow", "r_elbow"]),
        "left_wrist": _pick_idx(name_to_idx, ["left_wrist", "l_wrist"]),
        "right_wrist": _pick_idx(name_to_idx, ["right_wrist", "r_wrist"]),
        "left_hip": _pick_idx(name_to_idx, ["left_hip", "l_hip"]),
        "right_hip": _pick_idx(name_to_idx, ["right_hip", "r_hip"]),
    }
