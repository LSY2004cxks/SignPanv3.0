from typing import Any, Dict


def build_prefill_json(
    video_id: str,
    signer_id: str,
    clip_id: str,
    schema_type: str,
    left_decoded: Dict[str, Any],
    right_decoded: Dict[str, Any],
    tau: float,
    peak_radius: int,
    w_nms: int,
    Lmin: int,
    Lmax: int,
) -> Dict[str, Any]:
    return {
        "video_id": video_id,
        "signer_id": signer_id,
        "clip_id": clip_id,
        "schema_type": schema_type,
        "meta": {
            "tau": float(tau),
            "peak_radius": int(peak_radius),
            "w_nms": int(w_nms),
            "Lmin": int(Lmin),
            "Lmax": int(Lmax),
        },
        "left": left_decoded,
        "right": right_decoded,
    }
