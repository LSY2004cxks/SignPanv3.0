from typing import Dict, List, Sequence, Tuple

import numpy as np


def nms_1d(indices: Sequence[int], scores: np.ndarray, window: int) -> List[int]:
    if window <= 0:
        return sorted(set(int(i) for i in indices))
    order = sorted(set(int(i) for i in indices), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    suppressed = np.zeros((scores.shape[0],), dtype=bool)
    for i in order:
        if i < 0 or i >= scores.shape[0]:
            continue
        if suppressed[i]:
            continue
        keep.append(i)
        lo = max(0, i - window)
        hi = min(scores.shape[0], i + window + 1)
        suppressed[lo:hi] = True
    return sorted(keep)


def peak_picking(
    probs: np.ndarray,
    tau: float = 0.5,
    peak_radius: int = 7,
    nms_window: int = 7,
) -> List[Dict[str, float]]:
    p = np.asarray(probs, dtype=np.float32)
    T = p.shape[0]
    peak_radius = max(int(peak_radius), 0)
    cand = []
    for t in range(T):
        v = float(p[t])
        if v < tau:
            continue
        lo = max(0, t - peak_radius)
        hi = min(T, t + peak_radius + 1)
        if v >= float(np.max(p[lo:hi])):
            cand.append(t)

    keep = nms_1d(cand, p, nms_window)
    return [{"t": int(t), "score": float(p[t])} for t in keep]


def pair_spans(
    start_peaks: List[Dict[str, float]],
    end_peaks: List[Dict[str, float]],
    p_inside: np.ndarray,
    Lmin: int = 6,
    Lmax: int = 80,
    inside_recommend_thr: float = 0.5,
    pair_mode: str = "score_then_nearest",
) -> Tuple[List[Dict], List[Dict]]:
    starts = sorted(start_peaks, key=lambda x: int(x["t"]))
    ends = sorted(end_peaks, key=lambda x: int(x["t"]))

    used_end = set()
    spans: List[Dict] = []

    mode = str(pair_mode).lower()
    if mode not in {"nearest", "score", "score_then_nearest"}:
        mode = "score_then_nearest"

    for s in starts:
        ts = int(s["t"])
        ss = float(s["score"])
        candidates = []
        for j, e in enumerate(ends):
            if j in used_end:
                continue
            te = int(e["t"])
            length = te - ts + 1
            if length < int(Lmin):
                continue
            if length > int(Lmax):
                break
            candidates.append((j, e))
            if mode == "nearest":
                break
        chosen = None
        if candidates:
            if mode == "nearest":
                chosen = candidates[0]
            elif mode == "score":
                # ss is constant for a given start, so maximize end score is equivalent
                # to maximizing sqrt(ss * es).
                chosen = max(
                    candidates,
                    key=lambda x: float(x[1]["score"]),
                )
            else:  # score_then_nearest
                chosen = max(
                    candidates,
                    key=lambda x: (float(x[1]["score"]), -abs(int(x[1]["t"]) - ts)),
                )
        if chosen is None:
            continue

        j, e = chosen
        used_end.add(j)
        te = int(e["t"])
        es = float(e["score"])
        seg = p_inside[ts : te + 1]
        inside_mean = float(np.mean(seg)) if seg.size else 0.0
        span_score = float(np.sqrt(max(ss, 0.0) * max(es, 0.0)) * inside_mean)
        spans.append(
            {
                "start": ts,
                "end": te,
                "length": int(te - ts + 1),
                "start_score": ss,
                "end_score": es,
                "inside_mean": inside_mean,
                "score": span_score,
            }
        )

    spans_sorted = sorted(spans, key=lambda x: float(x["score"]), reverse=True)
    for i, s in enumerate(spans_sorted, start=1):
        s["rank"] = i

    rec = [
        s
        for s in spans_sorted
        if float(s["inside_mean"]) > float(inside_recommend_thr)
        and int(Lmin) <= int(s["length"]) <= int(Lmax)
    ]
    return spans_sorted, rec


def decode_boundaries(
    p_start: np.ndarray,
    p_end: np.ndarray,
    p_inside: np.ndarray,
    tau: float,
    peak_radius: int,
    nms_window: int,
    Lmin: int,
    Lmax: int,
    inside_recommend_thr: float,
    pair_mode: str = "score_then_nearest",
) -> Dict:
    s_peaks = peak_picking(
        p_start, tau=tau, peak_radius=peak_radius, nms_window=nms_window
    )
    e_peaks = peak_picking(
        p_end, tau=tau, peak_radius=peak_radius, nms_window=nms_window
    )
    spans, rec = pair_spans(
        s_peaks,
        e_peaks,
        p_inside,
        Lmin=Lmin,
        Lmax=Lmax,
        inside_recommend_thr=inside_recommend_thr,
        pair_mode=pair_mode,
    )
    return {
        "start_peaks": s_peaks,
        "end_peaks": e_peaks,
        "spans_proposed": spans,
        "spans_recommended": rec,
    }
