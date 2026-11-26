from typing import Dict, List, Tuple

import numpy as np

from .schema_parser import ParsedEvent


def gaussian_soft_labels(T: int, centers: List[int], sigma: float, radius: int) -> np.ndarray:
    y = np.zeros((T,), dtype=np.float32)
    if T <= 0:
        return y
    sigma = max(float(sigma), 1e-6)
    radius = max(int(radius), 0)
    for c in centers:
        if c < 0 or c >= T:
            continue
        lo = max(0, c - radius)
        hi = min(T - 1, c + radius)
        ts = np.arange(lo, hi + 1, dtype=np.float32)
        vals = np.exp(-((ts - float(c)) ** 2) / (2.0 * sigma * sigma))
        y[lo : hi + 1] = np.maximum(y[lo : hi + 1], vals)
    return y


def check_event_quality(events: List[ParsedEvent], T: int) -> List[str]:
    issues: List[str] = []
    spans = [(ev.start, ev.end) for ev in events]
    for i, (s, e) in enumerate(spans):
        if e < s:
            issues.append(f"negative_length:event_idx={i}:start={s}:end={e}")
        if s < 0 or e >= T:
            issues.append(f"out_of_range:event_idx={i}:start={s}:end={e}:T={T}")
    for i in range(1, len(spans)):
        ps, pe = spans[i - 1]
        s, e = spans[i]
        if s < ps:
            issues.append(f"unsorted:event_idx={i}:prev_start={ps}:cur_start={s}")
        if s <= pe:
            issues.append(f"overlap:event_idx={i}:prev_end={pe}:cur_start={s}")
    return issues


def build_targets(
    events: List[ParsedEvent],
    T: int,
    sigma: float,
    radius: int,
    use_valid_only: bool = True,
    allow_invalid_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    all_spans = []
    valid_spans = []
    for ev in events:
        all_spans.append((int(ev.start), int(ev.end)))
        if (not use_valid_only) or ev.train_valid:
            valid_spans.append((int(ev.start), int(ev.end)))

    # Optional fallback for noisy legacy labels. Keep disabled by default so
    # hands with no valid events remain pure negatives (inside all-0).
    if use_valid_only and bool(allow_invalid_fallback) and (not valid_spans) and all_spans:
        valid_spans = list(all_spans)

    starts = [s for s, _ in valid_spans]
    ends = [e for _, e in valid_spans]

    y_start = gaussian_soft_labels(T, starts, sigma=sigma, radius=radius)
    y_end = gaussian_soft_labels(T, ends, sigma=sigma, radius=radius)
    y_inside = np.zeros((T,), dtype=np.float32)
    for s, e in valid_spans:
        ss = max(0, min(T - 1, s))
        ee = max(0, min(T - 1, e))
        if ee >= ss:
            y_inside[ss : ee + 1] = 1.0

    gt_start = np.array([s for s in starts if 0 <= s < T], dtype=np.int64)
    gt_end = np.array([e for e in ends if 0 <= e < T], dtype=np.int64)

    return {
        "y_start": y_start,
        "y_end": y_end,
        "y_inside": y_inside,
        "gt_start": gt_start,
        "gt_end": gt_end,
        "gt_spans": valid_spans,
        "all_spans": all_spans,
        "quality_issues": check_event_quality(events, T),
    }
