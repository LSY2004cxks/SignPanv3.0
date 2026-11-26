import numpy as np


def sample_indices(start_frame: int, end_frame: int, T: int, strategy: str = 'biased_start', bias: float = 0.7):
    """
    Return a list of frame indices within [start_frame, end_frame].
    - biased_start: sample more from the first bias portion of the span
    - uniform: uniform sampling
    """
    if end_frame < start_frame:
        end_frame = start_frame
    span = end_frame - start_frame + 1
    if span <= 0:
        return [start_frame] * T

    if strategy == 'uniform' or span == 1:
        idxs = np.linspace(0, span - 1, T)
        return [start_frame + int(round(x)) for x in idxs]

    # biased_start
    early_end = start_frame + int(max(1, round(span * bias))) - 1
    early_end = min(early_end, end_frame)
    early_span = early_end - start_frame + 1
    late_span = end_frame - early_end

    n_early = int(round(T * 0.7))
    n_late = T - n_early
    idxs = []
    if early_span > 0:
        idxs += list(np.linspace(start_frame, early_end, n_early, dtype=int))
    if late_span > 0 and n_late > 0:
        idxs += list(np.linspace(early_end + 1, end_frame, n_late, dtype=int))

    if len(idxs) < T:
        # pad by repeating last
        idxs += [idxs[-1]] * (T - len(idxs))
    return idxs[:T]
