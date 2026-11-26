from typing import Dict, Iterable, List, Sequence, Tuple


def _counts_to_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f1": f1}


def match_points_one_to_one(gt: Sequence[int], pred: Sequence[int], window: int) -> Tuple[int, int, int]:
    g = sorted(int(x) for x in gt)
    p = sorted(int(x) for x in pred)
    used = set()
    tp = 0
    fp = 0
    for x in p:
        cands = [i for i, y in enumerate(g) if i not in used and abs(y - x) <= window]
        if not cands:
            fp += 1
            continue
        best_i = min(cands, key=lambda i: abs(g[i] - x))
        used.add(best_i)
        tp += 1
    fn = len(g) - len(used)
    return tp, fp, fn


def boundary_metrics_for_windows(
    gt_start: Sequence[int],
    gt_end: Sequence[int],
    pred_start: Sequence[int],
    pred_end: Sequence[int],
    windows: Iterable[int],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for w in windows:
        st_tp, st_fp, st_fn = match_points_one_to_one(gt_start, pred_start, int(w))
        en_tp, en_fp, en_fn = match_points_one_to_one(gt_end, pred_end, int(w))
        st = _counts_to_prf(st_tp, st_fp, st_fn)
        en = _counts_to_prf(en_tp, en_fp, en_fn)
        out[int(w)] = {
            "start_p": st["p"],
            "start_r": st["r"],
            "start_f1": st["f1"],
            "end_p": en["p"],
            "end_r": en["r"],
            "end_f1": en["f1"],
            "mean_f1": 0.5 * (st["f1"] + en["f1"]),
        }
    return out


def aggregate_boundary_counts(
    gt_start: Sequence[int],
    gt_end: Sequence[int],
    pred_start: Sequence[int],
    pred_end: Sequence[int],
    windows: Iterable[int],
    acc: Dict[int, Dict[str, int]],
) -> None:
    for w in windows:
        st_tp, st_fp, st_fn = match_points_one_to_one(gt_start, pred_start, int(w))
        en_tp, en_fp, en_fn = match_points_one_to_one(gt_end, pred_end, int(w))
        if int(w) not in acc:
            acc[int(w)] = {
                "start_tp": 0,
                "start_fp": 0,
                "start_fn": 0,
                "end_tp": 0,
                "end_fp": 0,
                "end_fn": 0,
            }
        acc[int(w)]["start_tp"] += st_tp
        acc[int(w)]["start_fp"] += st_fp
        acc[int(w)]["start_fn"] += st_fn
        acc[int(w)]["end_tp"] += en_tp
        acc[int(w)]["end_fp"] += en_fp
        acc[int(w)]["end_fn"] += en_fn


def finalize_aggregate_boundary(acc: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for w, c in acc.items():
        st = _counts_to_prf(c["start_tp"], c["start_fp"], c["start_fn"])
        en = _counts_to_prf(c["end_tp"], c["end_fp"], c["end_fn"])
        out[int(w)] = {
            "start_p": st["p"],
            "start_r": st["r"],
            "start_f1": st["f1"],
            "end_p": en["p"],
            "end_r": en["r"],
            "end_f1": en["f1"],
            "mean_f1": 0.5 * (st["f1"] + en["f1"]),
        }
    return out
