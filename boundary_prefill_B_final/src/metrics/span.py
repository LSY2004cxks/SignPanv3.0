from typing import Dict, List, Sequence, Tuple


def segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e < s:
        return 0.0
    inter = e - s + 1
    ua = a[1] - a[0] + 1
    ub = b[1] - b[0] + 1
    return inter / float(ua + ub - inter)


def match_spans_iou(
    gt_spans: Sequence[Tuple[int, int]],
    pred_spans: Sequence[Tuple[int, int]],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int]:
    matched = set()
    tp = 0
    fp = 0
    for p in pred_spans:
        best = -1
        best_iou = 0.0
        for i, g in enumerate(gt_spans):
            if i in matched:
                continue
            iou = segment_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best = i
        if best >= 0 and best_iou >= iou_thr:
            matched.add(best)
            tp += 1
        else:
            fp += 1
    fn = len(gt_spans) - len(matched)
    return tp, fp, fn


def counts_to_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f1": f1}
