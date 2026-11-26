#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.indexer import build_manifest
from src.data.split import load_signer_split


CANON_REPLACE = {
    "（": "(",
    "）": ")",
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
    "⑥": "6",
    "⑦": "7",
    "⑧": "8",
    "⑨": "9",
    "⑩": "10",
    "μ": "u",
    "µ": "u",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report dataset split stats and event-length histograms.")
    ap.add_argument("--video-root", default="/home/yuan/桌面/SignPanv2.0/data28")
    ap.add_argument("--label-root", default="/home/yuan/桌面/SignPanv2.0/data28")
    ap.add_argument("--pose-root", default="/home/yuan/桌面/SignPanv2.0/data28_pose")
    ap.add_argument("--split-file", default=str(ROOT / "splits/signer_split_data28.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "dataset_stats"))
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--bin-width", type=int, default=2, help="Fixed bin width in frames for Figure 3-1.")
    ap.add_argument(
        "--xmax",
        type=int,
        default=-1,
        help="Max x-axis for Figure 3-1. <=0 means auto min(100, ceil(P99+10)).",
    )
    ap.add_argument(
        "--y-mode",
        choices=["count", "density"],
        default="count",
        help="Histogram y axis mode.",
    )
    ap.add_argument(
        "--gloss-normalize",
        default="both",
        choices=["raw", "canonical", "both"],
        help="raw=direct string unique; canonical=normalized vocabulary unique; both=report both.",
    )
    return ap.parse_args()


def _safe_mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _canonical_gloss(s: str) -> str:
    x = str(s).strip()
    for k, v in CANON_REPLACE.items():
        x = x.replace(k, v)
    x = re.sub(r"\s+", "", x)
    x = x.replace("侧面", "")
    # Known typo in current annotations.
    x = x.replace("对讲话", "对讲机")
    # Common variant in this dataset.
    x = x.replace("围嘴儿", "围嘴二")
    # Parenthetical explanations are treated as aliases in canonical vocabulary.
    x = re.sub(r"\([^)]*\)", "", x)
    # Merge index variants like 下流2 / 硒鼓1 for vocabulary-level counting.
    x = re.sub(r"(\D)\d+$", r"\1", x)
    return x.strip().lower()


def _collect_gloss_vocab(label_root: Path) -> Dict[str, Any]:
    raw_gloss = set()
    canon_gloss = set()
    for p in sorted(label_root.rglob("*.json")):
        if "-label" not in p.parent.name:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        video = obj.get("video", {}) if isinstance(obj, dict) else {}
        g = video.get("gloss") if isinstance(video, dict) else None
        if g is not None and str(g).strip() != "":
            gs = str(g).strip()
            raw_gloss.add(gs)
            canon_gloss.add(_canonical_gloss(gs))
    return {
        "raw_count": len(raw_gloss),
        "raw_list": sorted(raw_gloss),
        "canonical_count": len(canon_gloss),
        "canonical_list": sorted(canon_gloss),
    }


def _extract_vocab_id_from_video_path(video_path: str) -> str:
    name = Path(str(video_path)).name
    m = re.search(r"-(\d+)(?:\.[A-Za-z0-9]+)?$", name)
    if m:
        return m.group(1)
    return ""


def _collect_vocab_ids(label_root: Path) -> Dict[str, Any]:
    vocab_ids = set()
    for p in sorted(label_root.rglob("*.json")):
        if "-label" not in p.parent.name:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        video = obj.get("video", {}) if isinstance(obj, dict) else {}
        vid = _extract_vocab_id_from_video_path(str(video.get("path", "")))
        if not vid:
            # fallback from label filename: S8_V6326.json
            m = re.search(r"V(\d+)", p.stem)
            vid = m.group(1) if m else ""
        if vid:
            vocab_ids.add(vid)
    ids_sorted = sorted(vocab_ids, key=lambda x: int(x))
    return {"count": len(ids_sorted), "ids": ids_sorted}


def _split_stats(records: List[Any], signers: List[str]) -> Dict[str, Any]:
    frames = [int(r.n_frames) for r in records if r.n_frames is not None]
    l_events = 0
    r_events = 0
    l_lengths: List[int] = []
    r_lengths: List[int] = []

    for rec in records:
        for ev in rec.events_by_hand.get("L", []):
            if not ev.train_valid:
                continue
            l_events += 1
            ln = int(ev.end) - int(ev.start) + 1
            if ln > 0:
                l_lengths.append(ln)
        for ev in rec.events_by_hand.get("R", []):
            if not ev.train_valid:
                continue
            r_events += 1
            ln = int(ev.end) - int(ev.start) + 1
            if ln > 0:
                r_lengths.append(ln)

    return {
        "signers": len(signers),
        "videos": len(records),
        "avg_frames": round(_safe_mean(frames), 2),
        "l_hand_events": int(l_events),
        "r_hand_events": int(r_events),
        "l_lengths": l_lengths,
        "r_lengths": r_lengths,
    }


def _build_bins(all_lengths: np.ndarray, bin_width: int, x_max: int) -> Tuple[np.ndarray, int]:
    bw = max(1, int(bin_width))
    if all_lengths.size == 0:
        xmax = max(20, int(x_max) if int(x_max) > 0 else 100)
    elif int(x_max) > 0:
        xmax = int(x_max)
    else:
        p99 = float(np.percentile(all_lengths, 99))
        xmax = int(min(100, np.ceil(p99 + 10)))
        xmax = max(xmax, int(all_lengths.max()) + 1)
    bins = np.arange(0, xmax + bw, bw, dtype=np.int32)
    if bins.size < 2:
        bins = np.asarray([0, bw], dtype=np.int32)
    return bins, xmax


def _add_quantile_lines(ax: plt.Axes, data: np.ndarray, y_top: float) -> None:
    if data.size == 0:
        return
    p50 = float(np.percentile(data, 50))
    p90 = float(np.percentile(data, 90))
    ax.axvline(p50, linestyle="--", linewidth=1.0, color="black")
    ax.axvline(p90, linestyle="-.", linewidth=1.0, color="black")
    ax.text(
        p50,
        y_top * 0.95,
        f"P50={p50:.0f}",
        ha="right",
        va="top",
        fontsize=8,
        color="black",
    )
    ax.text(
        p90,
        y_top * 0.84,
        f"P90={p90:.0f}",
        ha="right",
        va="top",
        fontsize=8,
        color="black",
    )


def _plot_hist(lengths: List[int], title: str, out_path: Path, bins: np.ndarray, x_max: int, y_mode: str, y_max: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(lengths, dtype=np.int32)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    if arr.size == 0:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_xlim(0, max(1, int(x_max)))
        ax.set_ylim(0, 1)
    else:
        ax.hist(
            arr,
            bins=bins,
            density=(y_mode == "density"),
            color="lightgray",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xlim(0, int(x_max))
        ax.set_ylim(0, max(1e-6, float(y_max)))
        _add_quantile_lines(ax, arr.astype(np.float32), y_top=max(1e-6, float(y_max)))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Length (frames)", fontsize=9)
    ax.set_ylabel("Density" if y_mode == "density" else "Count", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_hist_pair_bw(
    left_lengths: List[int],
    right_lengths: List[int],
    out_png: Path,
    out_pdf: Path,
    bin_width: int,
    x_max: int,
    y_mode: str,
) -> Tuple[np.ndarray, int, float]:
    left = np.asarray(left_lengths, dtype=np.int32)
    right = np.asarray(right_lengths, dtype=np.int32)
    all_arr = np.concatenate([left, right]) if left.size and right.size else (left if left.size else right)

    bins, xmax = _build_bins(all_arr, bin_width=bin_width, x_max=x_max)
    h_left, _ = np.histogram(left, bins=bins, density=(y_mode == "density"))
    h_right, _ = np.histogram(right, bins=bins, density=(y_mode == "density"))
    y_max = float(max(float(h_left.max()) if h_left.size else 0.0, float(h_right.max()) if h_right.size else 0.0))
    y_max = max(1e-6, y_max * 1.12)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True)
    for ax, data, title in [
        (ax1, left, "Figure 3-1a  Left-hand valid event length"),
        (ax2, right, "Figure 3-1b  Right-hand valid event length"),
    ]:
        if data.size == 0:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
        else:
            ax.hist(
                data,
                bins=bins,
                density=(y_mode == "density"),
                color="lightgray",
                edgecolor="black",
                linewidth=0.8,
            )
            _add_quantile_lines(ax, data.astype(np.float32), y_top=y_max)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Length (frames)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax1.set_ylabel("Density" if y_mode == "density" else "Count", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return bins, xmax, y_max


def main() -> None:
    args = parse_args()

    split_obj = load_signer_split(args.split_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_l: List[int] = []
    all_r: List[int] = []
    event_rows: List[Dict[str, Any]] = []

    for split_name in ["train", "val", "test"]:
        signers = [str(x) for x in split_obj.get(split_name, [])]
        manifest = build_manifest(
            video_root=args.video_root,
            label_root=args.label_root,
            pose_root=args.pose_root,
            signers=signers,
            allow_pose_missing=False,
        )
        for rec in manifest.records:
            for hand in ("L", "R"):
                hand_name = "left" if hand == "L" else "right"
                for ev in rec.events_by_hand.get(hand, []):
                    if not ev.train_valid:
                        continue
                    start = int(ev.start)
                    end = int(ev.end)
                    ln = end - start + 1
                    if ln <= 0:
                        continue
                    event_rows.append(
                        {
                            "split": split_name,
                            "signer": rec.signer_id,
                            "video_id": rec.video_id,
                            "clip_id": rec.clip_id,
                            "hand": hand_name,
                            "valid": 1,
                            "start": start,
                            "end": end,
                            "length": ln,
                        }
                    )
        st = _split_stats(manifest.records, signers)
        all_l.extend(st.pop("l_lengths"))
        all_r.extend(st.pop("r_lengths"))
        rows.append({"split": split_name, **st})

    # 28 signer sanity from data root
    all_signers = [
        p.name
        for p in Path(args.video_root).iterdir()
        if p.is_dir() and re.fullmatch(r"S\d+", p.name)
    ]
    all_signers = sorted(all_signers)

    gloss_info = _collect_gloss_vocab(Path(args.label_root))
    vocab_id_info = _collect_vocab_ids(Path(args.label_root))

    report = {
        "split_file": args.split_file,
        "video_root": args.video_root,
        "label_root": args.label_root,
        "pose_root": args.pose_root,
        "num_signer_dirs": len(all_signers),
        "signer_dirs": all_signers,
        "gloss_vocab_raw_size": int(gloss_info["raw_count"]),
        "gloss_vocab_canonical_size": int(gloss_info["canonical_count"]),
        "vocab_id_size": int(vocab_id_info["count"]),
        "splits": rows,
        "event_length": {
            "left": {
                "count": len(all_l),
                "mean": float(np.mean(all_l)) if all_l else 0.0,
                "p50": float(np.percentile(all_l, 50)) if all_l else 0.0,
                "p90": float(np.percentile(all_l, 90)) if all_l else 0.0,
            },
            "right": {
                "count": len(all_r),
                "mean": float(np.mean(all_r)) if all_r else 0.0,
                "p50": float(np.percentile(all_r, 50)) if all_r else 0.0,
                "p90": float(np.percentile(all_r, 90)) if all_r else 0.0,
            },
        },
    }

    (out_dir / "dataset_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.gloss_normalize in {"raw", "both"}:
        (out_dir / "gloss_vocab_raw.txt").write_text("\n".join(gloss_info["raw_list"]), encoding="utf-8")
    if args.gloss_normalize in {"canonical", "both"}:
        (out_dir / "gloss_vocab_canonical.txt").write_text(
            "\n".join(gloss_info["canonical_list"]), encoding="utf-8"
        )
    (out_dir / "vocab_ids.txt").write_text("\n".join(vocab_id_info["ids"]), encoding="utf-8")

    event_csv = out_dir / "event_table_effective.csv"
    with event_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "signer", "video_id", "clip_id", "hand", "valid", "start", "end", "length"],
        )
        writer.writeheader()
        writer.writerows(event_rows)

    bins, x_max, y_max = _plot_hist_pair_bw(
        all_l,
        all_r,
        out_png=out_dir / "figure_3_1_event_length_hist_LR.png",
        out_pdf=out_dir / "figure_3_1_event_length_hist_LR.pdf",
        bin_width=int(args.bin_width),
        x_max=int(args.xmax),
        y_mode=str(args.y_mode),
    )

    _plot_hist(
        all_l,
        "Figure 3-1a: Left-hand effective event length histogram",
        out_dir / "figure_3_1a_left_event_length_hist.png",
        bins=bins,
        x_max=x_max,
        y_mode=str(args.y_mode),
        y_max=y_max,
    )
    _plot_hist(
        all_r,
        "Figure 3-1b: Right-hand effective event length histogram",
        out_dir / "figure_3_1b_right_event_length_hist.png",
        bins=bins,
        x_max=x_max,
        y_mode=str(args.y_mode),
        y_max=y_max,
    )

    print("Split\tSigners\tVideos\tAvg frames\tL-hand events\tR-hand events")
    for r in rows:
        print(
            f"{r['split'].capitalize()}\t{r['signers']}\t{r['videos']}\t"
            f"{r['avg_frames']:.2f}\t{r['l_hand_events']}\t{r['r_hand_events']}"
        )
    print(f"num_signer_dirs={len(all_signers)}")
    if args.gloss_normalize in {"raw", "both"}:
        print(f"gloss_vocab_raw_size={gloss_info['raw_count']}")
    if args.gloss_normalize in {"canonical", "both"}:
        print(f"gloss_vocab_canonical_size={gloss_info['canonical_count']}")
    print(f"vocab_id_size={vocab_id_info['count']}")
    print(f"report_json={out_dir / 'dataset_stats.json'}")
    print(f"event_table_csv={event_csv}")
    print(f"fig_pair_png={out_dir / 'figure_3_1_event_length_hist_LR.png'}")
    print(f"fig_pair_pdf={out_dir / 'figure_3_1_event_length_hist_LR.pdf'}")
    print(f"fig_left={out_dir / 'figure_3_1a_left_event_length_hist.png'}")
    print(f"fig_right={out_dir / 'figure_3_1b_right_event_length_hist.png'}")


if __name__ == "__main__":
    main()
