#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize MediaPipe hand 3D coordinates as standalone plots.")
    ap.add_argument("--npz", required=True, help="Path to hand3d npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-world", action="store_true", help="Use world_xyz; fallback to xyz if empty.")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--show-index", action="store_true", help="Draw landmark index text")
    return ap.parse_args()


def _set_equal_3d(ax, x, y, z):
    mins = np.array([x.min(), y.min(), z.min()], dtype=np.float32)
    maxs = np.array([x.max(), y.max(), z.max()], dtype=np.float32)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    if span <= 1e-6:
        span = 1.0
    r = span / 2.0
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    frame_ids = z["frame_ids"].astype(int)
    detected = z["detected"].astype(np.uint8)
    xyz = z["world_xyz"] if args.use_world else z["xyz"]
    if args.use_world and np.allclose(xyz, 0.0):
        xyz = z["xyz"]

    T = int(xyz.shape[0])
    max_frames = T if args.max_frames <= 0 else min(T, int(args.max_frames))
    saved = 0

    for i in range(max_frames):
        if args.every > 1 and (i % args.every != 0):
            continue

        pts = xyz[i]  # (21,3)
        fid = int(frame_ids[i]) if i < len(frame_ids) else i
        det = int(detected[i]) if i < len(detected) else 0

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        if det > 0:
            x, y, zc = pts[:, 0], pts[:, 1], pts[:, 2]
            ax.scatter(x, y, zc, c="crimson", s=24, depthshade=True)
            for a, b in HAND_EDGES:
                ax.plot([x[a], x[b]], [y[a], y[b]], [zc[a], zc[b]], c="seagreen", lw=2)
            if args.show_index:
                for j in range(21):
                    ax.text(x[j], y[j], zc[j], str(j), fontsize=7, color="navy")
            _set_equal_3d(ax, x, y, zc)
            title = f"{npz_path.stem} | local={i} abs={fid} det=1"
        else:
            ax.text(0.5, 0.5, 0.5, "NO_HAND", fontsize=16, color="red")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            title = f"{npz_path.stem} | local={i} abs={fid} det=0"

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=18, azim=-65)

        out_path = out_dir / f"{npz_path.stem}_local{i:05d}_abs{fid:05d}_3d.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        saved += 1

    print(f"Saved {saved} 3D frames to {out_dir}")


if __name__ == "__main__":
    main()
