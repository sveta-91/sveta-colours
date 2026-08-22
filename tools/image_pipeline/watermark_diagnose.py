"""Inspection helper for the watermark-removal workflow.

Reads an image and writes diagnostic crops (the four corners + a centre strip
on each edge) plus a tone summary, so an agent that's planning a removal pass
can SEE every region where watermarks usually live without dumping the whole
file into context.

Usage:
    python watermark_diagnose.py <input_image> [--out <dir>]

Writes:
    <dir>/info.txt           — image size + per-region brightness stats
    <dir>/full.png           — the full image (downscaled to max-edge 1024)
    <dir>/tl.png             — top-left corner (35% × 20%)
    <dir>/tr.png             — top-right corner
    <dir>/bl.png             — bottom-left corner (incl. typical signature spot)
    <dir>/br.png             — bottom-right corner
    <dir>/right_edge.png     — full-height right strip (vertical watermarks)
    <dir>/bottom_edge.png    — full-width bottom strip
"""
import argparse
import os
import sys
import cv2
import numpy as np


def stats(gray_patch):
    p = gray_patch.ravel()
    return dict(
        min=int(p.min()),
        p50=int(np.percentile(p, 50)),
        p95=int(np.percentile(p, 95)),
        max=int(p.max()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("--out", default="/tmp/wm_diag")
    args = ap.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"ERROR: could not read {args.input}", file=sys.stderr)
        sys.exit(2)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.makedirs(args.out, exist_ok=True)

    regions = {
        "tl":          (0,              0,              int(W * 0.35), int(H * 0.20)),
        "tr":          (int(W * 0.65),  0,              W,             int(H * 0.20)),
        "bl":          (0,              int(H * 0.80),  int(W * 0.35), H),
        "br":          (int(W * 0.55),  int(H * 0.80),  W,             H),
        "right_edge":  (int(W * 0.88),  0,              W,             H),
        "bottom_edge": (0,              int(H * 0.85),  W,             H),
    }

    # Downscaled full image — for the agent to get an at-a-glance view that
    # fits comfortably in the context window without losing watermark legibility.
    scale = min(1024.0 / W, 1024.0 / H, 1.0)
    fw, fh = int(round(W * scale)), int(round(H * scale))
    full = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.out, "full.png"), full)

    info_lines = [f"Image: {W} x {H}", ""]
    for name, (x0, y0, x1, y1) in regions.items():
        crop = img[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), crop)
        st = stats(gray[y0:y1, x0:x1])
        info_lines.append(
            f"{name:12s}  bbox=({x0},{y0})-({x1},{y1})  "
            f"size={x1-x0}x{y1-y0}  "
            f"gray min={st['min']} p50={st['p50']} p95={st['p95']} max={st['max']}"
        )
    info = "\n".join(info_lines) + "\n"
    with open(os.path.join(args.out, "info.txt"), "w") as f:
        f.write(info)
    print(info)


if __name__ == "__main__":
    main()
