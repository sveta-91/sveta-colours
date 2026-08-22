#!/usr/bin/env python3
"""Trim uniform light/white borders from an image.

For each edge (top/bottom/left/right) walks inward as long as the row/col
is "near-uniform-bright" — defined by per-row min brightness above a
threshold AND low row variance. Stops at the first row that fails either
test. No interpolation, no resampling — pure pixel slicing.

Usage: trim_borders.py <input> <output> [--min-bright 240] [--max-std 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def find_trim(gray: np.ndarray, axis: int, reverse: bool,
              min_bright: int, max_std: float) -> int:
    """Walk along `axis` from the start (or end if reverse), counting
    how many rows/cols are near-uniform-bright."""
    n = gray.shape[axis]
    indices = range(n - 1, -1, -1) if reverse else range(n)
    cut = 0
    for i in indices:
        line = gray[i, :] if axis == 0 else gray[:, i]
        if line.min() >= min_bright and line.std() <= max_std:
            cut += 1
        else:
            break
    return cut


def trim_borders(img: np.ndarray, min_bright: int, max_std: float):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top    = find_trim(gray, axis=0, reverse=False, min_bright=min_bright, max_std=max_std)
    bottom = find_trim(gray, axis=0, reverse=True,  min_bright=min_bright, max_std=max_std)
    left   = find_trim(gray, axis=1, reverse=False, min_bright=min_bright, max_std=max_std)
    right  = find_trim(gray, axis=1, reverse=True,  min_bright=min_bright, max_std=max_std)
    h, w = gray.shape
    y0, y1 = top, h - bottom
    x0, x1 = left, w - right
    if y1 <= y0 or x1 <= x0:
        return img, dict(top=0, bottom=0, left=0, right=0)
    return img[y0:y1, x0:x1], dict(top=top, bottom=bottom, left=left, right=right)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--min-bright", type=int, default=240,
                   help="Min pixel value to count as 'bright' (0-255). Default 240.")
    p.add_argument("--max-std", type=float, default=8.0,
                   help="Max per-line std-dev to count as 'uniform'. Default 8.")
    args = p.parse_args()

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read: {args.input}")
    h0, w0 = img.shape[:2]

    cropped, trims = trim_borders(img, args.min_bright, args.max_std)
    h1, w1 = cropped.shape[:2]

    ext = args.output.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(args.output), cropped,
                    [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    else:
        cv2.imwrite(str(args.output), cropped)

    print(f"input:  {w0}x{h0}  {args.input}")
    print(f"trim:   top={trims['top']} bottom={trims['bottom']} "
          f"left={trims['left']} right={trims['right']}")
    print(f"output: {w1}x{h1}  {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
