"""Apply a watermark-removal spec to an image.

Workflow per operation (matches the watermark-removal skill's Method B refined
during the Noise And Silence job):
    1. TELEA inpaint the target box — kills the glyphs (their high gradient
       was beating MIXED_CLONE).
    2. NORMAL_CLONE a donor patch (same size as the target box) with tone-match
       against the ring OUTSIDE the box. Tone-match avoids the donor's mean
       brightness pulling the area off-tone.

The spec is JSON:

    {
      "preserve_zones": [
        {"name": "artist_signature", "box": [75, 730, 200, 760]}
      ],
      "operations": [
        {"name": "string label",
         "target_box":  [x0, y0, x1, y1],
         "donor_tl":    [sx, sy]}
      ]
    }

Donor patch size is derived from target_box dimensions. The script REFUSES
operations whose donor patch overlaps the target box itself or any
preserve_zone — these are the failure modes the agent must avoid in auto mode.

Usage:
    python watermark_remove.py <input> <output> --spec spec.json [--crops <dir>]

When --crops is given, dumps per-operation before/after crops so the caller
can confirm visually without reading the whole output image.
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np


def _tone(arr_bgr):
    return arr_bgr.reshape(-1, 3).astype(np.float32).mean(0)


def _tone_outside(img, x0, y0, x1, y1, pad=14):
    """Mean BGR tone of the ring AROUND a target box — never includes the
    watermark-contaminated interior."""
    H, W = img.shape[:2]
    oy0, oy1 = max(0, y0 - pad), min(H, y1 + pad)
    ox0, ox1 = max(0, x0 - pad), min(W, x1 + pad)
    outer = img[oy0:oy1, ox0:ox1].copy()
    iy0 = max(0, y0 - oy0); iy1 = iy0 + (y1 - y0)
    ix0 = max(0, x0 - ox0); ix1 = ix0 + (x1 - x0)
    mk = np.ones(outer.shape[:2], np.bool_)
    mk[iy0:iy1, ix0:ix1] = False
    samples = outer[mk]
    if samples.size == 0:
        return _tone(outer)
    return samples.reshape(-1, 3).astype(np.float32).mean(0)


def _boxes_overlap(a, b):
    """Both args are (x0, y0, x1, y1)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _inpaint_box(img, x0, y0, x1, y1, radius=8):
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def _clone_box(dst, donor_src, target_box, donor_tl, tone_ref):
    x0, y0, x1, y1 = target_box
    w, h = x1 - x0, y1 - y0
    sx, sy = donor_tl
    donor = donor_src[sy:sy + h, sx:sx + w].copy()
    cur = _tone(donor)
    donor = np.clip(donor.astype(np.float32) + (tone_ref - cur),
                    0, 255).astype(np.uint8)
    mask = np.full((h, w), 255, np.uint8)
    center = (x0 + w // 2, y0 + h // 2)
    return cv2.seamlessClone(donor, dst, mask, center, cv2.NORMAL_CLONE)


def validate_operation(op, preserve_boxes, img_shape):
    """Returns (ok, error_msg). Errors are written so the agent can fix the
    spec without re-reading source code."""
    H, W = img_shape[:2]
    tb = op.get("target_box")
    dtl = op.get("donor_tl")
    if not (isinstance(tb, list) and len(tb) == 4):
        return False, "target_box must be [x0,y0,x1,y1]"
    if not (isinstance(dtl, list) and len(dtl) == 2):
        return False, "donor_tl must be [sx,sy]"
    x0, y0, x1, y1 = tb
    w, h = x1 - x0, y1 - y0
    sx, sy = dtl
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        return False, f"target_box out of bounds for image {W}x{H}"
    if not (0 <= sx and 0 <= sy and sx + w <= W and sy + h <= H):
        return False, f"donor at ({sx},{sy})+{w}x{h} out of bounds for image {W}x{H}"
    donor_box = (sx, sy, sx + w, sy + h)
    if _boxes_overlap(donor_box, tuple(tb)):
        return False, "donor overlaps target_box (would clone the watermark onto itself)"
    for pb in preserve_boxes:
        if _boxes_overlap(donor_box, pb):
            return False, f"donor overlaps preserve_zone {pb}"
    return True, ""


def apply_spec(input_path, output_path, spec, crops_dir=None):
    img = cv2.imread(input_path)
    if img is None:
        raise SystemExit(f"could not read {input_path}")
    H, W = img.shape[:2]
    origin = img.copy()  # donor source is always the untouched original

    preserve = spec.get("preserve_zones", []) or []
    preserve_boxes = [tuple(pz["box"]) for pz in preserve]

    if crops_dir:
        os.makedirs(crops_dir, exist_ok=True)

    for op in spec.get("operations", []):
        name = op.get("name", "op")
        ok, err = validate_operation(op, preserve_boxes, img.shape)
        if not ok:
            raise SystemExit(f"[{name}] spec invalid: {err}")
        x0, y0, x1, y1 = op["target_box"]
        sx, sy = op["donor_tl"]

        # Save the before-crop so the caller can diff visually
        if crops_dir:
            cv2.imwrite(os.path.join(crops_dir, f"{name}_before.png"),
                        origin[y0:y1, x0:x1])

        ref = _tone_outside(img, x0, y0, x1, y1)
        img = _inpaint_box(img, x0, y0, x1, y1)
        img = _clone_box(img, origin, op["target_box"], (sx, sy), ref)
        print(f"[{name}] target={op['target_box']} donor={(sx, sy)} applied")

        if crops_dir:
            cv2.imwrite(os.path.join(crops_dir, f"{name}_after.png"),
                        img[y0:y1, x0:x1])

    cv2.imwrite(output_path, img,
                [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    print(f"wrote {output_path}")

    # Always emit a downscaled full-image preview when crops_dir is set, so the
    # agent can verify the global look in one Read.
    if crops_dir:
        scale = min(1024.0 / W, 1024.0 / H, 1.0)
        fw, fh = int(round(W * scale)), int(round(H * scale))
        full = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(crops_dir, "result_full.png"), full)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--crops", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.spec):
        print(f"ERROR: spec not found: {args.spec}", file=sys.stderr)
        sys.exit(2)
    with open(args.spec) as f:
        spec = json.load(f)
    apply_spec(args.input, args.output, spec, crops_dir=args.crops)


if __name__ == "__main__":
    main()
