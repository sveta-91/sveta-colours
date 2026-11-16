#!/usr/bin/env python3
"""Make a synthetic "raw phone photo" of a framed 16x20 painting, shot at an angle.

The painting has a red square at its top-left and a green one at its bottom-right,
a white grid and a yellow circle; it sits in a dark brown 70px frame, warped by a
known perspective onto a grey wall. The exact corners of the painted surface are
returned, so the admin page's Straighten + crop (and the AI corner finder) can be
measured against the truth.

  make_skewed.py OUT.jpeg            write the photo, print {"size", "painted_corners"} as JSON
  make_skewed.py OUT.jpeg --json T   also write that JSON to T

Needs Pillow + numpy (the image-pipeline venv has both). Used by scripts/verify_admin.py.
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

FRAME = 70
PHOTO = (2600, 3200)
FRAME_CORNERS = [(380, 330), (2180, 470), (2120, 2900), (340, 2760)]  # outer frame TL TR BR BL


def _coeffs(dst, src):
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(A, float), np.array(b, float)).tolist()


def make(out_path):
    pw, ph = 1600, 2000                       # 4:5, a 16x20 canvas
    art = Image.new('RGB', (pw, ph))
    d = ImageDraw.Draw(art)
    for y in range(ph):
        t = y / ph
        d.line([(0, y), (pw, y)], fill=(int(90 + 120 * t), int(130 + 60 * t), int(190 - 110 * t)))
    for i in range(0, pw, 200): d.line([(i, 0), (i, ph)], fill=(250, 250, 250), width=6)
    for j in range(0, ph, 200): d.line([(0, j), (pw, j)], fill=(250, 250, 250), width=6)
    d.rectangle([0, 0, 120, 120], fill=(220, 30, 30))
    d.rectangle([pw - 120, ph - 120, pw, ph], fill=(30, 160, 60))
    d.ellipse([500, 700, 1100, 1300], fill=(240, 200, 60))
    framed = Image.new('RGB', (pw + 2 * FRAME, ph + 2 * FRAME), (70, 45, 25))
    framed.paste(art, (FRAME, FRAME))
    fw, fh = framed.size
    c = _coeffs(FRAME_CORNERS, [(0, 0), (fw, 0), (fw, fh), (0, fh)])
    warped = framed.transform(PHOTO, Image.PERSPECTIVE, c, Image.BICUBIC, fillcolor=(0, 0, 0))
    mask = Image.new('L', (fw, fh), 255).transform(PHOTO, Image.PERSPECTIVE, c, Image.BILINEAR, fillcolor=0)
    wall = Image.new('RGB', PHOTO, (205, 200, 190))
    wall.paste(warped, (0, 0), mask)
    wall.save(out_path, quality=92)
    inv = _coeffs([(0, 0), (fw, 0), (fw, fh), (0, fh)], FRAME_CORNERS)

    def fwd(x, y):
        den = inv[6] * x + inv[7] * y + 1
        return [round((inv[0] * x + inv[1] * y + inv[2]) / den, 1), round((inv[3] * x + inv[4] * y + inv[5]) / den, 1)]
    corners = [fwd(FRAME, FRAME), fwd(fw - FRAME, FRAME), fwd(fw - FRAME, fh - FRAME), fwd(FRAME, fh - FRAME)]
    return {"size": list(PHOTO), "painted_corners": corners}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--json')
    a = ap.parse_args()
    truth = make(a.out)
    print(json.dumps(truth))
    if a.json:
        json.dump(truth, open(a.json, 'w'))
