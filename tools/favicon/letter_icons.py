#!/usr/bin/env python3
"""Letter favicon set: initials on a rounded square, in the site's own font.

Used for both sites (2026-11-15):
  sveta-colours  SM  in Marcellus           navy #1A2640 on light blue #C7D2EC
  olgazai        ZAI in Cormorant Garamond  #FBFBF8 on madder #7A2E2A, wght 600

  python3 letter_icons.py --text SM --font Marcellus-Regular.ttf \
      --bg '#C7D2EC' --fg '#1A2640' --fill-small 0.76 --fill-large 0.64 --track 0.04 \
      --out ../../icons --ico ../../favicon.ico
  python3 letter_icons.py --text ZAI --font 'CormorantGaramond[wght].ttf' --wght 600 \
      --bg '#7A2E2A' --fg '#FBFBF8' --out ../../sites/olga/icons --ico ../../sites/olga/favicon.ico

Fonts: the .ttf from github.com/google/fonts (ofl/<family>/). Needs fontTools
and Pillow. The SVG carries the glyphs as paths, so no font loads in the page.

Writes favicon.svg, favicon-16/32/48/64.png, icon-192/512.png,
apple-touch-icon.png (full square: iOS rounds it) and the .ico (16/32/48).
Small sizes (<= 32 px, and the SVG, which tabs use) get wider lettering
(--fill-small) so the initials stay readable in a browser tab.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--text', required=True)
    p.add_argument('--font', required=True, type=Path)
    p.add_argument('--wght', type=float, help='weight for a variable font (e.g. 600)')
    p.add_argument('--bg', required=True)
    p.add_argument('--fg', required=True)
    p.add_argument('--track', type=float, default=0.05, help='letter spacing, in em')
    p.add_argument('--fill-small', type=float, default=0.92, help='text width / icon width at <= 32 px')
    p.add_argument('--fill-large', type=float, default=0.80, help='text width / icon width above 32 px')
    p.add_argument('--out', required=True, type=Path, help='icons folder')
    p.add_argument('--ico', required=True, type=Path, help='favicon.ico path')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    font = TTFont(a.font)
    font_path = a.font
    if a.wght is not None:
        from fontTools.varLib import instancer
        font = instancer.instantiateVariableFont(font, {'wght': a.wght})
        font_path = a.out / '.letter-icons-font.ttf'
        font.save(font_path)
    gs, cmap, upm = font.getGlyphSet(), font.getBestCmap(), font['head'].unitsPerEm

    # Lay the glyphs out in font units; the text box is the tight ink bounds.
    x, parts = 0.0, []
    xmin = xmax = None
    ymin, ymax = 1e9, -1e9
    for ch in a.text:
        name = cmap[ord(ch)]
        bp = BoundsPen(gs); gs[name].draw(bp); b = bp.bounds
        if xmin is None:
            xmin = x + b[0]
        xmax = x + b[2]; ymin = min(ymin, b[1]); ymax = max(ymax, b[3])
        pen = SVGPathPen(gs); gs[name].draw(pen)
        parts.append((x, pen.getCommands(), ch))
        x += gs[name].width + a.track * upm
    w_fu, h_fu = xmax - xmin, ymax - ymin

    def layout(size: float, fill: float):
        sc = min(fill * size / w_fu, 0.62 * size / h_fu)
        tx = (size - w_fu * sc) / 2 - xmin * sc
        ty = (size + h_fu * sc) / 2 + ymin * sc          # baseline
        return sc, tx, ty

    sc, tx, ty = layout(32, a.fill_small)
    paths = ''.join(f'<path transform="translate({px:.1f},0)" d="{d}"/>' for px, d, _ in parts)
    (a.out / 'favicon.svg').write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">\n'
        f'  <rect width="32" height="32" rx="6" fill="{a.bg}"/>\n'
        f'  <g transform="translate({tx:.4f},{ty:.4f}) scale({sc:.6f},{-sc:.6f})" fill="{a.fg}">{paths}</g>\n'
        f'</svg>\n')

    def png(size: int, rounded: bool = True) -> Image.Image:
        k = 8; z = size * k                                # draw at 8x, then downscale
        im = Image.new('RGBA', (z, z), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
        if rounded:
            d.rounded_rectangle([0, 0, z - 1, z - 1], radius=round(z * 6 / 32), fill=a.bg)
        else:
            d.rectangle([0, 0, z, z], fill=a.bg)
        s, ox, oy = layout(z, a.fill_small if size <= 32 else a.fill_large)
        f = ImageFont.truetype(str(font_path), size=round(upm * s))
        for px, _, ch in parts:
            d.text((ox + px * s, oy), ch, font=f, fill=a.fg, anchor='ls')
        return im.resize((size, size), Image.LANCZOS)

    for n in (16, 32, 48, 64):
        png(n).save(a.out / f'favicon-{n}.png')
    for n in (192, 512):
        png(n).save(a.out / f'icon-{n}.png')
    png(180, rounded=False).convert('RGB').save(a.out / 'apple-touch-icon.png')
    png(48).save(a.ico, sizes=[(16, 16), (32, 32), (48, 48)])
    if font_path != a.font:
        font_path.unlink()
    print(f'wrote {a.out}/ (8 files) and {a.ico}')


if __name__ == '__main__':
    main()
