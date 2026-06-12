"""Render final SM favicon: Cochin Italic letters on Autumn Light gradient.

Outputs:
  favicon.svg              — vector source (Cochin glyphs as paths, no font dep)
  favicon-16.png           — tab favicon (legacy)
  favicon-32.png           — tab favicon (modern)
  favicon-48.png           — Windows desktop shortcut
  favicon-64.png           — extra
  apple-touch-icon.png     — 180×180 iOS bookmark
  icon-192.png             — PWA / Android home
  icon-512.png             — PWA splash / large preview
  favicon.ico              — multi-size legacy bundle (16+32+48)
  preview.png              — comparison sheet for final review
"""
from __future__ import annotations

from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "final"
OUT.mkdir(exist_ok=True)

INK   = (26, 38, 64, 255)
INK_HEX = "#1A2640"
AL_STOPS = [(126, 143, 194), (231, 135, 76), (124, 49, 27)]
AL_HEX = ["#7E8FC2", "#E7874C", "#7C311B"]

COCHIN_PATH = "/System/Library/Fonts/Supplemental/Cochin.ttc"
COCHIN_ITALIC_INDEX = 1  # 0=Roman, 1=Italic, 2=Bold, 3=BoldItalic typically


# ---------------- PNG renderer (Pillow + Cochin Italic) ----------------

def _antialias(draw_fn, size, scale=4):
    big = size * scale
    img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_fn(d, big)
    return img.resize((size, size), Image.LANCZOS)


def gradient_bg(s, stops, radius_frac=1/8):
    grad = Image.new('RGB', (1, s))
    n = len(stops) - 1
    for y in range(s):
        t = y / (s - 1)
        seg = min(int(t * n), n - 1)
        local = t * n - seg
        a, b = stops[seg], stops[seg + 1]
        r = int(a[0] + (b[0] - a[0]) * local)
        g = int(a[1] + (b[1] - a[1]) * local)
        bl = int(a[2] + (b[2] - a[2]) * local)
        grad.putpixel((0, y), (r, g, bl))
    grad = grad.resize((s, s))
    mask = Image.new('L', (s, s), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([(0, 0), (s, s)], radius=int(s * radius_frac), fill=255)
    out = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    out.paste(grad.convert('RGBA'), (0, 0), mask)
    return out


def render_png(size, radius_frac=1/8):
    def draw(d, s):
        bg = gradient_bg(s, AL_STOPS, radius_frac)
        d._image.paste(bg, (0, 0), bg)
        font = ImageFont.truetype(COCHIN_PATH, int(s * 0.62),
                                  index=COCHIN_ITALIC_INDEX)
        bbox = d.textbbox((0, 0), "SM", font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (s - tw) / 2 - bbox[0]
        y = (s - th) / 2 - bbox[1] - s * 0.05
        d.text((x, y), "SM", fill=INK, font=font)
    return _antialias(draw, size)


# ---------------- SVG renderer (Cochin glyphs as paths) ----------------

def build_svg():
    font = TTFont(COCHIN_PATH, fontNumber=COCHIN_ITALIC_INDEX)
    cmap = font.getBestCmap()
    hmtx = font['hmtx']
    gs = font.getGlyphSet()

    s_name = cmap[ord('S')]
    m_name = cmap[ord('M')]

    def path_d(name):
        pen = SVGPathPen(gs)
        gs[name].draw(pen)
        return pen.getCommands()

    def bounds(name):
        pen = BoundsPen(gs)
        gs[name].draw(pen)
        return pen.bounds

    s_path = path_d(s_name)
    m_path = path_d(m_name)
    s_b = bounds(s_name)
    m_b = bounds(m_name)
    s_adv = hmtx[s_name][0]

    # Combined bounds (M shifted by s_adv)
    xMin = min(s_b[0], s_adv + m_b[0])
    xMax = max(s_b[2], s_adv + m_b[2])
    yMin = min(s_b[1], m_b[1])
    yMax = max(s_b[3], m_b[3])
    w_g, h_g = xMax - xMin, yMax - yMin

    # Target: ~18.5 of 32 high, centered with -5% y nudge
    target_h = 18.5
    scale = target_h / h_g
    target_w = w_g * scale
    # Position so glyph bbox lands centered
    tx = (32 - target_w) / 2 - xMin * scale
    ty = (32 - target_h) / 2 + yMax * scale - 32 * 0.05
    # Flip Y: scale(scale, -scale); the yMax of glyph (top) goes to top of bbox

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="32" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{AL_HEX[0]}"/>
      <stop offset="50%" stop-color="{AL_HEX[1]}"/>
      <stop offset="100%" stop-color="{AL_HEX[2]}"/>
    </linearGradient>
  </defs>
  <rect width="32" height="32" rx="4" fill="url(#g)"/>
  <g transform="translate({tx:.4f},{ty:.4f}) scale({scale:.6f},{-scale:.6f})" fill="{INK_HEX}">
    <path d="{s_path}"/>
    <g transform="translate({s_adv},0)"><path d="{m_path}"/></g>
  </g>
</svg>
'''


# ---------------- Write outputs ----------------

svg_text = build_svg()
(OUT / "favicon.svg").write_text(svg_text)
print("wrote favicon.svg")

# PNGs at standard favicon sizes
sizes = {
    "favicon-16.png":         (16,  1/8),
    "favicon-32.png":         (32,  1/8),
    "favicon-48.png":         (48,  1/8),
    "favicon-64.png":         (64,  1/8),
    "apple-touch-icon.png":   (180, 1/8),
    "icon-192.png":           (192, 1/8),
    "icon-512.png":           (512, 1/8),
}
for fname, (sz, rad) in sizes.items():
    render_png(sz, rad).save(OUT / fname)
    print(f"wrote {fname} ({sz}×{sz})")

# Legacy favicon.ico with 16+32+48 (Pillow handles ICO)
ico_img = render_png(256, 1/8).convert('RGBA')
ico_img.save(OUT / "favicon.ico", format='ICO',
             sizes=[(16, 16), (32, 32), (48, 48)])
print("wrote favicon.ico (16+32+48)")

# Preview sheet — large + actual tab sizes
PAPER = (247, 245, 239, 255)
BONE = (234, 230, 222, 255)
TEXT = (32, 28, 24, 255)


def lbl(d, t, x, y, size=18, fill=TEXT, anchor="mm"):
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Georgia.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    d.text((x, y), t, fill=fill, font=font, anchor=anchor)


W, H = 1000, 540
sheet = Image.new('RGBA', (W, H), PAPER)
d = ImageDraw.Draw(sheet)
lbl(d, "Final favicon — Cochin Italic on Autumn Light gradient",
    W // 2, 40, size=22)
lbl(d, "left: 256px preview  •  right: actual tab sizes",
    W // 2, 70, size=13, fill=(100, 96, 88, 255))

# Big preview on left
big = render_png(512, 1/8).resize((320, 320), Image.LANCZOS)
box = Image.new('RGBA', (332, 332), BONE)
bd = ImageDraw.Draw(box)
bd.rectangle([(0, 0), (331, 331)], outline=(214, 207, 193, 255))
sheet.paste(box, (60, 110), box)
sheet.paste(big, (66, 116), big)

# Sizes on right with labels
x0 = 550
y0 = 130
sizes_show = [
    (180, "180 · apple-touch"),
    (64,  "64 · large tab"),
    (48,  "48 · Windows"),
    (32,  "32 · modern tab"),
    (16,  "16 · legacy tab"),
]
cur_y = y0
for sz, label in sizes_show:
    img = render_png(sz, 1/8)
    sheet.paste(img, (x0, cur_y), img)
    lbl(d, label, x0 + 200, cur_y + sz // 2, size=12,
        fill=(70, 65, 58, 255), anchor="lm")
    cur_y += sz + 12

sheet.convert('RGB').save(OUT / "preview.png", quality=95)
print("wrote preview.png")
print("\nAll outputs in:", OUT)
