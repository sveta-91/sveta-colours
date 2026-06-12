"""Render SM monogram variants with gradients sampled from actual paintings.

For each painting:
  1. K-means cluster pixels into 6 dominant colors
  2. Sort by hue/brightness to find a meaningful gradient direction
  3. Pick 3 stops that read as a coherent warm-to-warm-to-deep gradient
  4. Letters in dark ink-blue (not white), serif italic
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent

INK      = (26, 38, 64, 255)        # site accent for letters
NEAR_BLK = (32, 28, 24, 255)        # darker alternative
TEXT_DK  = (60, 38, 24, 255)        # warm-dark brown
PAPER    = (247, 245, 239, 255)
BONE     = (234, 230, 222, 255)
LABEL_TX = (32, 28, 24, 255)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]


def _font(s: int, italic: bool = True):
    paths = FONT_PATHS if italic else FONT_PATHS[1:]
    for p in paths:
        try:
            return ImageFont.truetype(p, int(s * 0.55))
        except OSError:
            continue
    return ImageFont.load_default()


def _antialias(draw_fn, size: int, scale: int = 4):
    big = size * scale
    img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_fn(d, big)
    return img.resize((size, size), Image.LANCZOS)


def extract_palette(image_path: Path, k: int = 6):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Downsample for speed
    h, w = img.shape[:2]
    scale = 200 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    pixels = img.reshape(-1, 3).astype(np.float32)
    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    palette = [(tuple(int(c) for c in centers[i]), counts[i] / counts.sum())
               for i in order]
    return palette  # list of ((r,g,b), weight) sorted by frequency


def make_gradient_bg(s: int, stops: list[tuple[int, int, int]]) -> Image.Image:
    """Multi-stop vertical gradient, masked to rounded rect."""
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
    md.rounded_rectangle([(0, 0), (s, s)], radius=s // 8, fill=255)
    out = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    out.paste(grad.convert('RGBA'), (0, 0), mask)
    return out


def variant_factory(stops, letter_color, slug):
    def fn(size):
        def draw(d, s):
            bg = make_gradient_bg(s, stops)
            d._image.paste(bg, (0, 0), bg)
            font = _font(s)
            bbox = d.textbbox((0, 0), "SM", font=font)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            x = (s - tw) / 2 - bbox[0]
            y = (s - th) / 2 - bbox[1] - s * 0.03
            d.text((x, y), "SM", fill=letter_color, font=font)
        return _antialias(draw, size)
    fn.__name__ = slug
    return fn


# Extract palettes
print("Autumn Light palette:")
pal_al = extract_palette(OUT / "Autumn Light.jpeg")
for c, w in pal_al:
    print(f"  {w*100:5.1f}%  rgb{c}  #{c[0]:02x}{c[1]:02x}{c[2]:02x}")

print("\nWhere the Forest Breathes palette:")
pal_wfb = extract_palette(OUT / "Where the Forest Breathes.jpeg")
for c, w in pal_wfb:
    print(f"  {w*100:5.1f}%  rgb{c}  #{c[0]:02x}{c[1]:02x}{c[2]:02x}")


def pick_gradient_autumn_light(palette):
    """Sky-to-trees gradient: violet-blue → sunset gold → burnt orange."""
    # Pick a cool blue (highest blue, low red+green), a warm gold (high R+G, low B),
    # and a deep red-orange (high red, low blue, mid-darkness)
    blues, golds, reds = [], [], []
    for (r, g, b), w in palette:
        if b > r and b > g and b > 70:
            blues.append((r, g, b, w))
        if r > 150 and g > 100 and b < 110 and (r - b) > 70:
            golds.append((r, g, b, w))
        if r > 100 and r > g + 20 and (r - b) > 30 and r + g + b < 380:
            reds.append((r, g, b, w))

    def pick(lst):
        return max(lst, key=lambda x: x[3])[:3] if lst else None

    blue = pick(blues) or (60, 80, 130)
    gold = pick(golds) or (220, 170, 80)
    red = pick(reds) or (140, 70, 40)
    return [tuple(map(int, blue)), tuple(map(int, gold)), tuple(map(int, red))]


def pick_gradient_forest(palette):
    """Forest gradient: pale sky-yellow → autumn yellow → pine green."""
    # Light warm yellow (high R+G, B mid, high brightness), saturated yellow,
    # deep forest green
    pale, yellows, greens = [], [], []
    for (r, g, b), w in palette:
        bright = r + g + b
        if bright > 480 and r > 150 and g > 150:
            pale.append((r, g, b, w))
        if 130 < r < 220 and g > 120 and b < 100 and r >= g and (r + g) > 280:
            yellows.append((r, g, b, w))
        if g > r and g > 80 and r + g + b < 360:
            greens.append((r, g, b, w))

    def pick(lst):
        return max(lst, key=lambda x: x[3])[:3] if lst else None

    pale_v = pick(pale) or (225, 215, 175)
    yel = pick(yellows) or (200, 145, 60)
    grn = pick(greens) or (50, 75, 50)
    return [tuple(map(int, pale_v)), tuple(map(int, yel)), tuple(map(int, grn))]


grad_al = pick_gradient_autumn_light(pal_al)
grad_wfb = pick_gradient_forest(pal_wfb)
print(f"\nAutumn Light gradient stops: {grad_al}")
print(f"Forest gradient stops:       {grad_wfb}")

# Two letter-color options per painting (test which reads better)
MOCKUPS = [
    ("AL-ink",      variant_factory(grad_al,  INK,       "AL-ink"),
     "Autumn Light · ink letters"),
    ("AL-darkbrn",  variant_factory(grad_al,  TEXT_DK,   "AL-darkbrn"),
     "Autumn Light · dark-brown letters"),
    ("WFB-ink",     variant_factory(grad_wfb, INK,       "WFB-ink"),
     "Forest Breathes · ink letters"),
    ("WFB-darkbrn", variant_factory(grad_wfb, TEXT_DK,   "WFB-darkbrn"),
     "Forest Breathes · dark-brown letters"),
]

for slug, fn, _ in MOCKUPS:
    for size in (16, 32, 64, 256):
        img = fn(size)
        img.save(OUT / f"painting-{slug}.{size}.png")


# Comparison sheet — 2x2 grid
def label(d, text, x, y, size=20, fill=LABEL_TX, anchor="mm"):
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Georgia.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    d.text((x, y), text, fill=fill, font=font, anchor=anchor)


SHEET_W = 1100
SHEET_H = 1080
PAD = 50
sheet = Image.new('RGBA', (SHEET_W, SHEET_H), PAPER)
d = ImageDraw.Draw(sheet)

label(d, "SM monogram — gradients sampled from paintings",
      SHEET_W // 2, 45, size=26)
label(d, "left col: Autumn Light  •  right col: Where the Forest Breathes  •  letters: ink vs warm-dark",
      SHEET_W // 2, 80, size=13, fill=(100, 96, 88, 255))

cell_w = (SHEET_W - 2 * PAD) // 2
cell_h = (SHEET_H - 140) // 2
big_size = 200

for i, (slug, fn, name) in enumerate(MOCKUPS):
    # Layout: col by painting (i % 2 == 0 → AL is left? actually:
    # 0=AL-ink, 1=AL-darkbrn, 2=WFB-ink, 3=WFB-darkbrn
    # Want: col 0 = AL (rows = letter colors), col 1 = WFB
    col = 0 if slug.startswith("AL") else 1
    row = 0 if "ink" in slug else 1
    cx = PAD + col * cell_w + cell_w // 2
    cy_top = 110 + row * cell_h

    label(d, name, cx, cy_top + 25, size=16)

    big_y = cy_top + 55
    big = fn(256).resize((big_size, big_size), Image.LANCZOS)
    box = Image.new('RGBA', (big_size + 14, big_size + 14), BONE)
    bd = ImageDraw.Draw(box)
    bd.rectangle([(0, 0), (box.width - 1, box.height - 1)],
                 outline=(214, 207, 193, 255))
    sheet.paste(box, (cx - box.width // 2, big_y - 7), box)
    sheet.paste(big, (cx - big_size // 2, big_y), big)

    sizes = [64, 32, 16]
    total_w = sum(sizes) + 24 * (len(sizes) - 1)
    x_cursor = cx - total_w // 2
    sizes_y = big_y + big_size + 25
    for sz in sizes:
        img = fn(sz)
        y = sizes_y + (sizes[0] - sz) // 2
        sheet.paste(img, (x_cursor, y), img)
        label(d, f"{sz}px", x_cursor + sz // 2, sizes_y + 80, size=11,
              fill=(100, 96, 88, 255))
        x_cursor += sz + 24

sheet.convert('RGB').save(OUT / "comparison-painting-palettes.png", quality=95)
print("\nwrote:", OUT / "comparison-painting-palettes.png")
