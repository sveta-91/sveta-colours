# Watermark Removal Playbook (for paintings / scanned artwork)

A practical, repeatable process for removing watermarks from photos of paintings.
Built from a real job: a winter-forest acrylic with two watermarks — a bright
`@sveta-colours` + Weibo logo on snow (bottom-right), and a faint semi-transparent
text overlay on busy sky texture (top-left).

The core idea: **there is no single method.** Pick the technique based on *what the
watermark sits on*, not what the watermark says.

---

## 0. TL;DR decision tree

```
Is the watermark...
├─ bright/opaque text on a SMOOTH, fairly UNIFORM background (snow, sky, flat wall)?
│     → threshold mask + cv2.inpaint(TELEA)        [Method A]  ← clean, easy
│
├─ on BUSY / TEXTURED paint (brushstrokes, foliage, bark, detail)?
│     → cv2.seamlessClone(MIXED_CLONE) with a donor patch from the same image  [Method B]
│        (classical inpaint will smear here — don't use it)
│
└─ you have GPU/network and want the best possible result on texture?
      → LaMa neural inpainting  [Method C]  ← best, but needs model download
```

Most real paintings need **both A and B** because watermarks usually have one part
on a smooth area and one part on detail.

---

## 1. Environment setup

```bash
pip install opencv-python-headless pillow numpy scipy --break-system-packages
```

Optional (Method C — neural inpaint, only if network allows the model download):

```bash
pip install torch --break-system-packages
pip install simple-lama-inpainting --no-deps --break-system-packages
```

> **Gotcha learned:** `simple-lama-inpainting` pins an ancient Pillow and tries to
> build it from source → fails. Install it with `--no-deps` and rely on the Pillow /
> OpenCV you already have.
>
> **Bigger gotcha:** the LaMa weights (`big-lama.pt`) live on a GitHub *release* asset,
> which redirects to `release-assets.githubusercontent.com`. In a locked-down sandbox
> that host is blocked (`x-deny-reason: host_not_allowed`) and the download 403s. If
> Method C matters, get that host allow-listed first, or run it on an unrestricted
> machine. Otherwise fall back to A + B (they're usually enough).

---

## 2. Standard workflow (do this every time)

1. **Load, note size & mode.** Keep the original resolution; export at the end.
2. **Find every watermark.** Don't trust a single glance — corners and edges hide
   faint ones. Crop each corner at 2–4× and eyeball it.
3. **Classify each watermark** by its background (smooth vs textured) → pick A or B.
4. **Build a tight mask** per watermark (see methods). Always *visualize the mask*
   as a red overlay before inpainting — this is the step that saves you.
5. **Inpaint / clone**, then **view the result zoomed in** on each watermark.
6. **Iterate** the threshold / kernel / donor until clean. Cheap to re-run.
7. **Export** at original resolution: JPEG q95 (`subsampling=0`) for sharing,
   PNG for a lossless archive copy.

The single most useful habit: **render the mask overlay and the result crop, look at
them, adjust, repeat.** Don't fly blind.

---

## 3. Method A — bright text on a uniform background

Works because watermark glyphs are *brighter* than the clean background there.

**Step 1 — confirm separation.** Sample a clean patch of the background vs the text
and check the brightness gap. In the real job: clean snow maxed at ~234, text was
>245. That gap is what makes thresholding safe.

```python
import numpy as np
g = np.array(img.convert("RGB")).mean(axis=2)
print("clean bg max:", g[y0c:y1c, x0c:x1c].max())   # a clean patch
print("text p90/max:", np.percentile(g[yt0:yt1, xt0:xt1],90), g[yt0:yt1, xt0:xt1].max())
```

Pick a threshold a few points above the clean-background max.

**Step 2 — mask + inpaint.**

```python
import cv2, numpy as np
img  = cv2.imread("orig.jpg"); H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bx0, by0, bx1, by1 = 1885, 1745, W, H        # watermark bounding box
reg = gray[by0:by1, bx0:bx1]

m = (reg > 237).astype(np.uint8) * 255       # threshold = clean_bg_max + margin
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

# drop tiny specks (canvas-thread highlights), keep glyph-sized blobs
n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
clean = np.zeros_like(m)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= 20:
        clean[lab == i] = 255

mask = np.zeros((H, W), np.uint8); mask[by0:by1, bx0:bx1] = clean
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))  # ← key
out  = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
```

> **Gotcha learned — the "ghost letters":** semi-transparent watermarks have a *dark
> anti-aliased rim* around each bright glyph. If you mask only the bright core, the rim
> survives and you get faint embossed ghost text. **Fix: dilate the mask generously
> (≈15px kernel)** so it swallows the rim. Safe to over-dilate when the background is
> uniform — TELEA just fills more clean snow/sky.

---

## 4. Method B — watermark on busy / textured paint

Classical inpainting (TELEA/NS) **smears** over texture and leaves grey blobs — proven
in this job. Instead, transplant real texture from elsewhere in the *same* painting and
let Poisson blending fuse it.

**Why `MIXED_CLONE`:** it keeps the *stronger* gradient at each pixel. So genuine strong
features in the original (dark tree trunks, hard edges) survive, while the faint
low-gradient watermark gets overwritten by the donor's texture. `NORMAL_CLONE` does not
do this and tends to leave the watermark shapes visible.

**Step 1 — find a donor patch:** same size as the watermark box, watermark-free, with
the *closest tone*. Score candidates by tone distance (and optionally by edge energy if
you want a smoother donor):

```python
ring = img[y0-8:y1+8, x0-8:x1+8].reshape(-1,3).mean(0)   # tone around the box
for name,(sx,sy) in candidates.items():
    p = img[sy:sy+h, sx:sx+w]
    tone_diff = np.abs(p.reshape(-1,3).mean(0) - ring).sum()
    edge_var  = cv2.Laplacian(cv2.cvtColor(p,cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    print(name, "tonediff", round(tone_diff,1), "edgevar", round(edge_var,1))
```

Lowest `tonediff` wins. A patch *right next to* the watermark usually matches tone and
texture best. (In the job, the neighbouring patch at +190px, tonediff 7.4, beat
everything.)

**Step 2 — clone:**

```python
x0,y0,x1,y1 = 66,78,250,142; w,h = x1-x0, y1-y0
sx,sy = 300,78                              # chosen donor top-left
donor  = img[sy:sy+h, sx:sx+w].copy()
center = (x0 + w//2, y0 + h//2)
mask   = np.full((h, w), 255, np.uint8)     # full-box clone
out = cv2.seamlessClone(donor, img, mask, center, cv2.MIXED_CLONE)
```

**Optional tone-match** if no donor matches well — shift donor channels to the box tone
before cloning:

```python
target = img[y0-8:y1+8, x0-8:x1+8].reshape(-1,3).astype(np.float32).mean(0)
cur    = donor.reshape(-1,3).astype(np.float32).mean(0)
donor  = np.clip(donor.astype(np.float32) + (target - cur), 0, 255).astype(np.uint8)
```

> **Notes learned:**
> - A donor with its *own* strong trunks can import a conflicting trunk; a smoother
>   donor avoids that but may match tone worse. Try both, view both, pick the believable
>   one. "Looks like plausible painting" beats "technically the original layout."
> - `seamlessClone` needs the watermark *fully inside* your box — extend the box a few
>   px past the watermark or residual rings leak in from outside.
> - This reconstructs the patch; it won't be pixel-true to the original brushwork.
>   At full-image scale that's invisible. Accept it.

---

## 5. Method C — neural inpaint (best on texture, when available)

```python
from simple_lama_inpainting import SimpleLama
from PIL import Image
lama = SimpleLama()                                  # downloads big-lama.pt
result = lama(Image.open("orig.jpg").convert("RGB"),
              Image.open("mask.png").convert("L"))    # white = remove
result.save("out.png")
```

Give it a slightly *generous* box mask; LaMa synthesizes plausible texture and can even
continue trunks/edges. Use this for the textured watermarks if you can get the weights.
See the network gotcha in §1.

---

## 6. Export

```python
from PIL import Image
im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
im.save("clean.jpg", quality=95, subsampling=0)   # share
im.save("clean.png")                               # lossless archive
```

Always keep original resolution. Keep the original file untouched.

---

## 7. Quick checklist

- [ ] Found *all* watermarks (check every corner zoomed in)
- [ ] Classified each: smooth bg → A, textured bg → B
- [ ] Visualized each mask as a red overlay before inpainting
- [ ] Method A: threshold = clean-bg-max + margin; **dilated mask ~15px** to kill rim ghosts
- [ ] Method B: donor matched on tone; `MIXED_CLONE`; box fully covers the watermark
- [ ] Viewed every watermark spot zoomed in on the result
- [ ] Exported at full res (JPG q95 + PNG), original preserved

---

## 8. Failure modes seen (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| Faint embossed ghost letters remain | Masked bright core only, not the dark AA rim | Dilate mask ~15px |
| Whole bright canvas gets masked | Threshold too low; canvas texture is bright too | Raise threshold above clean-bg max; filter small components |
| Grey smeared blobs over detail | Used `cv2.inpaint` on textured area | Switch to Method B (`MIXED_CLONE`) |
| Watermark still visible after clone | Used `NORMAL_CLONE`, or watermark extends past box | Use `MIXED_CLONE`; enlarge box |
| Cloned patch looks wrong-coloured | Donor tone mismatch | Pick lower-`tonediff` donor or tone-match donor channels |
| `pip install simple-lama-inpainting` fails on Pillow | Pinned old Pillow builds from source | Install with `--no-deps` |
| LaMa weights 403 / won't download | Sandbox blocks GitHub release CDN | Allow-list the host or run elsewhere; else use A+B |
