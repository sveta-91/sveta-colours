# Favicon Tools

Two scripts that survive from the 2026-06-12 favicon design run. Reusable for
any future favicon (rebrand, sub-site, social-share avatar).

## Scripts

| File | Purpose |
|---|---|
| `render_painting_palettes.py` | Download a named painting from the portfolio Drive folder, k-means cluster to 6 dominant colors, pick a 3-stop gradient (cool → warm → deep). Renders SM monogram preview on the sampled gradient with both ink-blue and warm-dark letter options. |
| `render_final.py` | Lock-in renderer for the chosen design: extracts glyph paths from a system font via `fontTools` (so the SVG renders identically without depending on the visitor's installed fonts), then emits SVG + PNG at all standard favicon sizes (16, 32, 48, 64, 180, 192, 512) + a multi-size `.ico`. |

## Setup

Both scripts share the `image_pipeline` venv:

```bash
cd ../image_pipeline
source .venv/bin/activate
pip install fonttools   # one-time, if not already there
```

## How to reuse for a new favicon

The full phase-by-phase recipe (concept → color → font → ship) lives in the
memory file `project_favicon_process.md` (loaded automatically by Claude when
the user asks for a favicon).

In short:

1. **Pick painting(s)** to sample colors from. Edit the painting IDs in
   `render_painting_palettes.py`. Run it. Review the gradient.
2. **Pick a font.** Edit `COCHIN_PATH` + `COCHIN_ITALIC_INDEX` in
   `render_final.py` to point at a different `.ttc` / index, OR change the
   `--ttc-index` of the same path.
3. **Edit `AL_STOPS` + `AL_HEX`** in `render_final.py` with the chosen
   3-stop gradient (from step 1).
4. **Run `render_final.py`** — emits SVG + 7 PNG + ICO to a `final/` subdir.
5. Copy outputs into `/icons/` (+ `/favicon.ico` at root) like the existing
   pair.

## Why these two scripts (not the 4 intermediate ones)

Only two phases generalize to "next favicon":
- sampling brand-real colors from artwork (phase 3 in the process recipe)
- locking-in the chosen design with font-as-paths + multi-size export (phase 5)

The other 4 intermediate scripts from the original run (`render_preview.py`,
`render_bright_variants.py`, `render_font_variants.py`,
the in-context concept generator) were tuned to specific A/B/C/D options
Sveta was choosing between in 2026-06-12. They aren't reusable as-is — easier
to rewrite a fresh small comparison renderer for whatever the next axis is.
