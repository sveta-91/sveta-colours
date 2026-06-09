# Sveta Colours

Personal art portfolio site + local image-processing tooling for Sveta.

## Layout

```
index.html                       — main portfolio page (editorial gallery)
CNAME                            — custom domain for GitHub Pages
docs/
  roadmap.md                     — image pipeline implementation roadmap
  photo-manager-setup.md         — one-time Google setup + acceptance checklist
  portfolio-playbook.md          — how to build a similar site from scratch
  superpowers/                   — design/spec docs and rewrite plans
scripts/
  templates/                     — reusable Selenium verify templates
tools/
  photo_manager.html             — local HTML page for processing photos end-to-end
  image_pipeline/                — Python pipeline (download → process → upload)
CLAUDE.md                        — project instructions for Claude Code agents
```

## Photo Manager (`tools/photo_manager.html`)

Local browser tool to upload a processed painting to the portfolio: drag-drop
image → fill form (defaults pulled from the sheet's most-common values) →
brightness/contrast/saturation/watermark adjustments → Save to Drive + Paintings
sheet. OAuth as Sveta. Served via `python3 -m http.server` from the repo root.

See `docs/photo-manager-setup.md` for one-time Google setup and the acceptance checklist.

## Image pipeline (`tools/image_pipeline/`)

Python tooling. Two distinct workflows:

- **Per-painting portfolio publishing** (the active path): Grok web UI for
  perspective+crop → `trim_borders.py` to strip Grok's white padding → discuss
  title/description with Claude in chat → `push_painting.py` to upload + upsert
  the sheet row. CLI alternative to `photo_manager.html`.
- **Batch pipeline** (`run_pipeline.py`): scan a Drive inbox folder, straighten
  tilted phone photos, crop background, save locally. Older workflow; see
  `docs/roadmap.md`.

See `tools/image_pipeline/README.md` for the full file list and per-painting
runbook.
