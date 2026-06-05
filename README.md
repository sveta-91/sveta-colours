# Sveta Colours

Personal art portfolio site + local image-processing tooling for Sveta.

## Layout

```
index.html                       — main portfolio page (editorial gallery)
CNAME                            — custom domain for GitHub Pages
docs/
  roadmap.md                     — image pipeline implementation roadmap
  photo-manager-setup.md         — one-time Google setup + acceptance checklist
  superpowers/                   — design/spec docs and rewrite plans
tools/
  photo_manager.html             — local HTML page for processing photos end-to-end
  image_pipeline/                — Python pipeline (download → process → upload)
CLAUDE.md                        — project instructions for Claude Code agents
```

## Photo Manager (`tools/photo_manager.html`)

Local-only HTML page for processing an artwork photo end-to-end:
upload → AI-suggested crop/perspective/adjustments → manual touch-up → watermark
→ Drive folder + Sheets row. Served via `python3 -m http.server` from the repo root.

See `docs/photo-manager-setup.md` for one-time Google setup and the acceptance checklist.

## Image pipeline (`tools/image_pipeline/`)

Python pipeline that downloads artwork photos from a Google Drive inbox, normalizes,
straightens, and crops them, then re-uploads. See `CLAUDE.md` and `docs/roadmap.md`
for stage-by-stage details.
