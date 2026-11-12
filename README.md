# Sveta Colours

Personal art portfolio site + local image-processing tooling for Sveta.

## Layout

```
index.html                       — main portfolio page (editorial gallery)
favicon.ico                      — root favicon (legacy auto-request)
CNAME                            — custom domain for GitHub Pages
icons/                           — favicon assets (SVG + multi-size PNG)
texts/                           — backup copies of the long site texts ([label] blocks); the live source is 3 published Google Docs (TEXT_DOCS)
  home.txt                       — [headline], [under headline], [from the studio], [on painting]
  about.txt                      — [about] paragraphs
  faq.txt                        — [question]/[answer] pairs, in page order
images/                          — local fallbacks for site photos served from Drive
  sveta-portrait.jpg             — painter photo (home hero); live copy is Drive file 1b-kjtmZ97SymW3DDZ7VI7XlgRNajyztm
  og-image.jpg                   — 1200×630 link-preview picture (og:image / twitter:image): Held Summer, Where the Forest Breathes, Autumn Light on --bg
inquiry/
  Code.gs                        — Apps Script backend for the inquiry form (not wired yet: FORM_ENDPOINT is empty)
docs/
  roadmap.md                     — image pipeline implementation roadmap
  photo-manager-setup.md         — one-time Google setup + acceptance checklist
  portfolio-playbook.md          — how to build a similar site from scratch; ends with "Extending the site" (backend plan)
  review.txt                     — external review of the site (sales, SEO, images), in Russian
  watermark-removal-playbook.md  — techniques for clearing watermarks off artwork photos
  superpowers/                   — design/spec docs and rewrite plans
scripts/
  audit_gallery.py               — read-only pre-deploy audit of the gallery sheet data
  verify_site.py                 — parameterized Selenium UI verification
  verify_collection_order.py     — Selenium check: collection order + button pictures vs the sheet rules
  dev.sh                         — start the local static server for the photo manager
  templates/                     — reusable Selenium verify templates
tools/
  photo_manager.html             — one-page publishing: prepared image → watermark → AI title/description → Drive + sheet → live check
  image_pipeline/                — Python pipeline (download → process → upload)
  favicon/                       — favicon generation scripts (palette sampling + final render)
.claude/
  skills/publish-painting/       — the per-painting publishing workflow (canonical spec)
  skills/sveta-writing/          — voice spec for site copy + bilingual prompt-reading rule
  brainstorm-panel/roster.md     — confirmed panel roster + standing rules
  brainstorm-panel/log.md        — per-panel retrospectives
CLAUDE.md                        — project instructions for Claude Code agents
```

## Photo Manager (`tools/photo_manager.html`)

Publishing a painting in one page. The image arrives already prepared (cropped);
the page never edits it except for an optional watermark.

1. Drop the image. It is uploaded **as is**; only a watermark (opacity > 0)
   re-saves it as JPEG 95%. A warning shows when the photo's proportions don't
   match the canvas `size`, or the image orientation disagrees with the form.
2. The record fills itself from the sheet: an existing title loads its row
   (Save then updates it), otherwise medium/size/price take the most common values.
3. **Title & description with AI**: a dialog with OpenAI about this painting.
   The voice rules are read live from `.claude/skills/sveta-writing/` (never
   copied). **Use this** fills the form; **Copy for corrections.md** keeps the
   voice skill learning. The API key is pasted once and stays in this browser.
4. **Save to Drive + Sheet** (under the form): uploads, writes the row by the
   sheet's own header (extra columns kept; `slug` written once if the column
   exists), moves `hero` if ticked, warns on a duplicate `recent` position, then
   watches the published CSV until the site shows the painting and links to it.

Run `scripts/dev.sh` and open http://localhost:8000/tools/photo_manager.html.
OAuth as Sveta.

See `docs/photo-manager-setup.md` for one-time Google setup and the acceptance checklist.

## Image pipeline (`tools/image_pipeline/`)

Python tooling. Two distinct workflows:

- **Per-painting portfolio publishing** (the active path): Grok web UI for
  perspective+crop → `trim_borders.py` to strip Grok's white padding → discuss
  title/description with Claude in chat → `push_painting.py` to upload + upsert
  the sheet row. The Claude Code path; `photo_manager.html` does the same in one page.
- **Batch pipeline** (`run_pipeline.py`): scan a Drive inbox folder, straighten
  tilted phone photos, crop background, save locally. Older workflow; see
  `docs/roadmap.md`.

See `tools/image_pipeline/README.md` for the full file list and per-painting
runbook.
