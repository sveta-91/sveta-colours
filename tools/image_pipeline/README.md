# Sveta Colours — Image Pipeline

Local tools for processing artwork photos: perspective correction, cropping,
watermark removal, and publishing to the portfolio (Drive + Paintings sheet).

## Files

| File | Purpose |
|---|---|
| `push_painting.py` | One-shot CLI: upload one processed image to Drive + upsert its row in the Paintings sheet. OAuth as the user. The publishing step of the per-painting workflow below. |
| `trim_borders.py` | Strip uniform light/white borders from an image. Built for Grok's 1024×1024 white padding but generic. |
| `run_pipeline.py` | Batch pipeline: scan Drive inbox → straighten/crop/normalize → save to local `output/`. Service account auth. |
| `geometry.py` | Library: perspective correction (Hough line intersection), post-warp crop, content-orientation detection. Used by `run_pipeline.py` and `server.py`. |
| `server.py` | Flask sidecar exposing `/analyze` for browser tools. Wraps `geometry.analyze()`. |
| `watermark_diagnose.py` | Dump per-corner crops + brightness stats to plan a watermark removal. |
| `watermark_remove.py` | Apply a JSON-spec'd plan: TELEA inpaint + tone-matched NORMAL_CLONE. Honors `preserve_zones` (e.g. signature). |
| `drive_auth.py` | Google API auth helpers (service account + OAuth user) shared by the pipeline scripts. |
| `config.example.json` | Template for local config. |
| `config.json` | Your local config (git-ignored). |
| `oauth-client.json` | OAuth client (desktop type) — git-ignored. |
| `oauth-token.json` | Cached refresh token for `run_pipeline.py` (drive scope) — git-ignored. |
| `push-painting-token.json` | Cached refresh token for `push_painting.py` (drive + spreadsheets scopes) — git-ignored. |
| `service-account-key.json` | Service account key for batch pipeline reads — git-ignored. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
cd tools/image_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install --only-binary cryptography -r requirements.txt
cp config.example.json config.json
# Fill in Drive folder IDs in config.json
```

`--only-binary cryptography`: on the macOS system Python (3.9, x86_64) the newest
`cryptography` ships no wheel, so a plain install tries to compile it from Rust
and fails on missing OpenSSL headers. The flag makes pip pick the newest version
that has a prebuilt wheel. `selenium` is in `requirements.txt` for the UI
verify scripts (`scripts/verify_site.py`). `.venv/` is gitignored.

You need:
- `service-account-key.json` (download from GCP) — used by `run_pipeline.py` reads
- `oauth-client.json` (desktop OAuth client from GCP) — used by `push_painting.py` (and `run_pipeline.py` uploads)

The OAuth token files are written on first browser auth and reused after.

## Portfolio publishing workflow (per painting)

The end-to-end recipe Sveta + Claude follow to add or update a single painting
on the portfolio site. Each step is reusable; nothing is fully automated
because Grok is the only tool that does perspective+crop reliably on artwork,
and it's accessed via web UI (no API key).

### 1. Perspective-correct + crop in Grok (manual, web UI)

Upload the raw painting photo to Grok with this prompt:

```
Perspective-correct the uploaded photo and crop to the painted canvas only.
Remove all wall/background and all frame/border areas. Do not normalize, redraw,
or restyle the painting; preserve the original artwork exactly, including the
signature if it is on the painted surface. Compare result and original painting —
the original artwork must be preserved exactly, failure if it isn't.
```

Download Grok's output to `~/Downloads/` (or drop into chat — Claude will save
it to `~/Downloads/<title>.jpeg` for you).

**Known artifact**: Grok sometimes pads the output to 1024×1024 with white bars
on the short axis. When that happens, step 2 strips them. When Grok returns
a clean rectangle (most of the time), skip step 2.

### 2. Trim Grok's white padding *(only if needed)*

```bash
python trim_borders.py \
  ~/Downloads/<grok-output>.jpg \
  ~/Downloads/<title>-trimmed.jpeg
```

Reports trim per edge; refuses to crop to an empty image. Pure pixel slicing —
no resampling, no re-encoding of pixel data, just JPEG q95 re-save.

If borders survive (off-white, light grey), lower `--min-bright` (default 240)
or raise `--max-std` (default 8).

### 3. Discuss title + description with Claude (in chat)

Drop the image into the chat. Claude proposes title + description and
discusses until you confirm. Description style (Sveta's voice, from existing
rows):

- 1–3 sentences, ≤60 words
- Fragment-style is fine: parallel participles, "X: Y" openers, possessive
  ("of my own", "kept close")
- Lean on contrast / palette / mood rather than literal subject

If a row with that title already exists in the sheet, Claude pulls the existing
medium / price / size / collection / orientation / hero / recent values and asks
only about empty fields.

### 4. Push to Drive + Sheet

```bash
python push_painting.py --site sveta \
  ~/Downloads/<title>.jpeg \
  --title "..." \
  --description "..." \
  --medium "acrylic" \
  --size "16x20 in" \
  --price "400" \
  --collection "Flowers,Seasons" \
  [--orientation portrait|landscape] \
  [--hero] [--sold] [--recent N] [--dry-run]
```

What it does:

- **Drive**: looks for `<title>.jpeg` in the portfolio folder. If found, replaces
  the bytes in place (file ID stable, sheet pointer doesn't move). If not, creates
  a new file. All portfolio files use `.jpeg` extension; the CLI normalizes
  `.jpg` → `.jpeg`.
- **Sheet** (`Paintings`): matches existing row by case-insensitive title in
  column A. Writes all 11 columns (update or append).
- **Preserve existing values**: for an *existing* row, any column not explicitly
  passed on the CLI keeps its current sheet value — so manual edits to `hero`,
  `recent`, or `orientation` survive a re-push. For a *new* row, `orientation`
  auto-detects from image dimensions and `sold` defaults to `no`; everything
  else is empty unless passed.
- **Orphan cleanup**: if the sheet row had a different old image ID before the
  push, that old Drive file is moved to Trash (recoverable for 30 days).
- **Auth**: first run opens a browser (consent: Drive + Sheets), token cached
  to `push-painting-token.json` — no browser on subsequent runs.

Use `--dry-run` to print what would happen without mutating Drive or sheet.

**Flags reference:**

| flag | effect |
|---|---|
| `--orientation portrait\|landscape` | Override auto-detect. Auto: `landscape` if width > height, else `portrait`. |
| `--sold` | Set `sold=yes`. To revert to `no`, edit the sheet (or pass everything except this flag — for an existing row, omitting the flag preserves whatever is there). |
| `--hero` | Set `hero=TRUE`. To unset (e.g. when promoting a different painting), edit the sheet manually. |
| `--recent N` | Set `recent='N'` (positions 1–6 represent the order in the home-page recent carousel). Pass `--recent ''` to clear. |
| `--collection "A,B"` | Multi-value as comma-separated string. Existing values: `Big City`, `Flowers`, `Forest`, `Seasons`. |

## Batch pipeline (older, separate workflow)

`run_pipeline.py` handles bulk processing of the Drive inbox — straightening
tilted phone photos, cropping background, detecting orientation. Output lands
in local `output/processed/{portrait,landscape}/`. Not part of the portfolio
publishing flow above.

```bash
python run_pipeline.py
```

`dry_run: true` by default in `config.json`. Reset to `true` after any real
run.

## Constants (for reference)

| | Value |
|---|---|
| Portfolio Drive folder | `1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX` |
| Paintings sheet | `170BuYN45Wk7H2g31gfRo35quok76JwrhFB8YV4-ckfI` |
| Sheet tab | `Paintings` |
| Columns (11, as of 2026-06-09) | `title, medium, price, sold, size, image, description, collection, orientation, hero, recent` |
| File extension convention | `.jpeg` (all existing portfolio files use this — `push_painting.py` normalizes `.jpg` → `.jpeg`) |
| `hero` semantics | `TRUE` or empty. Only one painting is hero at a time (homepage hero image). |
| `recent` semantics | `1`–`6` or empty. Position in the "recent works" carousel on the home page. |
| `orientation` semantics | `portrait` or `landscape`. Drives card sizing in the flex gallery: landscape cards are 350px wide with transposed image dims, NOT a 2-column span (see `docs/portfolio-playbook.md`). |
| Multi-value `collection` | Comma-separated, no spaces: `Seasons,Flowers`. |
