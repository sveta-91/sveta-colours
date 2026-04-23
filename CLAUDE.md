# Sveta Colours

Personal art portfolio site for Sveta. Contains both the static site and a local image processing pipeline.

## Project Structure

- `sveta-colours.html` — main site page
- `tools/image_pipeline/` — Python pipeline that downloads artwork photos from Google Drive, processes them, and re-uploads

## Image Pipeline

- **Entry point:** `tools/image_pipeline/run_pipeline.py`
- **Auth:** Google service account (`service-account-key.json`, git-ignored)
- **Config:** `config.json` (copy from `config.example.json`, git-ignored)
- **Python env:** `tools/image_pipeline/.venv/`
- **Dependencies:** `opencv-python`, `numpy`, `Pillow`, `pillow-heif`, `google-api-python-client`

### Pipeline stages (in order)
1. List source images from Google Drive inbox folder (implemented)
2. Download each file to `tmp/` (implemented)
3. Normalize to JPEG — HEIC/PNG → JPEG via Pillow + pillow-heif (implemented)
4. Detect orientation — content-aware for near-square images using OpenCV edge analysis (implemented)
5. Detect whether crop is needed — three-signal approach: Hough lines, corner-vs-center histograms, content coverage (implemented, 19/20 accuracy)
6. Crop artwork from background (not yet implemented — detection only so far)
7. Save processed output to `output/processed/{portrait,landscape}/` (implemented)
8. Upload result to Google Drive (not yet implemented)

### Crop detection approach
Uses three complementary signals in `_needs_crop()`:
- **Hough line detection** — finds straight canvas/frame edges in the outer border region
- **Corner-vs-center LAB histograms** — compares corner colour distributions to image centre (strict < 0.05, soft < 0.15 thresholds)
- **Content bounding box coverage** — checks if edge content fills the full image

Rules to avoid false positives:
- Hough lines on 3+ sides always trusted; on 2 sides only when coverage < 95%
- Corner histogram requires 3+ strict bg corners alone, or 2+ strict with Hough support
- Single Hough side confirmed only by corners with similarity in 0.0–0.12 range (excludes painted-sky negative correlations)
- One known miss: Honey Air (easel background with other paintings)

### Orientation detection
`_detect_content_orientation()` in `run_pipeline.py`:
- Non-square images: uses pixel dimensions directly
- Near-square images (aspect 0.92–1.08): analyses edge-detected content bounding box to find real artwork aspect ratio

### Running
```bash
cd tools/image_pipeline
source .venv/bin/activate
python run_pipeline.py
```

### Key conventions
- `dry_run: true` in config skips download and upload (listing still hits Drive API)
- Source images are iPhone photos (HEIC + JPEG)
- All temp/output/log dirs are git-ignored
- No Google Sheets integration yet

## Rules
- Do not commit `config.json`, `service-account-key.json`, or any credentials
- Do not add Google Sheets logic until explicitly asked
- Keep pipeline changes incremental — one stage at a time
- Test with `dry_run: true` before running with `dry_run: false`
- When tuning detection thresholds, verify against all 20 images — check for false positives and misses
