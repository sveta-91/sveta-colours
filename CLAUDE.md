# Sveta Colours

Personal art portfolio site for Sveta. Contains both the static site and a local image processing pipeline.

## Project Structure

- `sveta-colours.html` — main site page
- `tools/image_pipeline/` — Python pipeline that downloads artwork photos from Google Drive, processes them, and re-uploads
- `docs/roadmap.md` — pipeline implementation roadmap

## Image Pipeline

- **Entry point:** `tools/image_pipeline/run_pipeline.py`
- **Auth:** Google service account (`service-account-key.json`, git-ignored)
- **Config:** `config.json` (copy from `config.example.json`, git-ignored)
- **Python env:** `tools/image_pipeline/.venv/`
- **Dependencies:** `opencv-python`, `numpy`, `Pillow`, `pillow-heif`, `google-api-python-client`
- **Working dir for all pipeline commands:** `tools/image_pipeline/`

### Pipeline stages (in order)
1. List source images from Google Drive inbox folder (implemented)
2. Download each file to `tmp/` (implemented)
3. Normalize to JPEG — HEIC/PNG → JPEG via Pillow + pillow-heif (implemented)
4. Straighten tilted paintings — perspective correction via Hough line intersections (implemented)
5. Detect orientation — content-aware for near-square images (implemented)
6. Detect whether crop is needed — three-signal approach (implemented, 19/20 accuracy)
7. Crop artwork from background (not yet implemented)
8. Save processed output to `output/processed/{portrait,landscape}/` (implemented)
9. Upload result to Google Drive (not yet implemented)

### Key functions in run_pipeline.py
- `_detect_content_orientation()` — orientation for square images via edge bbox
- `_needs_crop()` — three-signal crop detection (Hough + corners + coverage)
- `_needs_straightening()` — contour-based angle detection
- `_find_frame_corners_from_hough()` — precise corners via Hough line intersections
- `_straighten_image()` — perspective correction (Hough preferred, contour fallback)
- `_check_hough_lines()` — find straight edges in border region
- `_check_corner_vs_center()` — LAB histogram comparison
- `_check_content_coverage()` — edge-based content bbox

### Test images (20 in Drive inbox)
Reference set for tuning — always verify changes against all 20:
- **Already straight, no crop needed (7):** Autumn Light, All Colors of Fall, Walking Home, MorningCoffee, Downtown, Mistress Of The Forest, From Under Eyelashes
- **Need crop, no straightening (10):** Copy of/Where the Forest Breathes, Waiting For Rain, Holding On In Winter, Morning Coffee, Held Summer, Summer Hut, Autumn Sun, Purple Sun, Honey Air (known miss)
- **Need straightening + crop (3):** Noise And Silence, Lost In The Fog, Snowfall

## Rules

### Never
- Commit `config.json`, `service-account-key.json`, or credentials
- Add Google Sheets logic until explicitly asked
- Straighten images that don't need it — always verify 0 false positives
- Re-encode images that don't need processing (pass-through for clean JPEGs)
- Run `dry_run: false` without user confirmation

### Always
- Work incrementally — one stage at a time, test before moving on
- Test with `dry_run: true` first
- After changing detection thresholds: run on all 20 images, check for both false positives AND misses
- Show the user the actual output images after processing changes
- Reset `dry_run: true` in config.json after real runs
- Clean output directories before test runs: `find output -name "*.jpeg" -delete`
- Activate venv before running: `cd tools/image_pipeline && source .venv/bin/activate`

### Pipeline development workflow
1. Make code changes
2. Verify imports: `python -c "import run_pipeline; print('OK')"`
3. Clear output: `find output -name "*.jpeg" -delete`
4. Run dry: `python run_pipeline.py` (with dry_run: true)
5. Run real: set dry_run: false, run, then immediately reset to true
6. Check results: view output images, compare to expectations
7. Show user the results before committing
