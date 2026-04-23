# Image Pipeline Roadmap

## Completed
- [x] Local scaffold (`run_pipeline.py`, config, requirements, directories)
- [x] Google Drive service account auth
- [x] List source images from Drive inbox (20 images found)
- [x] Download from Drive to `tmp/`
- [x] JPEG normalization (HEIC/PNG → JPEG via Pillow + pillow-heif)
- [x] Orientation detection — content-aware for square images using OpenCV
- [x] Crop-needed detection — three-signal approach (Hough lines, corner histograms, coverage)
- [x] Save processed output to `output/processed/{portrait,landscape}/`

## Next: Actual Artwork Cropping
- [ ] Implement crop for images flagged by `_needs_crop()`
- [ ] Handle different background types: plain walls, wooden tables, easels, frames, dark backgrounds
- [ ] Perspective correction for angled shots
- [ ] Route low-confidence crops to `output/needs-review/`
- [ ] Improve Honey Air-type detection (busy background with other artwork)

## Later: Drive Upload
- [ ] Upload processed files to processed Drive folder
- [ ] Upload needs-review files to needs-review Drive folder

## Future
- [ ] Google Sheets integration
- [ ] Light/color normalization
- [ ] HEIC source support (when HEIC files appear in inbox)
