# Pipeline API Surface — Image Processing Expert

## Functions to expose

| Current function (`run_pipeline.py`) | Current signature | Proposed wrapper | Returns |
| --- | --- | --- | --- |
| `_find_frame_corners_from_hough` (L585) | `(img: np.ndarray) -> np.ndarray\|None` shape (4,2) order TL,TR,BR,BL | `detect_perspective_corners(img_bgr) -> {tl,tr,br,bl} fractions or None` | Coords (fractions 0–1) |
| `_needs_straightening` (L501) | `(img: np.ndarray) -> (bool, quad)` (contour fallback) | folded into `detect_perspective_corners` as fallback when Hough returns None | Coords |
| `_needs_crop` (L384) | `(img_path: Path) -> (bool, reason)` — **takes a Path, calls `cv2.imread` internally** | `detect_crop(img_bgr) -> {needs:bool, reason:str, rect:{x,y,w,h}\|null, corners:{...}\|null}` | Coords (rect + optional corners) |
| `_find_best_quad` (L470) | `(img) -> np.ndarray\|None` (contour) | internal helper inside `detect_crop` | — |
| `_detect_content_orientation` (L221) | `(img_path: Path) -> "portrait"\|"landscape"` — **Path-bound** | **skip for v1** — browser already handles EXIF (`3ac46b6`); content-orientation is a 90° rotation, not a perspective op | — |
| `_perspective_crop` (L457), `_straighten_image` (L681), `_crop_image` (L776), `_trim_frame_border` (L724) | path-in / path-out warping & frame trim | **do not expose** — browser owns the warp via `applyHomographyToCanvas` (L920) | — |

The crop pipeline currently exposes a bounding box only implicitly: `_check_content_coverage` (L361) returns `(coverage, bbox-string)`, and `_crop_image` (L817) computes `cv2.boundingRect` and slices `img[y:y+bh, x:x+bw]`. Wrapper must surface that `(x,y,bw,bh)` as fractions before the slice.

## Recommended approach

**Coords-out, single endpoint.** The browser already has a working CPU homography (`computeHomography`/`applyHomographyToCanvas` L885–954) and applies fractional corners against `state.fullCanvas` (see `runAutoPersp` L815 and `autoApplyCrop` L843). Returning fractions matches the existing Gemini contract one-for-one, keeps the full-res JPEG in the browser, and avoids re-encoding through Python (a recurring rule in `CLAUDE.md`: "Re-encode images that don't need processing"). Upload the JPEG to `/analyze`, decode with `cv2.imdecode`, return JSON only.

## Refactor scope

Minimal, new file — do **not** edit `run_pipeline.py`.

- **New:** `tools/image_pipeline/geometry.py` — pure-numpy/cv2 detectors. Lift the bodies of `_find_frame_corners_from_hough`, `_find_best_quad`, `_needs_straightening`, the three `_check_*` helpers, and `_needs_crop` into in-memory variants taking `img: np.ndarray`. Replace `cv2.imread(str(img_path))` calls (L229, L393, L683, L782) with the passed array. Convert `logging.info` to return-value diagnostics (or keep, harmless).
- **New:** `tools/image_pipeline/server.py` — Flask app, single `POST /analyze` route, returns `{needs_perspective_correction, perspective_corners, needs_crop, crop, reason}` matching `AI_SCHEMA` (photo_manager.html L680).
- **Keep:** `run_pipeline.py` imports from `geometry.py` to stay DRY (one-line change per call site), or leave untouched for v1 and dedupe later.
- **Untouched:** `drive_auth.py`, `config.json`, dry-run flow.

## Risks

- **Hardcoded paths:** `_needs_crop` (L393), `_detect_content_orientation` (L229), `_straighten_image` (L683), `_crop_image` (L782) all do `cv2.imread(str(path))`. Clean swap — `imread` is the only path touch.
- **Temp-file side effects:** `_straighten_image` (L703) and `_crop_image` (L847) write `_straightened_*.jpg` / `_cropped_*.jpg` next to the source. Wrapper must not call these — only the detectors.
- **`PipelineConfig` coupling:** `_straighten_image` and `_crop_image` take `config` for `jpg_quality`. The detectors we want do **not** need config — clean lift.
- **Corner-order contract:** `_order_corners` (L445) returns TL,TR,BR,BL by sum/diff of coords. Browser's `applyPersp` (L978–981) expects the same `tl,tr,br,bl` order. Match.
- **Out-of-frame corners:** `_find_frame_corners_from_hough` (L645) accepts corners up to 50px outside the image. Browser handles negative source coords (L937 clamps to transparent black). Acceptable — but normalize *after* clamping or fractions may exceed [0,1]; document or clip.
- **`needs_review` diagonal case:** `_needs_crop` (L439) returns `(False, "Needs manual review — diagonal background")`. Wrapper should pass this through as `needs_crop=false` with a `reason` field so the UI can show it.

## Compatibility with browser-side homography

Confirmed. The browser expects (photo_manager.html L691–699, L817–824, L880):

```
perspective_corners: { tl:{x,y}, tr:{x,y}, br:{x,y}, bl:{x,y} }  // fractions 0–1
crop:                { x, y, w, h }                              // fractions 0–1
```

Pipeline produces a (4,2) numpy array in TL,TR,BR,BL order from `_find_frame_corners_from_hough` and an `(x,y,bw,bh)` rect inside `_crop_image`. Both convert to the browser shape by `coord / image_dimension`. The browser's `runAutoPersp` (L815) and `autoApplyCrop` (L843) already consume exactly this schema from Gemini — drop-in replacement.

## What I'm uncertain about

1. **Deployment model:** is the Flask server expected to run as a local sidecar (`localhost:PORT` from `photo_manager.html`) on Sveta's laptop, or get packaged differently? CORS + lifecycle question for the Frontend agent.
2. **Failure semantics:** Gemini always returns *something*; the pipeline returns `None` from Hough corners ~30% of the time on the 20-image test set. Should `/analyze` fall back to Gemini for those, return `needs_perspective_correction:false`, or surface an explicit "could not detect" so the UI shows manual handles? AI-manager call.
3. **Crop corners vs rect:** `_find_frame_corners_from_hough` produces *4 corners* that are simultaneously the perspective quad **and** the crop region (see `_crop_image` L789–795 reusing them). Should `/analyze` return them once under both keys, or only emit `crop` as a rect (losing the trapezoid)? Affects whether browser warp+crop in one step or two.
