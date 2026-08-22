# Python Backend Architecture — AI Integration Manager

## Recommended stack

**Flask** + the existing `tools/image_pipeline/.venv`. Reasoning: single-user, single-machine, sync workload — async (FastAPI) buys nothing. `http.server` would force hand-rolled multipart parsing and CORS, which is wasted effort. Flask is one `pip install`, ~30 lines of glue, and gives us multipart, JSON responses, and CORS via `flask-cors` for free. No Pydantic ceremony, no ASGI server.

## Endpoint design

Return **coordinates, not pixels.** The browser already has working homography (`computeHomography` + `applyHomographyToCanvas`) and crop code. The Gemini pipeline is already shaped this way (`state.aiData.perspective_corners`, `state.aiData.crop`). Python becomes a drop-in replacement that returns the same JSON shape — no roundtrip of full-res image bytes, no quality loss from a server-side re-encode, no change to the apply path. Python only needs to *describe* the geometry.

| Endpoint | Method | Request | Response | What it does |
|---|---|---|---|---|
| `/health` | GET | — | `{"ok": true, "version": "1"}` | Liveness check, called on page load |
| `/analyze` | POST | multipart: `image=<jpeg/png>` | `{ needs_perspective_correction, perspective_corners: {tl,tr,br,bl with x,y in 0–1}, needs_crop, crop: {x,y,w,h in 0–1}, source: "python" }` | Runs `_find_frame_corners_from_hough` → `_needs_straightening` → `_needs_crop` and returns normalized fractions matching the existing Gemini schema |

That is it. One geometry endpoint. No `/process`, no separate `/perspective` and `/crop` — they share 90% of the work (`_find_frame_corners_from_hough` is used by both straighten and crop) and the browser orchestrates the order anyway.

Image arrives as bytes; decode via `cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)`. Confirmed cv2 path; no disk write needed.

## CORS & startup

Frontend at `http://localhost:8000`, backend at `http://localhost:8001`. Enable CORS with `flask-cors` restricted to `http://localhost:8000` only (not `*`). Single startup script `tools/image_pipeline/serve.sh`:

```
#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec flask --app server run --port 8001
```

Run both from repo root with one command via a tiny `Makefile` or `scripts/dev.sh` that backgrounds the Python server and foregrounds `python3 -m http.server 8000`. Two-process is fine; don't try to fuse them.

## Auto-correct orchestration

Run Python geometry and Gemini text **in parallel** from `autoCorrectAll`. They are independent: Python owns `perspective_corners`, `crop`, `needs_*`; Gemini owns `description`, `medium`, `style`, `tags`, and the brightness/contrast/saturation hints. `Promise.all([fetchPythonGeometry(), fetchGemini()])` then merge into `state.aiData`. Block until both resolve before painting `aiBox` and enabling auto-apply, but the existing three-step UI (persp → crop → adjust) stays exactly as-is — only the *source* of the geometry fields changes.

If both succeed: Python wins on geometry fields (overwrites whatever Gemini sent for `perspective_corners` / `crop`). Gemini's are never used when Python is up.

## Failure-mode + fallback

On page load, fire-and-forget `GET /health` with a 500ms timeout. Result stored as `state.pythonBackend = { available, checkedAt }`. Show a small badge in the AI card: "Geometry: Python" (green) or "Geometry: Gemini fallback" (amber).

In `autoCorrectAll`: if `state.pythonBackend.available` is false, skip the Python call and use Gemini's `perspective_corners` / `crop` as today. If the call was thought available but throws (backend crashed mid-session), catch, mark unavailable, and fall through to Gemini. Never block the user — manual crop/perspective UI is always reachable.

## What I'm uncertain about

1. **Coordinate-system contract under straighten-then-crop.** Python today re-runs `_needs_crop` on the *straightened* image. The browser will apply perspective first, then crop. Either Python returns crop coords in the **post-perspective** frame (cleaner; browser maps trivially) or in the **original** frame (harder; browser must transform the crop rect through the homography). Image-processing agent should rule.
2. **What `_find_frame_corners_from_hough` returns when it can't find a frame.** Today it returns `None` and the pipeline falls back to contour quad / bbox crop. For the browser response, should `needs_perspective_correction` be `false` (no quad found = no straightening) or should we surface confidence so the frontend can decide? QA/frontend should weigh in on the UX contract.
3. **Decode performance on 24MP HEIC uploads.** `cv2.imdecode` handles JPEG/PNG from memory fine. HEIC requires `pillow-heif` → PIL → numpy bridge, which adds 0.5–2s. The browser already normalizes HEIC to a canvas before any AI call, so we can require the backend always receive JPEG bytes from `composeFullBlob` — confirm with frontend agent that this is acceptable (it should be; that's already what Gemini gets).
