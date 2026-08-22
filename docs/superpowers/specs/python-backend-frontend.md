# Frontend Integration — Frontend Developer

## Trigger model

**On upload, in parallel with Gemini.** `analyzeWithAI(file)` already fires from `handleFile` and Gemini already eats 1-3s of latency. Adding a sibling `fetchPythonGeometry(file)` on the same tick makes Auto-correct feel instant when the user is ready, and the two calls compose into a single `state.aiData` (Python overwrites the geometry fields, Gemini owns text + adjustments). Lazy-on-click is rejected: the user hits Auto-correct expecting an immediate result, not a 1-3s spinner stacked after Gemini's already-finished response. Cost (one localhost round-trip per upload) is irrelevant on a single-user dev box.

Gate both calls on `state.pythonBackend.available` (checked once at page load via `GET /health` with 500 ms timeout — see python-backend-arch.md). If Python is down at upload time, skip the geometry fetch entirely; Gemini's coords are the fallback, no degraded mode.

## Request format

`POST http://localhost:8001/analyze`, multipart `image=<blob>`, JPEG quality 0.85. Source is `await composeFullBlob('image/jpeg', 0.85)` — already EXIF-corrected, already what Gemini receives, already small enough for a localhost POST (~2-4 MB for 24 MP). No PNG: pipeline reads with `cv2.imdecode`, JPEG is lossless enough for Hough-line detection, and a 24 MP PNG is 50+ MB. Response shape matches the existing Gemini schema (`needs_perspective_correction`, `perspective_corners.{tl,tr,br,bl}.{x,y}` as 0-1 fractions, `needs_crop`, `crop.{x,y,w,h}` as 0-1 fractions) so the apply path (`runAutoPersp`, `autoApplyCrop`) needs zero changes.

## Auto-correct flow (new)

Order is unchanged: **persp → crop → adjust**. The merge happens earlier, in `handleFile`/`analyzeWithAI`, so by the time `autoCorrectAll` runs, `state.aiData` already has Python's geometry fields (or Gemini's, on fallback). `autoCorrectAll` itself is untouched.

In `analyzeWithAI`, replace the lone `fetch(geminiUrl)` with:

```js
const [geminiData, pythonGeom] = await Promise.allSettled([
  fetchGemini(blob, key),
  state.pythonBackend.available ? fetchPythonGeometry(blob) : Promise.reject('no-py')
]);
// merge: Python wins on geometry, Gemini owns everything else
```

`autoSteps` strip gains a leading "Analyzing…" pill that appears during upload (not during Auto-correct click). The existing three pills (Perspective, Crop, Adjustments) stay; nothing reads "Python" — Sveta doesn't care which backend ran.

## Toggle / settings

Single source-of-truth badge in the AI card header: **"Geometry: Python"** (green dot) or **"Geometry: Gemini fallback"** (amber dot). Auto-detected from `/health`; no toggle. Sveta has one mental model — "the geometry brain". Backend URL is a `const PYTHON_BACKEND = 'http://localhost:8001'` near the existing `CLIENT_ID` constant; not configurable. If she ever needs to point at a different port she edits one line, same as `CLIENT_ID`.

## Error states

| State | Detection | User message | Fallback |
|---|---|---|---|
| Backend down at page load | `/health` fetch rejects or 500 ms timeout | Amber badge: "Geometry: Gemini fallback" | Skip Python call on upload; Gemini coords used. No retry. |
| Backend up at load, down at upload | `/analyze` fetch rejects | Amber badge flips; status: "Python offline — using Gemini" | `state.pythonBackend.available = false`; Gemini coords used. |
| Python returns malformed JSON / missing fields | Try/catch around response parsing + shape check | Console warn, badge flips amber, status: "Python returned bad data" | Gemini coords used. |
| Python returns invalid coords (NaN, out of 0-1) | Validator in merge step | Same as malformed | Gemini coords used. |
| `/analyze` timeout (5 s) | `AbortController` with 5000 ms | Same as backend-down-at-upload | Gemini coords. |

Manual crop/perspective UI is always reachable regardless of state. No "manual mode" hint needed — buttons are already there.

## Code touchpoints

All in `tools/photo_manager.html`:

- **New constant** near line ~245: `const PYTHON_BACKEND = 'http://localhost:8001';`
- **New state field** in `state` object (line ~248): `pythonBackend: { available: false, checkedAt: 0 }`
- **New init** at end of script: `checkPythonBackend()` — fires `GET /health` with 500 ms `AbortController`, sets `state.pythonBackend.available`, paints badge.
- **New function** `fetchPythonGeometry(blob)` next to `analyzeWithAI` (around line 611): POSTs multipart to `/analyze` with a 5 s `AbortController`, returns the parsed JSON or throws.
- **Modify** `analyzeWithAI` (line 611): wrap the existing Gemini fetch in `Promise.allSettled` alongside `fetchPythonGeometry`; merge results with Python winning on `needs_perspective_correction`, `perspective_corners`, `needs_crop`, `crop`. Existing `sanitizeAiData` still runs on the merged object.
- **Modify** badge area inside the `aiBox` card (line ~125 in HTML) — add a `<span id="geomBadge">` next to the AI card title.
- **No change** to `autoCorrectAll` (line 515), `runAutoPersp` (line 815), `autoApplyCrop` (line 843), `computeHomography`, or `applyHomographyToCanvas`. That's the whole point of returning fractions in the same schema.

## What I'm uncertain about

1. **Should `fetchPythonGeometry` send the already-cropped/straightened canvas if Sveta re-clicks Auto-correct after a manual edit?** Today `state.aiApplied` blocks re-runs entirely. If we ever loosen that, Python needs to re-analyze the current `fullCanvas`, not the upload. Image-processing agent: is re-analysis after manual edits in scope, or is one-shot-per-upload the contract?
2. **Should the `/health` check be re-fired periodically (e.g., before each `/analyze`)** to recover from a backend that came online mid-session? Or is the page-load probe + per-upload fallback enough? AI-manager agent already implies "fall through on throw" — confirming we don't need a polling loop.
3. **HEIC normalization timing.** `handleFile` already bakes HEIC → canvas via EXIF path before `analyzeWithAI` runs, so `composeFullBlob` always emits JPEG. The Python backend can assume JPEG-only. Confirming with image-processing agent this matches their decode path.
