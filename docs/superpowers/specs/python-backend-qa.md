# QA Scenarios — Tester

Scope: `photo_manager.html` calling local Flask backend (`http://localhost:8001`) for `/health` and `/analyze`. Gemini stays as fallback. Adversarial mindset carried from `audit-r1-tester.md`: silent failure paths are blockers; "spinner forever" is a release-stopper.

## Must-pass (blocker class)

1. **Backend down at page load** | Open `photo_manager.html`, Flask not running. | Page renders, AI card shows "Geometry: Gemini fallback" amber badge within ~500 ms. Upload + Auto-correct still work using Gemini geometry. | CRITICAL if it hangs or throws uncaught.
2. **Backend down, then user uploads** | Same as #1, upload a known-tilted photo (Lost In The Fog). Click Auto-correct. | Three-step UI runs end-to-end using Gemini corners. No "Working…" spinner stuck. Status reads success. | CRITICAL.
3. **Backend crashes mid-session** | Start with backend up, upload one photo, kill `flask` process, upload a second photo. | Second analyze call fails fast (catch network error), badge flips to amber, Gemini fallback fires automatically. No infinite spinner. | CRITICAL.
4. **Auto-correct double-click during in-flight `/analyze`** | Upload photo, immediately click Auto-correct twice in <100 ms. | Second click is a no-op (button disabled synchronously, matches existing pattern at line 527). Single Python call. | CRITICAL — duplicate POSTs corrupt `state.aiData`.
5. **Save before Auto-correct finishes** | Upload, click Auto-correct, while spinner is spinning click "Save to Drive + Sheets". | Either Save is disabled while `aiApplied` is incomplete, or Save proceeds with the *current* canvas state (pre-correction), but never with partially-applied geometry. No torn state. | CRITICAL.
6. **CORS denied** | Backend running but `flask-cors` configured for wrong origin (e.g. `localhost:8000` while user opened the page via `file://` or `127.0.0.1:8000`). | Browser console shows CORS error; page treats backend as unavailable; falls back to Gemini. Not a hang. | CRITICAL.
7. **Backend returns malformed JSON** | Mock backend returns `"oops"` (not JSON) with 200. | `JSON.parse` throws → caught → Gemini fallback fires, badge flips amber. | HIGH-CRITICAL — uncaught throw freezes Auto-correct.
8. **Backend returns NaN / Infinity in corner coords** | Backend returns `{perspective_corners:{tl:{x:NaN,y:0},...}}` (degenerate Hough). | Frontend validates fractions ∈ [0,1] finite before applying; if invalid, sets `needs_perspective_correction:false` and continues. No `computeHomography` blow-up (DLT throws "Degenerate homography" at line 909 — must be caught). | CRITICAL.
9. **`_find_frame_corners_from_hough` returns None** (pure-color image, no edges, or tiny 50×50 image) | Upload a 200×200 solid-grey PNG. | Backend returns `needs_perspective_correction:false, needs_crop:false, source:"python"`. Auto-correct completes as no-op with green checks. | CRITICAL — must not 500.
10. **Backend returns HTTP 500** | Force exception inside `/analyze` (corrupted JPEG path). | Frontend catches `!res.ok` → Gemini fallback. No spinner-forever. | CRITICAL.

## Should-pass (high)

11. **HEIC from iPhone** | Drop a real iPhone `.heic`. | Browser decodes via canvas → `composeFullBlob` sends JPEG to backend (per arch §3 last bullet). Backend never sees HEIC. Geometry returns. | HIGH — if decode silently fails (audit r1 bug #2), backend never gets called.
12. **30 MP JPEG (~6000×5000)** | Drop a DSLR shot. | `composeFullBlob` produces ~5–10 MB JPEG; POST completes in <5 s; backend `cv2.imdecode` handles it; no canvas OOM on Safari. | HIGH.
13. **Two photos uploaded fast in sequence** | Drop photo A, before its `/analyze` returns drop photo B. | Photo A's analyze response is discarded (stale — `state.origImg` no longer matches), or its result is gated by a request token. No photo-A geometry painted onto photo-B canvas. | HIGH — cross-photo bleed is data corruption.
14. **Backend takes 30 s (slow Hough on 30 MP)** | Throttle backend with `time.sleep(30)` in `/analyze`. | Frontend shows spinner with a hint ("Still working…") after ~5 s. Abort button or auto-timeout at 20 s falls back to Gemini. | HIGH — Sveta will assume crash.
15. **Already-perfectly-rectangular photo** ("Autumn Light" from test set) | Upload, Auto-correct. | Backend returns `needs_perspective_correction:false, needs_crop:false`. Three steps tick green immediately, no actual transform applied, no double-encode quality loss. | HIGH.
16. **"Honey Air" known-miss case** | Upload Honey Air from Sveta's reference set. | Backend returns *something* (currently misses crop). Frontend doesn't crash; user can manually crop. Status copy doesn't claim "perfectly cropped". | HIGH — must not hide the miss behind a green check.
17. **Auto-correct → Reset → Auto-correct** | Apply, click Reset, click Auto-correct again. | Second run re-applies (matches existing reset logic at line 1084 clearing `aiApplied`). Cache `state.aiData` is reused (no second `/analyze` call). | HIGH — if it re-POSTs, that's wasted compute; if it uses stale dims, that's a bug.
18. **Toggle Python off mid-session via dev/settings** | Disable `state.pythonBackend.available` manually in console, run Auto-correct on a new photo. | Gemini fallback fires cleanly. No leftover "Python source" labels. | HIGH.
19. **Port collision** (8001 occupied by another tool) | Start backend, it fails to bind. | Backend script exits with clear error to Sveta's terminal; page treats it as unavailable. Page itself never broken. | HIGH — UX concern, not page-level CRITICAL.
20. **Coordinate-frame mismatch under straighten-then-crop** (arch open Q #1) | Upload Snowfall (needs both). | Crop rect is correctly applied to the *straightened* canvas, not the original. Visual: no double-cropping, no off-by-perspective offset. | HIGH — this is the integration's hardest correctness bug.

## Nice-to-handle (medium/low)

21. **Slow 3G localhost** (DevTools throttle to "Slow 3G") | Auto-correct. | Spinner held longer; eventually resolves. No timeout-induced fallback if the call is going to succeed in <20 s. | MEDIUM.
22. **Auto-correct in progress when user manually drags crop rectangle** | Start Auto-correct, before it finishes drag a crop rect on overlay. | Either crop tool is disabled while `aiApplied.crop=false`, or user's drag wins and Auto-correct's crop step is skipped (idempotency guard hits `aiApplied.crop=true` retroactively). No crop applied twice. | MEDIUM.
23. **Backend starts after page already open** | Page loaded with backend down (amber badge), Sveta starts Flask, uploads new photo. | Either page auto-rechecks `/health` on each upload, or there's a "Recheck" affordance. Otherwise badge is permanently amber until reload. | MEDIUM — reload-as-workaround is acceptable but document it.
24. **Backend on different port** (config drift, user edited `serve.sh`) | Backend on 8002. | Frontend `/health` to 8001 fails → amber badge → Gemini. Same code path as #1, just calling out the config-drift trigger. | MEDIUM.
25. **OpenCV raises on malformed JPEG** (truncated bytes, valid header) | Upload truncated JPEG file. | Browser `img.onerror` catches before backend is called (audit r1 bug #2 must be fixed first). If somehow it reaches backend, Flask returns 400 with error message, frontend falls back to Gemini. | MEDIUM.
26. **Backend returns 4 collinear corners** (degenerate quad — Hough found parallel lines only) | Force backend response with `tl=tr=br=bl` or 4 collinear points. | Frontend's `computeHomography` throws "Degenerate homography" (line 909) → caught at line 836 (`autoPersp` catch). Auto-correct continues to crop step. | MEDIUM.
27. **Backend returns crop bbox outside [0,1]** (e.g. `x:-0.05` from rounding) | Mock response with `crop:{x:-0.05,y:0,w:1.1,h:1}`. | Frontend clamps to [0,1] before applying, or rejects entirely. `autoApplyCrop` at line 845 needs validation — currently it would `drawImage` with negative offsets producing a fringe. | MEDIUM.
28. **Backend `/health` returns 200 but body says `{"ok":false}`** | Mock unhealthy backend. | Frontend treats as unavailable (must check `ok` field, not just status). | LOW.
29. **Multiple browser tabs open** (Sveta opens manager in 2 tabs, both call backend) | Both POST `/analyze` simultaneously. | Flask handles serially (default dev server is single-threaded); second request queues. Both succeed eventually. No state corruption — each tab owns its own `state`. | LOW.

## Test sequencing for acceptance

Run in this order to minimize setup churn and catch blockers early:

1. **Backend off** (scenarios 1, 2): open page, verify amber badge, upload, Auto-correct via Gemini. Confirms fallback path before adding the Python variable.
2. **Backend on, happy path** (15, 17, 20): start Flask, run the three reference photos covering "no work needed", "crop only", and "straighten+crop". Catches the coordinate-frame integration bug early.
3. **Backend on, adversarial inputs** (9, 11, 12, 16): tiny image, HEIC, 30 MP, "Honey Air". Confirms backend robustness on real-world edge cases.
4. **Backend failure injection** (3, 7, 8, 10, 26, 27): kill server, mock malformed responses. These need temporary code (devtools console overrides) but exercise every catch path.
5. **Concurrency + race** (4, 5, 13, 22): rapid-fire interactions. Hardest to repro; run last when other code paths are trusted.
6. **Config drift / environment** (6, 19, 23, 24): port collision, CORS, restart mid-session. Low frequency but each is a "deploy day" failure mode.
7. **Performance** (14, 21): throttling. Run last because they require DevTools setup and patience.

## What I'm uncertain about

1. **For the Image-Processing agent:** When `_find_frame_corners_from_hough` returns None (no quad found), should the response say `needs_perspective_correction:false` (silent skip) or include a confidence score so the frontend can show "couldn't auto-detect — adjust manually"? Silent skip risks Sveta thinking the photo was checked and is fine, when really we just gave up. (Mirrors arch open Q #2.)
2. **For the AI Manager:** What is the abort/timeout strategy when Python takes >20 s on a huge image? Hard timeout + fall back to Gemini, or show progress and let user cancel? `Promise.all` with `Promise.race(timeout)` is one option; AbortController on the fetch is cleaner.
3. **For the Frontend agent:** Should the "Geometry: Python/Gemini" badge be sticky per-photo (label the result actually used) or per-session (label what's *available*)? Per-photo is more honest for debugging Sveta's "this looked wrong" reports but adds UI noise.
