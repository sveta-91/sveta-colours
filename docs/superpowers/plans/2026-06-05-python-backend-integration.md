# Python Backend Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Checkbox (`- [ ]`) steps for tracking.

**Goal:** Photo manager calls Sveta's existing Python pipeline (Hough-based perspective + 3-signal crop, 19/20 accuracy) for geometry, instead of Gemini's approximate coords. Gemini stays for description / medium / tags / brightness suggestions.

**Architecture:** Flask sidecar on `http://localhost:8001`. Single `POST /analyze` endpoint takes JPEG bytes, returns **the same JSON shape Gemini produces** (`perspective_corners`, `crop`, `needs_*` fractions). Browser merges Python geometry + Gemini text into `state.aiData` via `Promise.allSettled` — zero changes in `runAutoPersp` or `autoApplyCrop`. If backend is down, browser silently falls back to Gemini for geometry.

**Tech stack:** Flask (Python), existing pipeline functions extracted to `geometry.py`. Frontend: vanilla JS, AbortController, parallel fetch.

**Source specs:** `docs/superpowers/specs/python-backend-{arch,pipeline,frontend,qa}.md` — 4-agent design reports.

---

## File structure

```
tools/image_pipeline/
  run_pipeline.py            — UNCHANGED (CLI flow stays intact)
  geometry.py                — NEW: detector functions extracted, taking np.ndarray in
  server.py                  — NEW: Flask app, one /analyze + /health
  requirements.txt           — UPDATE: add Flask
docs/photo-manager-setup.md  — UPDATE: add backend setup + dev.sh
tools/photo_manager.html     — UPDATE: parallel Python+Gemini, badge, /health probe
scripts/dev.sh               — NEW: one-command startup (http.server + Flask)
```

---

## Response contract — `POST /analyze`

Request: multipart/form-data with one field `image` (JPEG or PNG bytes).

Response: 200 JSON, matching the Gemini schema subset so `state.aiData` merge is trivial:

```json
{
  "needs_perspective_correction": true,
  "perspective_corners": {
    "tl": {"x": 0.04, "y": 0.06},
    "tr": {"x": 0.97, "y": 0.03},
    "br": {"x": 0.95, "y": 0.94},
    "bl": {"x": 0.06, "y": 0.96}
  },
  "needs_crop": false,
  "crop": null,
  "source": "python",
  "reason": null
}
```

Rules:
- If perspective corners are found, set `needs_perspective=true`, `needs_crop=false` (the warp already crops to the quad).
- If no perspective is needed but crop bounding box found: `needs_perspective=false`, `needs_crop=true`, `crop={x,y,w,h}` as fractions.
- If pipeline finds neither: `needs_*=false`, `reason="no_frame_detected"`. Browser falls back to Gemini coords for that image.
- All fractions are in **original image space** (the JPEG bytes the browser sent — already EXIF-corrected by `handleFile`).
- Errors return 500 with `{"error": "..."}`. Browser treats as fallback-to-Gemini.

`GET /health` → `{"ok": true}` 200, used for the page-load badge probe.

---

## Task 1: Extract geometry detectors

**Files:**
- Create: `tools/image_pipeline/geometry.py`

- [ ] **Step 1: Write the module skeleton + extracted functions**

The current pipeline uses `cv2.imread(str(path))` in 4 places. The new module accepts a `numpy.ndarray` directly. Copy the function bodies verbatim from `run_pipeline.py` and change only the entry signature.

```python
"""
Geometry detectors used by both the offline pipeline and the photo_manager HTTP backend.

Public surface (everything else is implementation detail):
    analyze(img_bgr) -> dict with the shape documented in
    docs/superpowers/plans/2026-06-05-python-backend-integration.md.

The detectors do NOT mutate the input image. They return coordinates/decisions only;
the actual warp/crop is the caller's job (the browser uses CPU homography from
photo_manager.html commit 69cae7c; the CLI uses run_pipeline.py's warpers).
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

# ── Imports from the existing pipeline ──────────────────────────────────────
# We re-export by literal copy because run_pipeline.py is CLI-shaped (reads
# paths, writes paths, prints) and we don't want to depend on it at import time.
# Keep these in sync if run_pipeline.py is ever refactored.

# (paste the function bodies of _check_hough_lines, _check_corner_vs_center,
# _check_content_coverage, _order_corners, _needs_straightening,
# _find_frame_corners_from_hough VERBATIM from run_pipeline.py — they already
# operate on numpy arrays, no changes needed.)

# Replacement for _needs_crop that takes np.ndarray instead of a Path:
def _needs_crop_array(img: np.ndarray) -> tuple[bool, str, Optional[tuple[int, int, int, int]]]:
    """
    Returns (needs_crop, reason, bbox).
    bbox is (x, y, w, h) in pixels if available, else None.
    Same three-signal logic as run_pipeline.py:_needs_crop (line 384), but
    operates on the numpy array directly and ALSO returns the bbox when one is found.
    """
    # paste the body of _needs_crop, replacing the cv2.imread call with the parameter,
    # and where the existing function returns a bool+reason, also extract the bbox from
    # cv2.boundingRect on the same content edges and return it.
    ...

def analyze(img_bgr: np.ndarray) -> dict:
    """
    Top-level entry. Returns the dict shape the photo_manager expects.
    Coordinates are fractions [0,1] of original image dimensions.
    """
    h, w = img_bgr.shape[:2]
    if h < 100 or w < 100:
        return {
            "needs_perspective_correction": False,
            "perspective_corners": None,
            "needs_crop": False,
            "crop": None,
            "source": "python",
            "reason": "image_too_small",
        }

    # 1) Try Hough-based 4-corner detection (covers both straighten + crop)
    corners = _find_frame_corners_from_hough(img_bgr)
    if corners is not None and len(corners) == 4:
        ordered = _order_corners(corners)  # TL,TR,BR,BL
        # Clip to image bounds — Hough can overshoot by ~50px (run_pipeline.py:645)
        clipped = np.clip(ordered, [[0, 0]], [[w - 1, h - 1]])
        frac = clipped.astype(float) / np.array([w, h])
        return {
            "needs_perspective_correction": True,
            "perspective_corners": {
                "tl": {"x": float(frac[0][0]), "y": float(frac[0][1])},
                "tr": {"x": float(frac[1][0]), "y": float(frac[1][1])},
                "br": {"x": float(frac[2][0]), "y": float(frac[2][1])},
                "bl": {"x": float(frac[3][0]), "y": float(frac[3][1])},
            },
            "needs_crop": False,
            "crop": None,
            "source": "python",
            "reason": None,
        }

    # 2) No perspective needed — try a pure crop via the same _needs_crop signals
    needs, reason, bbox = _needs_crop_array(img_bgr)
    if needs and bbox is not None:
        x, y, bw, bh = bbox
        return {
            "needs_perspective_correction": False,
            "perspective_corners": None,
            "needs_crop": True,
            "crop": {
                "x": x / w, "y": y / h,
                "w": bw / w, "h": bh / h,
            },
            "source": "python",
            "reason": reason,
        }

    # 3) Neither — nothing to do, let browser fall back to Gemini coords if present
    return {
        "needs_perspective_correction": False,
        "perspective_corners": None,
        "needs_crop": False,
        "crop": None,
        "source": "python",
        "reason": "no_frame_detected",
    }
```

- [ ] **Step 2: Smoke-test the module via the existing test set**

```bash
cd /Users/sveta.mordkovich/private/sveta-colours/tools/image_pipeline
source .venv/bin/activate
python -c "
import cv2
from geometry import analyze
import os, json, pathlib
# Use one image from the existing test set — pick any JPEG in output/processed/ if exists,
# else any sample. If neither, skip and warn — full test happens via the server.
samples = list(pathlib.Path('output/processed').rglob('*.jpeg'))[:3] if pathlib.Path('output/processed').exists() else []
for p in samples:
    img = cv2.imread(str(p))
    print(p.name, '→', json.dumps(analyze(img), indent=2))
"
```

Expected: each image gets a dict back, no exceptions.

- [ ] **Step 3: Commit**

```bash
git add tools/image_pipeline/geometry.py
git commit -m "image-pipeline: extract Hough+3-signal detectors into geometry.py for HTTP reuse"
```

---

## Task 2: Flask server

**Files:**
- Create: `tools/image_pipeline/server.py`
- Modify: `tools/image_pipeline/requirements.txt` (or `pyproject.toml` if that's the canonical) — add Flask

- [ ] **Step 1: Add Flask to requirements**

Check if `requirements.txt` exists. If yes, append `flask`. If not, create one with the full dependency list per `CLAUDE.md`:

```
opencv-python
numpy
Pillow
pillow-heif
google-api-python-client
google-auth
google-auth-oauthlib
flask
```

- [ ] **Step 2: Install Flask in the venv**

```bash
cd /Users/sveta.mordkovich/private/sveta-colours/tools/image_pipeline
source .venv/bin/activate
pip install flask
```

- [ ] **Step 3: Write `server.py`**

```python
"""
photo_manager.html backend.

Wraps tools/image_pipeline/geometry.py behind a tiny Flask app so the
browser can ask 'where are the corners?' on every upload, parallel to
its Gemini call.

Run from the project root or from this dir:
    python -m tools.image_pipeline.server
    # or
    cd tools/image_pipeline && python server.py

CORS is hard-pinned to http://localhost:8000 (where http.server serves the page).
"""
from __future__ import annotations
import io
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

# When run as a script (`python server.py`), make the sibling module importable.
sys.path.insert(0, str(Path(__file__).parent))
from geometry import analyze  # noqa: E402

ALLOWED_ORIGIN = "http://localhost:8000"
MAX_BYTES = 50 * 1024 * 1024  # 50 MB, accommodates ~30 MP JPEG

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

log = logging.getLogger("pm-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.after_request
def add_cors(resp):
    return _cors_headers(resp)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True), 200


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    f = request.files.get("image")
    if f is None:
        # Allow raw body POST as a fallback (Content-Type: image/jpeg)
        raw = request.get_data()
        if not raw:
            return jsonify(error="missing image"), 400
        data = np.frombuffer(raw, dtype=np.uint8)
    else:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="could not decode image"), 400

    try:
        result = analyze(img)
    except Exception as e:  # detector exceptions shouldn't crash the server
        log.exception("analyze failed")
        return jsonify(error=str(e), source="python", reason="exception"), 500

    log.info("analyze: shape=%s result=%s", img.shape, {
        k: result[k] for k in ("needs_perspective_correction", "needs_crop", "reason")
    })
    return jsonify(result), 200


if __name__ == "__main__":
    # Bind only to loopback — this is a local sidecar, never exposed externally.
    app.run(host="127.0.0.1", port=8001, debug=False, threaded=True)
```

- [ ] **Step 4: Run + smoke test with curl**

```bash
cd /Users/sveta.mordkovich/private/sveta-colours/tools/image_pipeline
source .venv/bin/activate
python server.py &
SERVER_PID=$!
sleep 1

# Health probe
curl -s http://localhost:8001/health
# Expected: {"ok":true}

# Analyze a real image (pick any JPEG)
TEST=$(find output -name "*.jpeg" | head -1 || find tmp -name "*.jpeg" 2>/dev/null | head -1)
if [ -n "$TEST" ]; then
  curl -s -X POST -F "image=@$TEST" http://localhost:8001/analyze | python -m json.tool
fi

# CORS preflight
curl -s -X OPTIONS -H "Origin: http://localhost:8000" -H "Access-Control-Request-Method: POST" -i http://localhost:8001/analyze | head -5

kill $SERVER_PID
```

Expected: health returns `{"ok": true}`, analyze returns a dict with `needs_perspective_correction` etc., OPTIONS returns 204 with `Access-Control-Allow-Origin` header.

- [ ] **Step 5: Commit**

```bash
git add tools/image_pipeline/server.py tools/image_pipeline/requirements.txt
git commit -m "image-pipeline: Flask sidecar on :8001 exposing /analyze for photo_manager"
```

---

## Task 3: One-command dev startup

**Files:**
- Create: `scripts/dev.sh`
- Modify: `docs/photo-manager-setup.md` — replace "python3 -m http.server 8000" section with this script's usage

- [ ] **Step 1: Write `scripts/dev.sh`**

```bash
#!/usr/bin/env bash
# Start both the static file server (port 8000) and the Python backend (port 8001).
# Foregrounds the static server, backgrounds the backend; both die on Ctrl-C.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# Backend (Flask)
(
  cd tools/image_pipeline
  source .venv/bin/activate
  python server.py
) &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT INT TERM

# Give the backend a moment to bind
sleep 0.6

echo "Backend  ▸ http://localhost:8001  (pid $BACKEND_PID)"
echo "Frontend ▸ http://localhost:8000/tools/photo_manager.html"
echo

# Static file server (foreground)
python3 -m http.server 8000
```

- [ ] **Step 2: Make executable and smoke-test**

```bash
chmod +x scripts/dev.sh
./scripts/dev.sh &
DEV_PID=$!
sleep 2

curl -s http://localhost:8000/tools/photo_manager.html -o /dev/null -w "%{http_code}\n"   # 200
curl -s http://localhost:8001/health                                                     # {"ok":true}

kill $DEV_PID 2>/dev/null
lsof -ti:8000 -ti:8001 2>/dev/null | xargs kill 2>/dev/null || true
```

- [ ] **Step 3: Update `docs/photo-manager-setup.md`**

Replace the "Running the page" section's `python3 -m http.server 8000` command with:

```bash
./scripts/dev.sh
```

Add a one-paragraph note above it explaining that both servers must run (frontend serves the HTML, backend does Hough-based perspective/crop detection).

- [ ] **Step 4: Commit**

```bash
git add scripts/dev.sh docs/photo-manager-setup.md
git commit -m "tooling: scripts/dev.sh starts both http.server and Flask backend in one command"
```

---

## Task 4: Browser — `/health` probe + badge

**Files:**
- Modify: `tools/photo_manager.html`

- [ ] **Step 1: Add backend constants and state**

Near the existing constants (around the `DRIVE_FOLDER`/`SHEET_ID` block):

```js
const PYTHON_BACKEND = 'http://localhost:8001';
const PYTHON_BACKEND_HEALTH_TIMEOUT_MS = 500;
const PYTHON_BACKEND_ANALYZE_TIMEOUT_MS = 8000;   // 5s was too tight on 30-MP JPEGs
```

In `state` (the object introduced in Group B):

```js
pythonBackend: { available: false, checked: false, lastError: null },
```

- [ ] **Step 2: Add a badge to the AI card**

In the HTML, inside the `aiBox`'s card (currently around line ~125 area, the bottom-layout "AI analysis" card), insert after the card title:

```html
<div class="card-title-row" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">
  <span></span>
  <span id="geomBadge" class="status" aria-live="polite" style="font-size:0.75rem"></span>
</div>
```

(Adjust the existing layout if the card already wraps the title in a different structure — keep the badge to the right of the title.)

- [ ] **Step 3: Add the probe function**

Near the end of the script section (before `?test=1` block):

```js
async function checkPythonBackend() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PYTHON_BACKEND_HEALTH_TIMEOUT_MS);
  try {
    const res = await fetch(`${PYTHON_BACKEND}/health`, { signal: ctrl.signal });
    state.pythonBackend.available = res.ok;
    state.pythonBackend.lastError = res.ok ? null : `HTTP ${res.status}`;
  } catch (e) {
    state.pythonBackend.available = false;
    state.pythonBackend.lastError = e.name === 'AbortError' ? 'timeout' : e.message;
  } finally {
    clearTimeout(timer);
    state.pythonBackend.checked = true;
    renderGeomBadge();
  }
}

function renderGeomBadge() {
  const el = document.getElementById('geomBadge');
  if (!el) return;
  if (!state.pythonBackend.checked) { el.textContent = ''; return; }
  if (state.pythonBackend.available) {
    el.textContent = 'Geometry: Python ✓';
    el.style.color = 'var(--success)';
  } else {
    el.textContent = `Geometry: Gemini fallback (${state.pythonBackend.lastError || 'backend down'})`;
    el.style.color = 'var(--text-muted)';
  }
}

// Probe on page load
checkPythonBackend();
```

- [ ] **Step 4: Smoke test in browser**

Reload `http://localhost:8000/tools/photo_manager.html` while backend is running → badge shows `Geometry: Python ✓` in success color.
Kill the backend → reload page → badge shows `Geometry: Gemini fallback (...)` in muted color, within 500ms.

- [ ] **Step 5: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: /health probe + 'Geometry: Python/Gemini' badge"
```

---

## Task 5: Browser — call Python in parallel with Gemini

**Files:**
- Modify: `tools/photo_manager.html`

- [ ] **Step 1: Add `fetchPythonGeometry(blob)`**

Below `analyzeWithAI` (or above — placement doesn't matter):

```js
async function fetchPythonGeometry(blob) {
  if (!state.pythonBackend.available) throw new Error('backend not available');
  const form = new FormData();
  form.append('image', blob, 'image.jpg');
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), PYTHON_BACKEND_ANALYZE_TIMEOUT_MS);
  try {
    const res = await fetch(`${PYTHON_BACKEND}/analyze`, {
      method: 'POST', body: form, signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    return j;
  } finally {
    clearTimeout(timer);
  }
}
```

- [ ] **Step 2: Merge Python geometry into the AI flow**

Inside `analyzeWithAI`, AFTER the Gemini fetch produces `state.aiData` (the place where `state.aiData = sanitizeAiData(json)` is set), insert a parallel-but-after Python call. We use `await` here because we want the merge to be done before `analyzeWithAI` resolves:

```js
// (inside analyzeWithAI, after Gemini succeeds and state.aiData is set)

// If the Python backend reported healthy at page load, ask it for geometry.
// Errors here are silent — Gemini's geometry remains in place as fallback.
if (state.pythonBackend.available) {
  try {
    // Reuse the SAME downscaled JPEG we sent to Gemini, so coordinates match.
    // (downscaled.blob was created above; if it isn't in scope, recompute via
    //  composeFullBlob + downscaleToMaxEdge to 2048.)
    const geo = await fetchPythonGeometry(downscaled.blob);
    if (geo && geo.source === 'python') {
      // Override geometry fields when Python found something useful
      if (geo.needs_perspective_correction && geo.perspective_corners) {
        state.aiData.needs_perspective_correction = true;
        state.aiData.perspective_corners = geo.perspective_corners;
        state.aiData.needs_crop = false;
        state.aiData.crop = null;
      } else if (geo.needs_crop && geo.crop) {
        state.aiData.needs_perspective_correction = false;
        state.aiData.perspective_corners = null;
        state.aiData.needs_crop = true;
        state.aiData.crop = geo.crop;
      }
      // If Python said "no_frame_detected" — keep Gemini's geometry untouched
    }
  } catch (e) {
    // Flip badge to fallback for this session if backend goes down mid-flow
    state.pythonBackend.available = false;
    state.pythonBackend.lastError = e.name === 'AbortError' ? 'timeout' : e.message;
    renderGeomBadge();
    console.warn('Python geometry failed:', e);
  }
}
```

- [ ] **Step 3: Handle the no-Gemini-key case too**

When the user has no Gemini key, `analyzeWithAI` currently returns early (no-key path). If Python backend is available, geometry alone is still useful — apply it standalone. Modify the no-key branch:

```js
// (inside analyzeWithAI, the existing no-key branch)
const key = ensureGeminiKey();
if (!key) {
  // No text analysis possible, but Python geometry can still help
  setStatus('aiBox', 'No Gemini key — geometry via Python only.', null);
  document.getElementById('btnManualPaste').hidden = false;
  if (state.pythonBackend.available) {
    const blob = await composeFullBlob('image/jpeg', 0.85);
    const { blob: small } = await downscaleToMaxEdge(blob, 2048);
    try {
      const geo = await fetchPythonGeometry(small);
      // Build a minimal state.aiData with just the geometry — fields that follow stay empty
      state.aiData = {
        description: '', medium: '', style: '', tags: [],
        brightness_adjust: 0, contrast_adjust: 0, saturation_adjust: 0,
        needs_perspective_correction: !!geo.needs_perspective_correction,
        perspective_corners: geo.perspective_corners || null,
        needs_crop: !!geo.needs_crop,
        crop: geo.crop || null,
        quality_notes: geo.reason || '',
      };
      state.aiDataDims = { w: state.fullCanvas.width, h: state.fullCanvas.height };
    } catch (e) {
      console.warn('Python geometry failed:', e);
    }
  }
  return;
}
```

- [ ] **Step 4: Smoke test in browser**

With backend running and Gemini key set: upload a photo of a painting taken at an angle. Click Auto-correct. Painting should straighten using the Python pipeline's Hough corners. The AI card should still show Gemini's description.

With backend running, NO Gemini key (delete `pm_gemini_key` from localStorage and reload): upload the same photo. The AI card shows "No Gemini key — geometry via Python only." Auto-correct still straightens. Description/Medium/Tags are empty (Sveta fills manually).

With backend OFF: badge shows fallback, Auto-correct uses Gemini's approximate coords (less accurate, but at least works).

- [ ] **Step 5: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: Python pipeline geometry in parallel with Gemini text analysis"
```

---

## Task 6: Acceptance scenarios (auto-runnable via curl)

**Files:**
- Modify: `docs/photo-manager-setup.md` — add a backend-specific checklist section

- [ ] **Step 1: Add backend acceptance checklist**

Append to `docs/photo-manager-setup.md`:

```markdown
## Python backend acceptance

Subset of qa scenarios that can be checked via curl without a browser:

- [ ] `curl http://localhost:8001/health` → `{"ok":true}`
- [ ] `curl -X POST -F image=@output/processed/landscape/MorningCoffee.jpeg http://localhost:8001/analyze` returns valid JSON with `source:"python"`
- [ ] `curl -X POST -F image=@/etc/hosts http://localhost:8001/analyze` returns 400 with `error:"could not decode image"`
- [ ] `curl -X OPTIONS -H "Origin: http://localhost:8000" -i http://localhost:8001/analyze` returns 204 with `Access-Control-Allow-Origin: http://localhost:8000`
- [ ] Tiny image (100×100): `python -c "import cv2,numpy as np;cv2.imwrite('/tmp/tiny.jpg', np.zeros((100,100,3),'uint8'))"`, then POST it → returns `reason:"no_frame_detected"`, no exception

Browser-only checks (after curl checks pass):

- [ ] Backend running + Gemini key: upload angled painting photo → Auto-correct straightens precisely (Python wins over Gemini for geometry)
- [ ] Backend running + NO Gemini key: status says "Geometry via Python only", Auto-correct still works
- [ ] Backend OFF: badge shows fallback within 500ms of page load; Auto-correct uses Gemini coords
- [ ] Backend dies mid-session: next upload's analyze fails silently, badge flips to fallback

```

- [ ] **Step 2: Run the curl checks**

```bash
./scripts/dev.sh &
DEV_PID=$!
sleep 2

# Run each curl from the checklist; expected outputs in comments
curl -s http://localhost:8001/health
echo
TEST=$(find tools/image_pipeline/output -name "*.jpeg" 2>/dev/null | head -1)
[ -n "$TEST" ] && curl -s -X POST -F "image=@$TEST" http://localhost:8001/analyze | python3 -m json.tool

# Bad input
curl -s -X POST -F "image=@/etc/hosts" http://localhost:8001/analyze

# CORS preflight
curl -s -X OPTIONS -H "Origin: http://localhost:8000" -H "Access-Control-Request-Method: POST" -i http://localhost:8001/analyze | head -5

kill $DEV_PID 2>/dev/null
lsof -ti:8000 -ti:8001 2>/dev/null | xargs kill 2>/dev/null || true
```

Expected: every step succeeds with the documented behaviour.

- [ ] **Step 3: Commit**

```bash
git add docs/photo-manager-setup.md
git commit -m "docs: backend acceptance checklist + curl smoke-test recipes"
```

---

## Acceptance criteria

1. **`/health` returns 200 within 500ms** from `http://localhost:8001`.
2. **`/analyze` on a known-angle test painting returns valid `perspective_corners` fractions** matching the Hough output (deterministic — must match the existing Python pipeline's behavior bit-for-bit, since both call the same detector).
3. **Browser badge accurately reflects backend state** (Python ✓ when running, fallback when down or timing out).
4. **`state.aiData` after parallel analyze has Python corners + Gemini text** (verifiable via DevTools console: `state.aiData.perspective_corners` should differ from Gemini's by at most decimal noise).
5. **No regressions on the photo manager rewrite acceptance checklist** — full-res Drive uploads, formula-injection blocked, dedup, EXIF, OAuth cancel, auto-correct idempotency, keyboard.
6. **CORS works only from `http://localhost:8000`** — `Access-Control-Allow-Origin: http://localhost:8000` set; other origins blocked at browser layer.

## Out of scope

- Authentication on the Python backend (it's loopback-only — `127.0.0.1`).
- Concurrency beyond `threaded=True` Flask default.
- Re-analysis after manual edits (user can Reset + upload).
- Re-polling `/health` periodically (single check on page load + reactive flip on /analyze failure).
- Exposing Python's orientation detection (browser already handles EXIF).
- Refactoring `run_pipeline.py` — `geometry.py` is a new file that doesn't disturb the CLI flow.
