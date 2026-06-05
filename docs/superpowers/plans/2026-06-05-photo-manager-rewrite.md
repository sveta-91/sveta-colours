# Photo Manager Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `tools/photo_manager.html` so a 30-MP painting upload survives end-to-end: original-resolution Drive file, correct Sheets row, no formula injection, working AI auto-fill, served from `http://localhost` with editorial styles aligned to `index.html`.

**Architecture:** Single-file HTML/CSS/vanilla-JS, served via `python3 -m http.server`. Two canvases: an **off-screen full-resolution canvas** as source of truth, a **visible preview canvas** auto-fit to viewport. All exports, watermarks, and transforms read from the full-res canvas. Coordinates from AI are stored as normalized fractions (0–1) and applied to the current full-res canvas at apply-time. AI is Gemini 2.0 Flash via free-tier API key, with a manual-paste-into-ChatGPT.com escape hatch when offline or over quota.

**Tech Stack:** HTML5 + CSS3 + vanilla JS, Canvas 2D (no WebGL), Google Identity Services (GIS) for OAuth, Drive v3 + Sheets v4, Google AI Studio Gemini 2.0 Flash. No build step, no test framework.

**Source spec:** `docs/superpowers/specs/audit-summary.md` (consolidated from 5-agent audit, Rounds 1+2).

---

## File structure

This rewrite keeps `tools/photo_manager.html` as a single file (consistent with the rest of `tools/`, no build step needed). Internal organization changes:

```
tools/photo_manager.html         — single page; sections clearly delimited
docs/photo-manager-setup.md      — NEW: one-time Cloud Console + http.server setup
README.md                        — UPDATE: add a Photo Manager section pointing at setup doc
```

Internal sections of `photo_manager.html` (top-to-bottom, with section banners):

1. `<head>` — meta, title, lang, defined `:root` token block (no more `var(--color-text-primary)` ghosts), preconnect to fonts.
2. CSS — editorial palette ported from `index.html`, layout grid with `min()` clamps, focus-visible rules, `.sr-only`, dark-mode no-op (sRGB only).
3. `<body>` — semantic structure (`<main>`, `<form>`, `<fieldset>`, labels), keyboard-accessible drop zone (`<button>` wrapping `<input type=file>`), `aria-live` for AI/save status.
4. JS — modular sections delimited by banners:
   - State (single `state` object, no scattered globals)
   - Image I/O (load, EXIF orientation, blob/dataURL helpers)
   - Canvas pair (full-res offscreen + preview)
   - Transforms (crop, perspective via CPU homography, adjustments, resize, watermark)
   - AI (Gemini + manual-paste fallback, sanitization, fraction storage)
   - Google (OAuth, Drive upload from full-res, Sheets RAW writes, dedup, in-flight lock)
   - UI wiring (event listeners, status announcements)

---

## Preconditions (one-time, Sveta does manually)

These don't fit into a TDD task — they're external setup. Document in `docs/photo-manager-setup.md` and link from README.

1. **Google Cloud Console:**
   - Project: Sveta's existing project (the one that issued `CLIENT_ID` ending `…emi8qt`)
   - OAuth consent screen: **External**, **Testing** mode, add Sveta's email as a test user
   - OAuth client: **Web application** type, authorized JS origins `http://localhost:8000` and `http://127.0.0.1:8000`
   - Enable: Google Drive API, Google Sheets API
   - Scopes: `drive.file`, `spreadsheets`
2. **Google AI Studio (Gemini):**
   - Get a free API key from `https://aistudio.google.com/apikey`
   - Free tier: 1500 req/day, 15 RPM, more than enough for one painter
   - Sveta pastes the key into the photo manager's settings drawer on first run; stored in `localStorage` under `pm_gemini_key`
3. **Serving:**
   - `cd /Users/sveta.mordkovich/private/sveta-colours && python3 -m http.server 8000`
   - Open `http://localhost:8000/tools/photo_manager.html`

---

## Testing strategy

No JS test framework in this repo. Verification is **manual smoke-tests in a real browser**, codified as numbered scenarios in `docs/photo-manager-setup.md` under "Acceptance checklist." Each task lists the scenarios it must keep passing. The full checklist (Task 12) is run before commit-to-main.

For the two pure-math units that warrant isolated testing — `computeHomography()` and the EXIF orientation parser — extract them into a script-tag-loadable `<script type="module">` block and validate with `console.assert` snippets that Sveta can run via DevTools (kept inline in the file, behind a `?test=1` query string). YAGNI on jest/vitest.

---

## Task 1: Project setup doc + serving

**Files:**
- Create: `docs/photo-manager-setup.md`
- Modify: `README.md` (add a Photo Manager section)

- [ ] **Step 1: Write `docs/photo-manager-setup.md`**

```markdown
# Photo Manager — Setup

## One-time Google setup

### OAuth (Drive + Sheets)
1. Go to https://console.cloud.google.com → select the project that issued the OAuth client `754552250731-hdl67ujna2lk1lataah35mnl69emi8qt`
2. **OAuth consent screen** → External, Testing. Add `Sveta.Mordkovich@basis.com` as a test user. Add scopes `https://www.googleapis.com/auth/drive.file` and `https://www.googleapis.com/auth/spreadsheets`.
3. **Credentials** → edit the OAuth client → Authorized JavaScript origins: add `http://localhost:8000` and `http://127.0.0.1:8000`.
4. **APIs & Services** → enable **Google Drive API** and **Google Sheets API**.

### Gemini API key
1. Visit https://aistudio.google.com/apikey, sign in, create a key. Free tier is 1500 req/day, 15 RPM.
2. On first photo upload, the manager will prompt you to paste this key. It's saved in your browser only (localStorage).

## Running the page

```bash
cd /Users/sveta.mordkovich/private/sveta-colours
python3 -m http.server 8000
```

Open `http://localhost:8000/tools/photo_manager.html` (NOT `file://`).

## Acceptance checklist
- [ ] 30-MP painting upload → Drive file is full resolution, not a thumbnail
- [ ] Title `=IMPORTDATA("https://evil.example.com")` saves as a literal string in Sheets, no formula
- [ ] Reload page mid-flow → no stuck spinners, no orphan Drive file with no Sheets row
- [ ] OAuth popup closed by user → status shows "Sign-in cancelled," no infinite spinner
- [ ] iPhone JPEG with EXIF orientation 6 → preview shows it the right way up, Drive file is also correct
- [ ] AI offline (Gemini quota / no key) → manual-paste button works as fallback
- [ ] All form fields keyboard-reachable from the file input through Save
- [ ] Auto-correct twice in a row on the same upload doesn't double-crop
```

- [ ] **Step 2: Update README.md**

Add this section (place wherever fits the existing top-level layout):

```markdown
## Photo Manager (`tools/photo_manager.html`)

Local-only HTML page for processing an artwork photo end-to-end:
upload → AI-suggested crop/perspective/adjustments → manual touch-up → watermark
→ Drive folder + Sheets row. Served via `python3 -m http.server` from the repo root.

See `docs/photo-manager-setup.md` for one-time Google setup and the acceptance checklist.
```

- [ ] **Step 3: Verify the doc renders**

Open `docs/photo-manager-setup.md` in a markdown previewer (or VS Code preview). Confirm formatting, no broken anchors.

- [ ] **Step 4: Commit**

```bash
git add docs/photo-manager-setup.md README.md
git commit -m "docs: photo manager setup + acceptance checklist"
```

---

## Task 2: Define CSS token block aligned with index.html

**Files:**
- Modify: `tools/photo_manager.html:5-62` (the `<style>` block)

- [ ] **Step 1: Replace the top of the `<style>` block**

The current style starts directly with `*{box-sizing:border-box;...}` and references undefined `var(--color-text-primary)` etc. throughout. Prepend a `:root` token block ported from `index.html`, then rewrite the existing rules to use the new tokens. Keep the layout grid but rename variable references.

Replace lines 5-62 with:

```html
<style>
:root {
  /* Editorial bone palette — aligned with index.html */
  --bg: #EAE6DE;
  --surface: #F2EFE8;
  --surface-alt: #E0DBD1;
  --paper: #F7F5EF;
  --border: #D6CFC1;
  --border-strong: #BFB6A4;
  --hairline: rgba(20, 19, 15, 0.12);

  --text: #14130F;
  --text-secondary: #3B362D;
  --text-muted: #6B6357;
  --text-faint: #6B6258;

  --accent: #1A2640;
  --accent-hover: #2A3A5C;
  --success: #1F5132;
  --danger: #7A1F1F;

  --font-display: 'Fraunces', Georgia, 'Times New Roman', serif;
  --font-body: 'Inter', system-ui, -apple-system, sans-serif;

  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --shadow-1: 0 1px 2px var(--hairline);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body { background: var(--bg); color: var(--text); font-family: var(--font-body); }
body { padding: 1.5rem; line-height: 1.5; }

h1, h2, h3 { font-family: var(--font-display); font-weight: 500; color: var(--text); }
h1 { font-size: 1.75rem; margin-bottom: 1rem; }

.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; }

.top-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 1.5rem; align-items: start; }
.col-right { display: flex; flex-direction: column; gap: 1rem; }
.bottom-layout { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem; }

@media (max-width: 800px) {
  .top-layout, .bottom-layout { grid-template-columns: 1fr; }
}

.card { background: var(--paper); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.25rem; }
.card-title { font-family: var(--font-display); font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 0.75rem; }

.drop-zone { border: 2px dashed var(--border-strong); border-radius: var(--radius-md); padding: 3rem 1rem; text-align: center; cursor: pointer; background: var(--surface); transition: background 0.15s, border-color 0.15s; display: flex; flex-direction: column; align-items: center; gap: 0.5rem; }
.drop-zone:hover, .drop-zone:focus-visible, .drop-zone.drag-over { background: var(--surface-alt); border-color: var(--accent); outline: none; }
.drop-zone:focus-visible { box-shadow: 0 0 0 3px rgba(26,38,64,0.25); }
.drop-zone p { color: var(--text-secondary); }
.drop-zone small { color: var(--text-faint); font-size: 0.85em; }

input[type=file] { position: absolute; left: -9999px; }

.canvas-area { position: relative; background: var(--surface); border-radius: var(--radius-md); overflow: hidden; min-height: 180px; display: flex; align-items: center; justify-content: center; }
#preview-canvas { max-width: 100%; display: block; border-radius: var(--radius-md); }
#crop-overlay, #persp-overlay { position: absolute; top: 0; left: 0; pointer-events: none; border-radius: var(--radius-md); }

button { cursor: pointer; font-family: var(--font-body); font-size: 0.9rem; padding: 0.5rem 0.95rem; border-radius: var(--radius-md); border: 1px solid var(--border); background: var(--paper); color: var(--text); display: inline-flex; align-items: center; gap: 0.35rem; transition: background 0.12s, border-color 0.12s; }
button:hover { background: var(--surface-alt); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
button.primary { background: var(--accent); color: var(--paper); border-color: var(--accent); }
button.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
button:disabled { opacity: 0.4; cursor: not-allowed; }

input[type=text], input[type=number], select, textarea { font-family: var(--font-body); font-size: 0.9rem; padding: 0.45rem 0.65rem; border: 1px solid var(--border); border-radius: var(--radius-md); background: var(--paper); color: var(--text); width: 100%; }
input:focus-visible, select:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; border-color: var(--accent); }

.field-row { display: flex; gap: 0.5rem; align-items: flex-start; margin-bottom: 0.6rem; }
.field-row label { font-size: 0.85rem; color: var(--text-secondary); min-width: 80px; padding-top: 0.55rem; flex-shrink: 0; }

.status { font-size: 0.85rem; color: var(--text-faint); margin-top: 0.4rem; min-height: 1.2em; }
.status.ok { color: var(--success); }
.status.err { color: var(--danger); }

.spinner { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

hr.divider { border: 0; border-top: 1px solid var(--border); margin: 0.75rem 0; }
</style>
```

- [ ] **Step 2: Verify by serving + visual check**

Start the server: `python3 -m http.server 8000` from repo root.
Open `http://localhost:8000/tools/photo_manager.html`.
**Expected:** page renders with bone background, ink-blue accents, Fraunces card titles, Inter body text. No more raw-HTML appearance.

- [ ] **Step 3: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: define :root token block aligned with index.html palette"
```

---

## Task 3: Semantic HTML structure + keyboard a11y

**Files:**
- Modify: `tools/photo_manager.html` (`<head>` and `<body>`)

- [ ] **Step 1: Set `<html lang>` and add `<title>`**

In the `<html>` tag, set `lang="en"`. In `<head>`, add:

```html
<title>Sveta Colours — Photo Manager</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
```

- [ ] **Step 2: Wrap content in `<main>` and proper form structure**

Replace the existing `<body>` opening through the closing of `.bottom-layout` with this structure (preserving all the existing input/canvas/button IDs so JS still works):

```html
<body>
  <h1 class="sr-only">Photo Manager</h1>

  <main class="top-layout">
    <section class="col-left" aria-labelledby="canvas-h">
      <div class="card">
        <h2 id="canvas-h" class="card-title">Canvas</h2>

        <label id="dropZone" class="drop-zone" for="fileInput" tabindex="0" role="button"
               aria-label="Upload an image: drag a file here or press Enter to browse">
          <i class="ti ti-upload" aria-hidden="true" style="font-size:36px;color:var(--text-faint)"></i>
          <p>Drag &amp; drop or press Enter to upload</p>
          <small>JPG, PNG, WEBP, HEIC — any size</small>
        </label>
        <input type="file" id="fileInput" accept="image/*,.heic,.heif">

        <div id="workArea" hidden>
          <div class="tool-bar" role="toolbar" aria-label="Image tools">
            <!-- existing buttons preserved verbatim — only IDs/onclick stay -->
            <button id="btnAutoCorrect" class="primary" type="button" onclick="autoCorrectAll()">Auto-correct</button>
            <button id="btnCrop" type="button" onclick="toggleCrop()">Crop</button>
            <button id="btnApplyCrop" type="button" onclick="applyCrop()" hidden class="primary">Apply crop</button>
            <button id="btnCancelCrop" type="button" onclick="cancelCrop()" hidden>Cancel</button>
            <button id="btnPersp" type="button" onclick="togglePersp()">Perspective</button>
            <button id="btnAutoPersp" type="button" onclick="autoPersp()" hidden>Auto-detect</button>
            <button id="btnApplyPersp" type="button" onclick="applyPersp()" hidden class="primary">Apply</button>
            <button id="btnCancelPersp" type="button" onclick="cancelPersp()" hidden>Cancel</button>
            <button type="button" onclick="resetToOriginal()">Reset</button>
          </div>
          <div class="auto-steps" id="autoSteps" aria-live="polite" hidden></div>
          <div class="canvas-area" id="canvasArea">
            <canvas id="preview-canvas" aria-label="Image preview"></canvas>
            <canvas id="crop-overlay"></canvas>
            <canvas id="persp-overlay"></canvas>
          </div>
          <div class="status" id="dimInfo" aria-live="polite"></div>
        </div>
      </div>
    </section>

    <aside class="col-right" aria-label="Controls">
      <!-- Resize / Adjustments / Export cards: keep existing inner markup but wrap labels with <label for=...> instead of stand-alone <label> tags -->
      <!-- Resize -->
      <div class="card">
        <h2 class="card-title">Resize</h2>
        <div class="field-row">
          <label for="resizeW">Width</label>
          <input id="resizeW" type="number" min="1" max="8000" placeholder="px">
        </div>
        <div class="field-row">
          <label for="resizeH">Height</label>
          <input id="resizeH" type="number" min="1" max="8000" placeholder="px">
        </div>
        <div class="field-row">
          <label for="lockAR">Lock ratio</label>
          <input id="lockAR" type="checkbox" checked>
          <button type="button" onclick="applyResize()" class="primary">Apply</button>
        </div>
      </div>

      <!-- Adjustments (brightness/contrast/saturation/watermark) — keep IDs, wrap labels -->
      <!-- See lines existing lines 113-128 in the current file; convert to <label for=...> pattern -->

      <!-- Export -->
      <div class="card">
        <h2 class="card-title">Export</h2>
        <div class="field-row">
          <label for="exportFormat">Format</label>
          <select id="exportFormat">
            <option value="jpeg" selected>JPEG</option>
            <option value="png">PNG</option>
            <option value="webp">WEBP</option>
          </select>
        </div>
        <div class="field-row" id="qualityRow">
          <label for="jpegQuality">Quality</label>
          <input id="jpegQuality" type="range" min="10" max="100" value="90">
          <span id="qv" aria-live="off">90%</span>
        </div>
        <div class="btn-row">
          <button type="button" onclick="downloadImg()">Download</button>
          <button type="button" class="primary" onclick="saveAll()" id="btnSaveAll">Save to Drive + Sheets</button>
        </div>
        <div class="status" id="saveStatus" aria-live="polite" role="status"></div>
      </div>
    </aside>
  </main>

  <section class="bottom-layout" id="bottomLayout" hidden>
    <div class="card">
      <h2 class="card-title">AI analysis</h2>
      <div class="status" id="aiBox" aria-live="polite">Awaiting upload…</div>
      <button id="btnManualPaste" type="button" hidden>Use ChatGPT manually instead</button>
    </div>

    <form class="card" id="portfolioForm" onsubmit="event.preventDefault()">
      <h2 class="card-title">Portfolio record</h2>
      <fieldset style="border:0;padding:0">
        <legend class="sr-only">Painting metadata</legend>
        <div class="field-row"><label for="fTitle">Title *</label><input id="fTitle" type="text" required></div>
        <div class="field-row"><label for="fDesc">Description</label><textarea id="fDesc" rows="3"></textarea></div>
        <div class="field-row"><label for="fMedium">Medium</label><input id="fMedium" type="text" placeholder="oil on canvas"></div>
        <div class="field-row"><label for="fSize">Size (cm)</label><input id="fSize" type="text" placeholder="40×50"></div>
        <div class="field-row"><label for="fYear">Year</label><input id="fYear" type="text"></div>
        <div class="field-row"><label for="fTags">Tags</label><input id="fTags" type="text" placeholder="comma-separated"></div>
        <div class="field-row"><label for="fStatus">Status</label>
          <select id="fStatus">
            <option value="">—</option>
            <option value="available">Available</option>
            <option value="sold">Sold</option>
            <option value="reserved">Reserved</option>
            <option value="not for sale">Not for sale</option>
          </select>
        </div>
        <div class="field-row"><label for="fPrice">Price</label><input id="fPrice" type="text" placeholder="350 CAD"></div>
        <div class="field-row"><label for="fNotes">Notes</label><input id="fNotes" type="text"></div>
      </fieldset>
    </form>
  </section>
```

- [ ] **Step 3: Wire keyboard activation on the drop zone**

In the JS section, where `dz.addEventListener('click',...)` lives, add:

```js
dz.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fi.click();
  }
});
```

- [ ] **Step 4: Manual smoke test**

Reload the page. Tab through: drop zone → (upload an image) → Auto-correct → Crop → Perspective → ... → all the way to Save. Every interactive element should be reachable and have a visible focus ring.

- [ ] **Step 5: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: semantic structure, label/for, keyboard drop zone, aria-live"
```

---

## Task 4: Full-resolution off-screen canvas as source of truth

**Why this exists:** The current `setupCanvas` (line ~248) caps the *only* canvas at ~460px. Downloads and Drive uploads both read from that capped canvas, so every saved file is a thumbnail. Image-processing R1 identified this as the #1 blocker; it shows up in Drive uploads, downloads, and watermark sizing.

**Approach:** Introduce a second canvas (`fullCanvas`) held off-screen. `workingImg` is replaced by a function `getFullBlob()` that always returns from this canvas. The visible canvas is purely a preview, recomputed on every transform via `drawPreview()`.

**Files:**
- Modify: `tools/photo_manager.html` JS section (state, `setupCanvas`, `applyAll`, `applyCrop`, `applyPersp`, `applyResize`, `downloadImg`, save path)

- [ ] **Step 1: Introduce the offscreen canvas in state**

Replace the existing state declarations near the top of the script (currently `let origImg=null,workingImg=null,...`) with:

```js
const state = {
  origImg: null,           // original Image (untouched, post-EXIF-orientation)
  fullCanvas: document.createElement('canvas'),  // FULL RESOLUTION — source of truth
  aiData: null,
  aiDataDims: null,        // { w, h } of the image AI was called with — for fraction-to-pixel conversion
  cropMode: false, cropStart: null, cropRect: null,
  perspMode: false, perspPoints: [], dragIdx: -1,
  cachedToken: null,
  cachedTokenExpiresAt: 0, // ms epoch
  saveInFlight: false,
  geminiKey: localStorage.getItem('pm_gemini_key') || null,
};
const fullCtx = state.fullCanvas.getContext('2d', { willReadFrequently: false });
```

- [ ] **Step 2: Replace `setupCanvas` so it initializes both canvases**

```js
function setupCanvas(img) {
  // 1) Seed the full-res canvas from the original image (already EXIF-corrected by loadImageWithExif)
  state.fullCanvas.width = img.width;
  state.fullCanvas.height = img.height;
  fullCtx.drawImage(img, 0, 0);

  // 2) Size the visible preview canvas to fit the viewport (max 460 tall, max area width)
  fitPreviewToFull();
  syncOverlayCanvases();
  updateDimInfo();
}

function fitPreviewToFull() {
  const area = document.getElementById('canvasArea');
  const maxW = area.clientWidth || 600, maxH = 460;
  const scale = Math.min(maxW / state.fullCanvas.width, maxH / state.fullCanvas.height, 1);
  canvas.width = Math.round(state.fullCanvas.width * scale);
  canvas.height = Math.round(state.fullCanvas.height * scale);
  drawPreview();
}

function syncOverlayCanvases() {
  for (const ov of [cropOv, perspOv]) {
    ov.width = canvas.width;
    ov.height = canvas.height;
    ov.style.width = canvas.width + 'px';
    ov.style.height = canvas.height + 'px';
  }
}

function updateDimInfo() {
  document.getElementById('dimInfo').textContent =
    `${state.fullCanvas.width} × ${state.fullCanvas.height} px (preview ${canvas.width}×${canvas.height})`;
}
```

- [ ] **Step 3: `drawPreview` (replaces what `applyAll` was doing on the visible canvas)**

```js
function drawPreview() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.filter = buildFilter();
  ctx.drawImage(state.fullCanvas, 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none';
  drawWatermarkOnContext(ctx, canvas.width, canvas.height);
}

function applyAll() {
  if (!state.fullCanvas.width) return;
  drawPreview();
}
```

- [ ] **Step 4: Composite-on-export helper (the function exports MUST call)**

```js
// Returns a Blob of the full-res image with all filters + watermark applied.
async function composeFullBlob(format = 'image/jpeg', quality = 0.9) {
  const out = document.createElement('canvas');
  out.width = state.fullCanvas.width;
  out.height = state.fullCanvas.height;
  const octx = out.getContext('2d');
  octx.filter = buildFilter();
  octx.drawImage(state.fullCanvas, 0, 0);
  octx.filter = 'none';
  drawWatermarkOnContext(octx, out.width, out.height);
  return await new Promise(r => out.toBlob(r, format, quality));
}
```

- [ ] **Step 5: Update `drawWatermark` to be context-agnostic**

```js
function drawWatermarkOnContext(c, w, h) {
  const text = document.getElementById('wmText').value.trim();
  if (!text) return;
  const op = +document.getElementById('wmOpacity').value / 100;
  const fs = Math.max(12, Math.round(w / 24));
  c.save();
  c.font = `${fs}px sans-serif`;
  c.fillStyle = `rgba(255,255,255,${op})`;
  c.strokeStyle = `rgba(0,0,0,${op * 0.5})`;
  c.lineWidth = Math.max(1, fs / 16);
  c.textAlign = 'right';
  c.textBaseline = 'bottom';
  const x = w - Math.round(w / 100);
  const y = h - Math.round(h / 100);
  c.strokeText(text, x, y);
  c.fillText(text, x, y);
  c.restore();
}
```

- [ ] **Step 6: Rewrite `downloadImg`**

```js
async function downloadImg() {
  const fmt = document.getElementById('exportFormat').value;
  const q = +document.getElementById('jpegQuality').value / 100;
  const mime = fmt === 'png' ? 'image/png' : fmt === 'webp' ? 'image/webp' : 'image/jpeg';
  const blob = await composeFullBlob(mime, q);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (document.getElementById('fTitle').value.trim() || 'untitled') + '.' + fmt;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
```

- [ ] **Step 7: Rewrite the Drive upload's blob source**

Anywhere the current code calls `canvas.toBlob(...)` for upload (around line 517 area), replace with `await composeFullBlob(...)`.

- [ ] **Step 8: Update Crop apply to operate on `fullCanvas`**

`applyCrop()` currently re-draws the visible canvas. It must instead crop `state.fullCanvas` and re-fit preview:

```js
function applyCrop() {
  if (!state.cropRect) return cancelCrop();
  const { x, y, w, h } = state.cropRect;             // pixels in preview-canvas coords
  const sx = canvas.width, sy = canvas.height;
  // Convert preview rect → full-res rect
  const fx = Math.round(x / sx * state.fullCanvas.width);
  const fy = Math.round(y / sy * state.fullCanvas.height);
  const fw = Math.round(w / sx * state.fullCanvas.width);
  const fh = Math.round(h / sy * state.fullCanvas.height);

  const next = document.createElement('canvas');
  next.width = fw; next.height = fh;
  next.getContext('2d').drawImage(state.fullCanvas, fx, fy, fw, fh, 0, 0, fw, fh);
  state.fullCanvas.width = fw; state.fullCanvas.height = fh;
  fullCtx.drawImage(next, 0, 0);

  state.cropMode = false; state.cropRect = null;
  toggleCropUI(false);
  fitPreviewToFull();
  syncOverlayCanvases();
  updateDimInfo();
}
```

(The `toggleCropUI(false)` helper just shows/hides the Apply/Cancel buttons — extract from the existing inline code.)

- [ ] **Step 9: Update Resize to operate on `fullCanvas`**

```js
function applyResize() {
  const w = +document.getElementById('resizeW').value;
  const h = +document.getElementById('resizeH').value;
  if (!(w > 0 && h > 0 && w <= 8000 && h <= 8000)) {
    setStatus('saveStatus', 'Resize: width and height must be 1–8000', 'err');
    return;
  }
  const next = document.createElement('canvas');
  next.width = w; next.height = h;
  next.getContext('2d').drawImage(state.fullCanvas, 0, 0, w, h);
  state.fullCanvas.width = w; state.fullCanvas.height = h;
  fullCtx.drawImage(next, 0, 0);
  fitPreviewToFull();
  syncOverlayCanvases();
  updateDimInfo();
}
```

- [ ] **Step 10: Manual smoke test (the linchpin scenario)**

Upload a real 4000×3000 photo. Open DevTools → Application → IndexedDB / Console: in Console, run `state.fullCanvas.width, state.fullCanvas.height`. Confirm they match the original.
Click Download. Open the downloaded file → confirm dimensions match the original (not 600×460).
Run Drive save. In Drive, open the uploaded file → confirm full resolution.

- [ ] **Step 11: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: full-res offscreen canvas as source of truth; exports stop being thumbnails"
```

---

## Task 5: EXIF orientation on JPEG load

**Why:** iPhone JPEGs commonly arrive with EXIF orientation 6 (rotate 90° CW). Canvas ignores EXIF, so the preview shows it sideways. The fix is read EXIF once on load and bake the rotation into `state.fullCanvas`.

**Files:**
- Modify: `tools/photo_manager.html` JS section (`handleFile`, add `loadImageWithExif`)

- [ ] **Step 1: Add `readExifOrientation` helper**

```js
// Returns orientation 1–8, or 1 if missing/unparseable.
function readExifOrientation(arrayBuffer) {
  const v = new DataView(arrayBuffer);
  if (v.getUint16(0) !== 0xFFD8) return 1;       // not JPEG
  let off = 2;
  while (off < v.byteLength) {
    const marker = v.getUint16(off);
    off += 2;
    if (marker === 0xFFE1) {                       // APP1
      if (v.getUint32(off + 2) !== 0x45786966) return 1; // "Exif"
      const tiff = off + 8;
      const little = v.getUint16(tiff) === 0x4949;
      const get16 = o => v.getUint16(o, little);
      const get32 = o => v.getUint32(o, little);
      const ifd = tiff + get32(tiff + 4);
      const entries = get16(ifd);
      for (let i = 0; i < entries; i++) {
        const entry = ifd + 2 + i * 12;
        if (get16(entry) === 0x0112) return get16(entry + 8);
      }
      return 1;
    }
    if ((marker & 0xFF00) !== 0xFF00) return 1;
    off += v.getUint16(off);
  }
  return 1;
}
```

- [ ] **Step 2: Add `applyOrientationToCanvas(img, orientation) → canvas`**

```js
function applyOrientationToCanvas(img, o) {
  const c = document.createElement('canvas');
  const ctx2 = c.getContext('2d');
  const w = img.width, h = img.height;
  // Rotated outputs swap dimensions
  if (o >= 5 && o <= 8) { c.width = h; c.height = w; }
  else { c.width = w; c.height = h; }
  switch (o) {
    case 2: ctx2.translate(w, 0); ctx2.scale(-1, 1); break;
    case 3: ctx2.translate(w, h); ctx2.rotate(Math.PI); break;
    case 4: ctx2.translate(0, h); ctx2.scale(1, -1); break;
    case 5: ctx2.rotate(0.5 * Math.PI); ctx2.scale(1, -1); break;
    case 6: ctx2.rotate(0.5 * Math.PI); ctx2.translate(0, -h); break;
    case 7: ctx2.rotate(0.5 * Math.PI); ctx2.translate(w, -h); ctx2.scale(-1, 1); break;
    case 8: ctx2.rotate(-0.5 * Math.PI); ctx2.translate(-w, 0); break;
  }
  ctx2.drawImage(img, 0, 0);
  return c;
}
```

- [ ] **Step 3: Update `handleFile` to read EXIF before drawing**

Replace the existing `reader.onload` with:

```js
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/') && !/\.(heic|heif)$/i.test(file.name)) {
    setStatus('saveStatus', 'Only image files (JPG, PNG, WEBP, HEIC) supported.', 'err');
    return;
  }
  state.aiData = null; state.aiDataDims = null;

  const reader = new FileReader();
  reader.onload = async e => {
    const buf = e.target.result;
    const orientation = file.type === 'image/jpeg' ? readExifOrientation(buf) : 1;
    const blob = new Blob([buf], { type: file.type || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus('saveStatus', 'Could not decode image (corrupted or unsupported format).', 'err');
    };
    img.onload = () => {
      URL.revokeObjectURL(url);
      const oriented = applyOrientationToCanvas(img, orientation);
      // From here treat `oriented` as the source image — pass through everything that expected origImg.
      state.origImg = oriented;
      setupCanvasFromCanvas(oriented);
      // ...existing UI revealing + AI call...
      document.getElementById('resizeW').value = oriented.width;
      document.getElementById('resizeH').value = oriented.height;
      document.getElementById('fYear').value = new Date().getFullYear();
      dz.hidden = true;
      document.getElementById('workArea').hidden = false;
      document.getElementById('bottomLayout').hidden = false;
      const name = file.name.replace(/\.[^.]+$/, '');
      document.getElementById('fTitle').value = name;
      drawPreview();
      analyzeWithAI(file).catch(err => console.warn('AI:', err));
    };
    img.src = url;
  };
  reader.readAsArrayBuffer(file);
}

function setupCanvasFromCanvas(srcCanvas) {
  state.fullCanvas.width = srcCanvas.width;
  state.fullCanvas.height = srcCanvas.height;
  fullCtx.drawImage(srcCanvas, 0, 0);
  fitPreviewToFull();
  syncOverlayCanvases();
  updateDimInfo();
}
```

- [ ] **Step 4: Manual smoke test**

Need an iPhone JPEG with orientation tag 6 (most iPhone photos taken in portrait mode). Upload → preview should be upright. Download → image is upright in any viewer.

- [ ] **Step 5: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: read EXIF orientation, bake rotation into full-res canvas"
```

---

## Task 6: Replace WebGL affine with CPU homography

**Why:** The current perspective transform (around lines 440–461) uses WebGL to draw two textured triangles. That's an affine-per-triangle warp with a visible kink along the diagonal — not a true homography. Image-processing R1 confirmed straight lines bend.

**Approach:** Compute the 3×3 homography mapping the four user-picked points to a rectangle, then sample per pixel into the output canvas. For a 3000×4000 source this is ~12M pixels → ~150ms on M-series Mac. Acceptable for a single-painting workflow.

**Files:**
- Modify: `tools/photo_manager.html` JS section (`applyPersp` and the WebGL block)

- [ ] **Step 1: Write `computeHomography(srcPts, dstPts) → [9 numbers]`**

Standard DLT solver. Both args are arrays of `{x, y}` length 4.

```js
// Direct Linear Transform. Returns row-major 3x3.
function computeHomography(s, d) {
  const A = [];
  for (let i = 0; i < 4; i++) {
    const { x: X, y: Y } = s[i];
    const { x: u, y: v } = d[i];
    A.push([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u]);
    A.push([0, 0, 0, -X, -Y, -1, X * v, Y * v, v]);
  }
  // Solve A * h = 0 with last column treated as the constant side (assume h[8] = 1).
  // Standard trick: move last column to RHS and solve 8x8 system.
  const M = A.map(r => r.slice(0, 8));
  const b = A.map(r => -r[8]);
  const h = solveLinear(M, b);  // length 8
  return [...h, 1];
}

// Gaussian elimination on an 8x8 system.
function solveLinear(M, b) {
  const n = M.length;
  const A = M.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    // partial pivot
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r;
    [A[c], A[pivot]] = [A[pivot], A[c]];
    const piv = A[c][c];
    if (Math.abs(piv) < 1e-12) throw new Error('Degenerate homography');
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / piv;
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((row, i) => row[n] / row[i]);
}
```

- [ ] **Step 2: Write `applyHomographyToCanvas(srcCanvas, H_inv, outW, outH) → canvas`**

Bilinear sample per output pixel.

```js
function applyHomographyToCanvas(src, Hinv, outW, outH) {
  const srcCtx = src.getContext('2d');
  const srcData = srcCtx.getImageData(0, 0, src.width, src.height).data;
  const out = document.createElement('canvas');
  out.width = outW; out.height = outH;
  const outCtx = out.getContext('2d');
  const outData = outCtx.createImageData(outW, outH);
  const od = outData.data;
  const [a, b, c, d, e, f, g, h, i] = Hinv;

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const w = g * x + h * y + i;
      const sx = (a * x + b * y + c) / w;
      const sy = (d * x + e * y + f) / w;
      const idx = (y * outW + x) * 4;
      if (sx < 0 || sy < 0 || sx >= src.width - 1 || sy >= src.height - 1) {
        od[idx] = od[idx+1] = od[idx+2] = 0; od[idx+3] = 0; continue;
      }
      const x0 = sx | 0, y0 = sy | 0, dx = sx - x0, dy = sy - y0;
      const i00 = (y0 * src.width + x0) * 4;
      const i10 = i00 + 4;
      const i01 = i00 + src.width * 4;
      const i11 = i01 + 4;
      const wa = (1 - dx) * (1 - dy), wb = dx * (1 - dy), wc = (1 - dx) * dy, wd = dx * dy;
      od[idx]   = srcData[i00]   * wa + srcData[i10]   * wb + srcData[i01]   * wc + srcData[i11]   * wd;
      od[idx+1] = srcData[i00+1] * wa + srcData[i10+1] * wb + srcData[i01+1] * wc + srcData[i11+1] * wd;
      od[idx+2] = srcData[i00+2] * wa + srcData[i10+2] * wb + srcData[i01+2] * wc + srcData[i11+2] * wd;
      od[idx+3] = 255;
    }
  }
  outCtx.putImageData(outData, 0, 0);
  return out;
}
```

- [ ] **Step 3: Replace `applyPersp()`**

```js
function applyPersp() {
  if (state.perspPoints.length !== 4) return cancelPersp();
  // Points are in preview-canvas coords; scale to full-res
  const sxScale = state.fullCanvas.width / canvas.width;
  const syScale = state.fullCanvas.height / canvas.height;
  const srcPts = state.perspPoints.map(p => ({ x: p.x * sxScale, y: p.y * syScale }));

  // Output rectangle: bounding box of the source quad (preserves overall scale)
  const minX = Math.min(...srcPts.map(p => p.x));
  const maxX = Math.max(...srcPts.map(p => p.x));
  const minY = Math.min(...srcPts.map(p => p.y));
  const maxY = Math.max(...srcPts.map(p => p.y));
  const outW = Math.round(maxX - minX);
  const outH = Math.round(maxY - minY);

  // dst corners: rectangle 0,0 → outW,outH ordered tl,tr,br,bl (must match input order)
  const dstPts = [{x:0,y:0},{x:outW,y:0},{x:outW,y:outH},{x:0,y:outH}];

  // Forward H maps src → dst; we need the inverse for per-output-pixel sampling
  const H = computeHomography(srcPts, dstPts);
  const Hinv = computeHomography(dstPts, srcPts); // direct inverse via swap is fine for non-degenerate

  const warped = applyHomographyToCanvas(state.fullCanvas, Hinv, outW, outH);
  state.fullCanvas.width = warped.width;
  state.fullCanvas.height = warped.height;
  fullCtx.drawImage(warped, 0, 0);

  state.perspMode = false; state.perspPoints = [];
  togglePerspUI(false);
  fitPreviewToFull();
  syncOverlayCanvases();
  updateDimInfo();
}
```

- [ ] **Step 4: Delete the WebGL block**

The old `applyPersp` and any WebGL shader setup. Keep the overlay drawing (which is just 2D lines + handles).

- [ ] **Step 5: Self-test snippet (inline `?test=1`)**

In the script, near the end, add:

```js
if (new URLSearchParams(location.search).get('test') === '1') {
  // Sanity: identity homography
  const H = computeHomography(
    [{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}],
    [{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}]
  );
  console.assert(Math.abs(H[0]-1) < 1e-9, 'identity[0]');
  console.assert(Math.abs(H[4]-1) < 1e-9, 'identity[4]');
  console.assert(Math.abs(H[8]-1) < 1e-9, 'identity[8]');
  console.log('Homography self-test OK');
}
```

- [ ] **Step 6: Manual smoke test**

Open `http://localhost:8000/tools/photo_manager.html?test=1`, DevTools → Console → see `Homography self-test OK`.
Then upload a photo of a painting taken at an angle, run Perspective, place 4 corners on the painting frame, Apply. Painting should appear as a clean rectangle; straight edges in the painting (frame edges, signature line) should be straight, no diagonal kink.

- [ ] **Step 7: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: replace WebGL affine with CPU homography (DLT + bilinear)"
```

---

## Task 7: Sheets safety — RAW, atomic, dedup, token expiry

**Files:**
- Modify: `tools/photo_manager.html` JS section (lines around current 488–560: token caching, Sheets writes)

- [ ] **Step 1: Honor `expires_in` from token response**

Find the GIS token-client callback (around line 488 in the old code). Replace the hardcoded 58-min cache with:

```js
state.cachedToken = r.access_token;
state.cachedTokenExpiresAt = Date.now() + (r.expires_in - 30) * 1000; // 30s safety margin
```

In `getAccessToken`, check `Date.now() > state.cachedTokenExpiresAt` before reusing. Also handle the GIS error callback (popup closed by user) so the promise rejects instead of hanging:

```js
function getAccessToken() {
  return new Promise((resolve, reject) => {
    if (state.cachedToken && Date.now() < state.cachedTokenExpiresAt) return resolve(state.cachedToken);
    const client = google.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: 'https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/spreadsheets',
      callback: r => {
        if (r.error) return reject(new Error('OAuth: ' + r.error));
        state.cachedToken = r.access_token;
        state.cachedTokenExpiresAt = Date.now() + (r.expires_in - 30) * 1000;
        resolve(r.access_token);
      },
      error_callback: err => reject(new Error('OAuth cancelled: ' + (err && err.type)))
    });
    client.requestAccessToken();
  });
}
```

- [ ] **Step 2: In-flight save lock**

In `saveAll` (and anywhere Drive+Sheets are written together):

```js
async function saveAll() {
  if (state.saveInFlight) return;
  state.saveInFlight = true;
  const btn = document.getElementById('btnSaveAll');
  btn.disabled = true;
  setStatus('saveStatus', 'Saving…');
  try {
    const token = await getAccessToken();
    const driveLink = await uploadToDrive(token);  // composes full-res blob inside
    await upsertSheetRow(token, driveLink);
    setStatus('saveStatus', 'Saved.', 'ok');
  } catch (e) {
    setStatus('saveStatus', 'Save failed: ' + e.message, 'err');
  } finally {
    state.saveInFlight = false;
    btn.disabled = false;
  }
}
```

- [ ] **Step 3: Switch Sheets writes to `RAW` and normalize values**

Find every `valueInputOption=USER_ENTERED` (existing lines 547, 550, 553) and change to `RAW`.

Add a sanitizer used on every cell:

```js
// Sheets RAW already prevents formula execution, but normalize for dedup/display.
function sanitizeCell(s) {
  if (s == null) return '';
  return String(s).replace(/\r\n/g, '\n').trim();
}

function normalizeTitle(s) {
  return sanitizeCell(s).toLowerCase().replace(/\s+/g, ' ');
}
```

Apply `sanitizeCell` to every value pushed into a row, `normalizeTitle` to both sides of the dedup comparison.

- [ ] **Step 4: Fix the header-init race**

The current logic (line ~547) races: header write fires, then row append fires before header write completes. Refactor to a single ordered sequence:

```js
async function upsertSheetRow(token, driveLink) {
  // 1) Read existing A:B (Date + Title) to find dedup target
  const readRes = await fetch(
    `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${SHEET_TAB}!A:B`,
    { headers: { Authorization: 'Bearer ' + token } }
  );
  if (!readRes.ok) throw new Error('Sheets read failed: ' + readRes.status);
  const rows = (await readRes.json()).values || [];

  // 2) Ensure header row exists FIRST and is committed before any further write
  if (rows.length === 0 || normalizeTitle(rows[0][1] || '') !== 'title') {
    const headRes = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${SHEET_TAB}!A1?valueInputOption=RAW`,
      { method: 'PUT', headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: [COLS] }) }
    );
    if (!headRes.ok) throw new Error('Sheets header init failed: ' + headRes.status);
  }

  // 3) Compose row from form fields
  const title = document.getElementById('fTitle').value;
  const normTitle = normalizeTitle(title);
  const row = COLS.map(col => sanitizeCell(getCellForCol(col, driveLink)));

  // 4) Find existing row by normalized title (skip header)
  let foundIdx = -1;
  for (let i = 1; i < rows.length; i++) {
    if (rows[i] && normalizeTitle(rows[i][1] || '') === normTitle) { foundIdx = i; break; }
  }

  if (foundIdx >= 0) {
    const rowNum = foundIdx + 1;
    const range = `${SHEET_TAB}!A${rowNum}:${String.fromCharCode(64 + COLS.length)}${rowNum}`;
    const res = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${range}?valueInputOption=RAW`,
      { method: 'PUT', headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
        body: JSON.stringify({ range, values: [row] }) }
    );
    if (!res.ok) throw new Error('Sheets update failed: ' + res.status);
  } else {
    const res = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${SHEET_ID}/values/${SHEET_TAB}!A1:append?valueInputOption=RAW`,
      { method: 'POST', headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: [row] }) }
    );
    if (!res.ok) throw new Error('Sheets append failed: ' + res.status);
  }
}

function getCellForCol(col, driveLink) {
  const m = {
    'Date': new Date().toISOString().slice(0, 10),
    'Title': document.getElementById('fTitle').value,
    'Description': document.getElementById('fDesc').value,
    'Medium': document.getElementById('fMedium').value,
    'Size': document.getElementById('fSize').value,
    'Year': document.getElementById('fYear').value,
    'Tags': document.getElementById('fTags').value,
    'Status': document.getElementById('fStatus').value,
    'Price': document.getElementById('fPrice').value,
    'Notes': document.getElementById('fNotes').value,
    'Dimensions': `${state.fullCanvas.width}×${state.fullCanvas.height}`,
    'Format': document.getElementById('exportFormat').value,
    'Drive link': driveLink,
  };
  return m[col] || '';
}
```

- [ ] **Step 5: Manual smoke test**

1. With an empty sheet, save → confirm headers in row 1, data in row 2 (not the other way around).
2. Use title `=IMPORTDATA("https://example.com")`. Open the sheet → cell B should show the literal text, NOT execute the formula.
3. Click Save twice fast. Confirm only one Drive file and one Sheets row.
4. Edit the title to match an existing row, save. Confirm the existing row is updated, not duplicated.

- [ ] **Step 6: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: Sheets RAW, atomic save lock, header-init order, dedup normalization, token expiry from response"
```

---

## Task 8: Drive upload from full-res canvas

**Files:**
- Modify: `tools/photo_manager.html` JS section (`uploadToDrive` — currently `saveToDrive`)

- [ ] **Step 1: Compose-on-export inside `uploadToDrive`**

```js
async function uploadToDrive(token) {
  const fmt = document.getElementById('exportFormat').value;
  const mime = fmt === 'png' ? 'image/png' : fmt === 'webp' ? 'image/webp' : 'image/jpeg';
  const q = +document.getElementById('jpegQuality').value / 100;
  const blob = await composeFullBlob(mime, q);

  const title = (document.getElementById('fTitle').value.trim() || 'untitled') + '.' + fmt;
  const meta = { name: title, parents: [DRIVE_FOLDER] };

  const form = new FormData();
  form.append('metadata', new Blob([JSON.stringify(meta)], { type: 'application/json' }));
  form.append('file', blob);

  const res = await fetch(
    'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink',
    { method: 'POST', headers: { Authorization: 'Bearer ' + token }, body: form }
  );
  if (!res.ok) throw new Error('Drive upload failed: ' + res.status);
  const j = await res.json();
  return j.webViewLink;
}
```

- [ ] **Step 2: Manual smoke test**

Upload a 4000×3000 painting. Save to Drive. In the Drive folder, open the file in Google Drive's image preview → confirm 4000×3000 in Properties. (The previous behavior was ~600×460.)

- [ ] **Step 3: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: Drive upload composes from full-res canvas (no more thumbnail uploads)"
```

---

## Task 9: Gemini integration + key prompt

**Files:**
- Modify: `tools/photo_manager.html` JS section (replace `analyzeWithClaude`)

- [ ] **Step 1: Add key prompt + storage**

```js
function ensureGeminiKey() {
  if (state.geminiKey) return state.geminiKey;
  const k = window.prompt(
    'Paste your Google AI Studio (Gemini) API key.\n' +
    'Get one free at https://aistudio.google.com/apikey\n' +
    'Stored locally in this browser only.'
  );
  if (k && k.trim()) {
    state.geminiKey = k.trim();
    localStorage.setItem('pm_gemini_key', state.geminiKey);
    return state.geminiKey;
  }
  return null;
}
```

- [ ] **Step 2: Replace `analyzeWithClaude` with `analyzeWithAI`**

```js
async function analyzeWithAI(file) {
  const box = document.getElementById('aiBox');
  const key = ensureGeminiKey();
  if (!key) {
    box.textContent = 'No Gemini key — use "Use ChatGPT manually" below.';
    document.getElementById('btnManualPaste').hidden = false;
    return;
  }
  box.innerHTML = '<span class="spinner"></span> Analyzing with Gemini…';

  // Send the FULL-resolution image (or downscale to 2048px max edge to keep payloads sane)
  const blob = await composeFullBlob('image/jpeg', 0.85);
  const downscaled = await downscaleToMaxEdge(blob, 2048);
  const b64 = await blobToBase64(downscaled.blob);

  // Track what dimensions AI saw — fractions are normalized to this
  state.aiDataDims = { w: downscaled.w, h: downscaled.h };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${encodeURIComponent(key)}`;
  const body = {
    contents: [{
      parts: [
        { inline_data: { mime_type: 'image/jpeg', data: b64 } },
        { text: AI_PROMPT }
      ]
    }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: AI_SCHEMA
    }
  };

  let json;
  try {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const t = await res.text();
      if (res.status === 401 || res.status === 403) {
        localStorage.removeItem('pm_gemini_key');
        state.geminiKey = null;
        throw new Error('Gemini key invalid');
      }
      if (res.status === 429) throw new Error('Gemini rate-limited — try manual paste');
      throw new Error('Gemini ' + res.status + ': ' + t.slice(0, 120));
    }
    const j = await res.json();
    const text = j.candidates?.[0]?.content?.parts?.[0]?.text;
    if (!text) throw new Error('Gemini returned no content');
    json = JSON.parse(text);  // responseMimeType=application/json guarantees parseable
  } catch (e) {
    box.textContent = 'AI: ' + e.message;
    document.getElementById('btnManualPaste').hidden = false;
    return;
  }

  state.aiData = sanitizeAiData(json);
  renderAiBox(state.aiData);
  // Do NOT auto-apply transforms — user picks via Auto-correct
  // Do NOT auto-fill fields until user clears the auto-filled-from-filename Title
}

const AI_PROMPT = `Analyze this artwork photo. Reply ONLY with JSON matching the schema.
- crop and perspective_corners use fractions 0–1 of the IMAGE YOU ARE LOOKING AT.
- brightness/contrast/saturation are integers in -100..100.
- description: 2-3 sentences, neutral, art-catalog tone.`;

const AI_SCHEMA = {
  type: "object",
  properties: {
    description: { type: "string" },
    medium: { type: "string" },
    style: { type: "string" },
    tags: { type: "array", items: { type: "string" } },
    brightness_adjust: { type: "integer" },
    contrast_adjust: { type: "integer" },
    saturation_adjust: { type: "integer" },
    needs_perspective_correction: { type: "boolean" },
    perspective_corners: {
      type: "object",
      properties: {
        tl: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } } },
        tr: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } } },
        br: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } } },
        bl: { type: "object", properties: { x: { type: "number" }, y: { type: "number" } } },
      }
    },
    needs_crop: { type: "boolean" },
    crop: {
      type: "object",
      properties: {
        x: { type: "number" }, y: { type: "number" }, w: { type: "number" }, h: { type: "number" }
      }
    },
    quality_notes: { type: "string" }
  },
  required: ["description", "medium", "tags"]
};

// Strip control chars, cap lengths, drop leading =+-@ from any string field that will be sent to Sheets
function sanitizeAiData(d) {
  const cleanStr = s => typeof s === 'string'
    ? s.replace(/[ -]/g, '').slice(0, 1000).replace(/^[=+\-@]+/, '')
    : '';
  return {
    description: cleanStr(d.description),
    medium: cleanStr(d.medium),
    style: cleanStr(d.style),
    tags: Array.isArray(d.tags) ? d.tags.map(cleanStr).filter(Boolean).slice(0, 12) : [],
    brightness_adjust: clamp(Math.round(d.brightness_adjust || 0), -100, 100),
    contrast_adjust: clamp(Math.round(d.contrast_adjust || 0), -100, 100),
    saturation_adjust: clamp(Math.round(d.saturation_adjust || 0), -100, 100),
    needs_perspective_correction: !!d.needs_perspective_correction,
    perspective_corners: d.perspective_corners,
    needs_crop: !!d.needs_crop,
    crop: d.crop,
    quality_notes: cleanStr(d.quality_notes),
  };
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

async function downscaleToMaxEdge(blob, maxEdge) {
  const url = URL.createObjectURL(blob);
  const img = await new Promise((res, rej) => {
    const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url;
  });
  URL.revokeObjectURL(url);
  const s = Math.min(maxEdge / img.width, maxEdge / img.height, 1);
  const c = document.createElement('canvas');
  c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  const out = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.85));
  return { blob: out, w: c.width, h: c.height };
}

function blobToBase64(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(',')[1]);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

// Render — escape with textContent, never innerHTML
function renderAiBox(d) {
  const box = document.getElementById('aiBox');
  box.textContent = '';
  const desc = document.createElement('p'); desc.textContent = d.description || '';
  const meta = document.createElement('p');
  meta.style.color = 'var(--text-muted)';
  meta.style.marginTop = '0.5rem';
  meta.textContent = [d.medium, d.style].filter(Boolean).join(' · ');
  const tags = document.createElement('p');
  tags.style.color = 'var(--text-faint)';
  tags.style.marginTop = '0.25rem';
  tags.textContent = (d.tags || []).join(', ');
  box.append(desc, meta, tags);
}
```

- [ ] **Step 3: Apply AI coordinates *only* on Auto-correct, against current full-res dims**

In `autoCorrectAll`, when applying `perspective_corners` or `crop`, convert fractions against current `state.fullCanvas` dimensions (NOT against `state.aiDataDims` — fractions are normalized so they're interchangeable, but always re-derive from current canvas to handle the case where the user already cropped before clicking Auto-correct):

```js
function autoApplyCrop(c) {
  if (!c || !state.aiData?.needs_crop) return;
  const fx = Math.round(c.x * state.fullCanvas.width);
  const fy = Math.round(c.y * state.fullCanvas.height);
  const fw = Math.round(c.w * state.fullCanvas.width);
  const fh = Math.round(c.h * state.fullCanvas.height);
  // ... build the cropped canvas as in applyCrop()
}
```

- [ ] **Step 4: Manual smoke test**

Without a key: upload → AI box shows "No Gemini key" + manual-paste button visible.
With a key: upload → AI box populates with description + medium + tags, key persists across reloads.
Bad key: upload → key cleared from localStorage, status says "Gemini key invalid", manual-paste button shown.
Inject `=HYPERLINK(...)` into a real photo's metadata via Exiftool, upload → confirm sanitizer strips leading `=`.

- [ ] **Step 5: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: Gemini 2.0 Flash with structured output, key prompt, sanitization, normalized-fraction storage"
```

---

## Task 10: Manual-paste fallback button

**Files:**
- Modify: `tools/photo_manager.html` (wire `btnManualPaste`, add modal)

- [ ] **Step 1: Wire the button**

```js
document.getElementById('btnManualPaste').addEventListener('click', async () => {
  const blob = await composeFullBlob('image/jpeg', 0.85);
  const { blob: small } = await downscaleToMaxEdge(blob, 2048);
  // Download the downscaled image so Sveta can drag it into ChatGPT.com
  const a = document.createElement('a');
  a.href = URL.createObjectURL(small);
  a.download = 'for-chatgpt.jpg';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);

  // Copy the prompt to clipboard
  await navigator.clipboard.writeText(
    AI_PROMPT + '\n\nReturn the JSON only, no surrounding text.'
  );

  const json = window.prompt(
    'Image downloaded as for-chatgpt.jpg and prompt copied to clipboard.\n' +
    'In ChatGPT, drop the image, paste the prompt, copy the JSON reply, paste it here:'
  );
  if (!json) return;
  try {
    const parsed = JSON.parse(json);
    state.aiData = sanitizeAiData(parsed);
    state.aiDataDims = { w: state.fullCanvas.width, h: state.fullCanvas.height };
    renderAiBox(state.aiData);
  } catch (e) {
    setStatus('saveStatus', 'Could not parse JSON from ChatGPT', 'err');
  }
});
```

- [ ] **Step 2: Manual smoke test**

Hide your Gemini key (open DevTools → Application → Local Storage → delete `pm_gemini_key`). Upload a photo. Click "Use ChatGPT manually." Confirm the JPEG downloads and the prompt is on your clipboard. Paste into ChatGPT.com, get the JSON reply, paste it back, confirm fields populate.

- [ ] **Step 3: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: manual-paste fallback into ChatGPT.com for offline / quota-exhausted days"
```

---

## Task 11: Error handling pass

**Files:**
- Modify: `tools/photo_manager.html` JS

- [ ] **Step 1: `img.onerror` everywhere**

Any `new Image()` followed by setting `.src` must have an `.onerror` that surfaces a status. The new `handleFile` (Task 5) already has one; check the `?test=1` block and any download helpers.

- [ ] **Step 2: Don't clobber user input from AI fields**

In `onTitleInput` / wherever AI fills fields, never overwrite a field the user has typed in:

```js
function fillFields(d) {
  const fields = [
    ['fDesc',   d.description],
    ['fMedium', d.medium],
    ['fTags',   d.tags?.length ? d.tags.join(', ') : ''],
    ['fNotes',  d.style ? d.style : ''],
  ];
  for (const [id, val] of fields) {
    const el = document.getElementById(id);
    if (!el.value.trim() && val) el.value = val;   // only fill if currently empty
  }
}
```

- [ ] **Step 3: Auto-correct doesn't re-apply on already-applied AI**

Add a flag:

```js
state.aiApplied = { crop: false, persp: false, adjust: false };
```

Set each to `true` after its branch runs in `autoCorrectAll`. On re-click, skip the branches that are already `true` and show "Already applied — Reset first" status.

- [ ] **Step 4: Status helper**

```js
function setStatus(id, msg, kind) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.remove('ok', 'err');
  if (kind === 'ok') el.classList.add('ok');
  else if (kind === 'err') el.classList.add('err');
}
```

Use it everywhere status text is set (replace the existing scattered `.innerHTML = ...` lines).

- [ ] **Step 5: Manual smoke test**

- Drop a `.txt` file: status says "Only image files..." and the work area stays hidden.
- Drop a corrupt JPEG (truncate one with `head -c 100 foo.jpg > broken.jpg`): status says "Could not decode image..."
- Type a Description manually, then wait for AI to come back: your typed text stays, only Tags + Medium get filled.
- Click Auto-correct twice on the same upload: second click is a no-op with a status hint.
- Cancel the OAuth popup: status shows "OAuth cancelled," no infinite spinner.

- [ ] **Step 6: Commit**

```bash
git add tools/photo_manager.html
git commit -m "photo-manager: error-handling pass — onerror, AI doesn't clobber input, auto-correct idempotent, status helper"
```

---

## Task 12: Acceptance run + final commit

**Files:**
- Modify: `docs/photo-manager-setup.md` (tick the checklist if you want)

- [ ] **Step 1: Run the full acceptance checklist from Task 1, Step 1**

```
[ ] 30-MP painting upload → Drive file is full resolution, not a thumbnail
[ ] Title `=IMPORTDATA("...")` saves as literal string in Sheets
[ ] Reload page mid-flow → no stuck spinners, no orphan Drive file with no Sheets row
[ ] OAuth popup closed by user → status shows "Sign-in cancelled"
[ ] iPhone JPEG with EXIF orientation 6 → preview and Drive file are upright
[ ] AI offline (Gemini quota / no key) → manual-paste fallback works
[ ] All form fields keyboard-reachable
[ ] Auto-correct twice in a row doesn't double-crop
```

For each: take a screenshot of evidence, save into `docs/superpowers/specs/photo-manager-acceptance-evidence/`.

- [ ] **Step 2: Show the user evidence + ask for sign-off**

Show Sveta the screenshots and confirm everything passes.

- [ ] **Step 3: Final commit + tag**

```bash
git add docs/superpowers/specs/photo-manager-acceptance-evidence/
git commit -m "photo-manager: rewrite acceptance evidence — all blockers cleared"
```

---

## Out of scope (deferred / never)

- Matching the Python pipeline's pixel quality. Bar is Instagram + portfolio.
- ICC / P3 color preservation. Canvas is sRGB; iPhone P3 will shift. Documented, not fought.
- Mobile-first responsive layout. Single-column at <800px is enough.
- Build step / bundler / JS framework. Single file stays.
- Multi-user, multi-tab, real-time sync. Sveta + her browser, one tab.

## Acceptance criteria — must-pass before merge to main

1. **Full-resolution round-trip:** 30-MP painting upload → identical (or only watermark-different) pixel dimensions in the Drive file.
2. **No formula injection:** sheet cell containing `=IMPORTDATA(...)` from any source (title, AI description, manual paste) saves as literal text.
3. **Localhost only:** page loads cleanly at `http://localhost:8000/tools/photo_manager.html`; refuses to mislead Sveta if opened at `file://` (could add a `if (location.protocol === 'file:')` warning banner — nice-to-have).
4. **No dangling state on failure:** save fails midway → no orphan Drive file with no Sheets row, no stuck spinners.
5. **EXIF-correct preview:** iPhone portrait JPEG displays upright; saved file also upright.
6. **Idempotent auto-correct:** second click is a no-op with a hint, not a double-crop.
7. **Graceful AI degradation:** missing key, bad key, 429, network failure, malformed JSON all → status message + manual-paste button visible.
8. **Keyboard-complete flow:** every interactive control is reachable and visibly focusable.
