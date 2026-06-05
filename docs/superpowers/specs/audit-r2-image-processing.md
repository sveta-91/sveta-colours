# Image Processing Expert — Round 2 Critique

## Updated verdict
`rewrite` — holding firm. The other reports independently confirm the export-canvas bug (Google) and surface non-atomic save + AI clobber (QA) and the broken-token aesthetic (Frontend), which together rule out a single-pass "improve". But I'm narrowing the *scope* of the rewrite: the pixel-quality bar should be "Instagram-grade, ~3000px sRGB", not a port of the Python pipeline.

## Strongest points I agree with from the others
- **Google, issue #5 + risk register:** Drive uploads are not de-duplicated by name (line 519–523), so once the export bug is fixed, every re-save still produces an orphan. My fix ("export from a full-res offscreen canvas") is necessary but not sufficient without their `files?q=name=... in parents` PATCH pattern.
- **QA, issue #5 (non-atomic save):** Drive succeeds → Sheets fails → orphan file. This is a real failure mode that interacts with my "keep an offscreen full-res canvas" recommendation: that canvas must survive a failed save so retry doesn't re-encode (and re-watermark) from `workingImg` a second time.
- **AI manager, issue #5 (trust-but-verify on AI geometry):** clamp corners to [0,1], reject crops with area <25% or >99%, require convex quad. My Round 1 flagged "AI at 800px = ±15 source-pixel error"; their convexity/area gate is the right *runtime* response to that, not just a "send full-res" wish.
- **AI manager out-of-scope bullet about the 800px downscale (line 480 of source):** they're right that this *actively harms* perspective-corner accuracy. This is the cross-lane item below.
- **Frontend, issue #1 (undefined CSS tokens) + QA's note that watermark size is tied to preview canvas:** the watermark being sized to a 460px-tall canvas is the *same bug* as the export-resolution bug, viewed from a different angle. Fixing one fixes both.

## Points I push back on
- **QA's "MEDIUM | resize | negative values" and similar input-validation bugs** are real but trivial next to the data-flow rewrite needed. Treating them as separate fixes inflates the work; they all disappear if `workingImg` and `displayCanvas` are properly separated and every transform goes through one validated `applyTransform(srcBitmap, op) → newBitmap` path.
- **Frontend's "rewrite" verdict is partly aesthetic.** Re-skinning to match `index.html` is the right call but shouldn't gate the pixel-quality rewrite — they're independent work streams and could ship in either order. Don't conflate "doesn't match the editorial style" with "doesn't preserve pixels".
- **AI manager's primary recommendation (Gemini Flash) is correct, but I push back on the implicit assumption that AI-derived corners are good enough as the *only* perspective input.** Even Gemini at full resolution will be ±5–10 source pixels on a real painting edge. The browser still needs the manual-corner-drag refine step as the *final* input, with AI corners as a starting position — never as a fire-and-forget transform.
- **QA's "MEDIUM | re-running Auto-correct re-applies the same fractional crop"** — agreed it's a bug, but the fix isn't "remember if we cropped already". It's that `aiData.crop` should be *consumed* on apply (set to null) so it can't fire twice, and re-analysis should re-issue from the new `workingImg` state. This is part of the data-flow rewrite, not a patch.
- **My own Round 1 "fix the WebGL homography in the shader" stance is over-engineered for Sveta's use case.** See quality bar below — a 3000px-long-edge output warped on CPU via a 4×4 sample-grid (or even `OffscreenCanvas.drawImage` with the right transform decomposition) is fast enough and produces the right math. Don't ship a fragment-shader homography for a portfolio tool.

## Quality bar for v2
**Target output:** ~3000px long edge, sRGB JPEG q=0.9, EXIF orientation honored, ICC profile *not* required (sRGB is fine — Instagram strips P3 anyway, and the portfolio site is sRGB).

Match the Python pipeline where it's cheap:
- **Perspective warp must be a true homography.** Not because Sveta will pixel-peep, but because affine-per-triangle bends straight canvas edges visibly at 3000px. Use CPU homography via 4-corner solve + `OffscreenCanvas.drawImage` with a per-row transform, or a WebGL fragment-shader divide. Either is acceptable; no triangle-strip UV interpolation.
- **Export resolution must equal source resolution.** The "save the display canvas" bug is the one inviolable thing.
- **EXIF orientation must be applied** before any transform. iPhone HEIC will be the most common input.

Don't try to match where it's expensive:
- **Skip ICC/P3 awareness.** Canvas forces sRGB, sRGB is the target — accept the (small) gamut shift. Mention it once in a tooltip.
- **Skip LAB-space adjustments.** `ctx.filter` is fine if the slider range is `0.7–1.3` (not `0–2`) so the user can't crush the image. That's a 3-line fix, not a per-pixel ImageData loop.
- **Skip auto-detection of crop/perspective from CV.** That's what AI is for now. Don't reimplement Hough in JS.
- **Skip HEIC decoding via WASM.** If the browser can't decode, show a clear error and tell Sveta to export JPEG from Photos. Don't ship libheif-js.

## Coordinate-space contract (cross-lane with AI manager)
The AI returns normalized `[0,1]` coordinates against the *image it saw*, which is the 800px-downscaled JPEG. Two coordinate spaces matter, and the code currently conflates them:

1. **Source space** — `workingImg.naturalWidth × .naturalHeight` (or its current transformed dimensions after a crop/warp). This is the only space transforms ever apply in.
2. **Display space** — the visible `<canvas>` (~600×460). For pointer input and overlay rendering only. Never an input to a transform.

Contract:
- AI returns `{corners: [[x,y]×4], crop: {x,y,w,h}}` in `[0,1]` normalized against the image *sent to AI*. Send the AI the full-resolution image (or 2048px long edge at minimum) — the 800px downscale must go. Cost is the same order of magnitude.
- On receipt, validate (clamp to [0,1], convex-quad check, area sanity per AI manager #5), then store as **normalized fractions** in `aiData`. Never convert to pixels at this stage.
- On apply, convert to source-space pixels at the *moment of transform* using current `workingImg` dimensions: `srcX = corners[i][0] * workingImg.width`.
- The transform writes a new full-resolution `OffscreenCanvas`, which becomes the new `workingImg` (via `transferToImageBitmap()` — no dataURL roundtrip).
- The display canvas is re-derived from `workingImg` by a single `drawImage` with `imageSmoothingQuality='high'`. Watermark is drawn *only* on the display canvas for preview, and *separately* on the export canvas at export time, sized in source-space units (`fontSize = workingImg.width / 60`).
- On save failure (QA's non-atomic concern), `workingImg` is unchanged — no need to roll back. The full-res offscreen canvas is still in memory; retry re-encodes from it, not from a re-applied transform chain.

This makes the export path: `workingImg → offscreen canvas at source dims → draw watermark at source-scaled font → toBlob(jpeg, 0.9) → upload`. The display canvas is never on that path.
