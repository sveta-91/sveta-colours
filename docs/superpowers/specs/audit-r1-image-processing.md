# Image Processing Expert — Round 1 Audit

## Verdict

`rewrite` — the export path silently saves the **display-resolution** canvas (capped at ~460 px tall, line 250–252 + line 479), not the full-resolution `workingImg`; every "saved" painting is therefore a downscaled thumbnail with a watermark sized for that thumbnail. This single bug invalidates the whole pipeline for an artist who needs publishable images, and the surrounding processing (affine-per-triangle WebGL "perspective", `ctx.filter` in sRGB, no EXIF/ICC handling) is not at the quality bar Sveta is used to from the Python pipeline. The structure is fine; the math and the data flow are wrong.

## Top 3 strengths

1. **Manual crop preserves source resolution** (line 471). `applyCrop` correctly scales the overlay rectangle from canvas-space to `workingImg`-space (`sx=cropRect.x*workingImg.width/canvas.width`) and crops from the original `workingImg`, not the displayed canvas. So manual crop is the one pixel-correct operation in the file.
2. **WebGL is used at all for perspective warp** (lines 440–461). Doing the warp on GPU at the source dimensions, not on the preview, is the right architectural choice for large paintings — it avoids re-rasterising at preview resolution. The execution of the math is wrong (see Issue #2) but the framing is correct.
3. **Auto-correct pipeline order is sensible** (lines 278, 292–309). Perspective → crop → adjustments matches the Python pipeline's order and is the right sequence (perspective first so subsequent crop operates on a rectified image; adjustments last so brightness/contrast aren't baked into pixels the user may re-crop).

## Top 5 issues

1. **[BLOCKER] Export saves the displayed canvas, not the full-resolution image** (lines 248–254, 264–268, 479, 517).
   - `setupCanvas` sizes `canvas` to `min(area.clientWidth, 460, 1) × img scale` — typically ~600×460 px regardless of input.
   - `downloadImg` (line 479) and `uploadToDrive` (line 517) both call `canvas.toDataURL(...)` / `canvas.toBlob(...)` on that *display* canvas.
   - A 4000×3000 painting becomes a ~600×460 export. Sveta uploads thumbnails to Drive and writes "4000×3000" to the Sheet (`dims` on line 532 reads `workingImg.width/height`, so the metadata is honest but the file isn't).
   - **Fix:** keep a separate full-resolution offscreen canvas that mirrors every transform; export from that, never from the preview canvas. The preview canvas is for display only.

2. **[BLOCKER] WebGL perspective warp is affine-per-triangle, not a homography** (lines 440–461, 386, 437).
   - The shader at line 444 samples `texture2D(u_tex, v_uv)` with linearly interpolated UVs. With a single quad split into two triangles (indices `[0,1,2,0,2,3]`, line 449), each triangle gets *affine* UV interpolation — there's no projective `w` divide. This is the classic "two-triangle texture warp" artifact: a visible diagonal seam and incorrect foreshortening on non-trivial quadrilaterals.
   - The Python pipeline uses `cv2.getPerspectiveTransform` + `cv2.warpPerspective` (run_pipeline.py:465–467), which solves the 3×3 homography correctly.
   - For a painting photographed at 15° tilt this produces visibly bent vertical lines, and the more skewed the input, the worse it gets — exactly the cases where the user *needs* the tool.
   - **Fix:** compute the 3×3 homography on CPU (solve 8 linear equations from 4 point pairs), pass it as a uniform, do the perspective divide in the fragment shader (`vec2 src = (H * vec3(gl_FragCoord.xy, 1.0)).xy / w`). Or render with `mat3` per-vertex and use `varying vec3 v_uv` with `gl_FragColor = texture2D(u_tex, v_uv.xy/v_uv.z)`. Either way, a real projective warp is ~20 extra lines.

3. **[HIGH] `ctx.filter` adjustments operate in gamma-encoded sRGB on 8-bit data; sliders map to a range that crushes shadows / blows highlights at the extremes** (lines 116–120, 260, 263).
   - `buildFilter` returns `brightness(${1+b/100}) contrast(${1+c/100}) saturate(${1+s/100})` for slider range `-100..100`.
   - At `b = -100`, multiplier is `0` → image goes pure black. At `c = -100`, the image goes pure mid-grey. Sliders have no soft limit — first click on the far-left rail destroys the image.
   - `ctx.filter` does its work in the canvas's sRGB-encoded 8-bit space, so:
     - **Brightness** is a multiplicative gain in gamma space — highlights clip to 255 long before shadows lift. The same brightness boost applied in linear space (then re-encoded) would be far more even.
     - **Contrast** pivots around 0.5 in sRGB, not around middle-grey in linear, so the perceived effect on warm vs cool tones is asymmetric.
     - **Saturation** is HSL-style, which over-saturates already-saturated reds/oranges and dulls them later — particularly bad for an oil painter whose work *is* saturated reds and yellows.
   - Compounding: after every slider drag, `applyAll` redraws the watermark on top of the filtered preview, then export takes that as truth.
   - **Fix:** narrow slider range to `-50..+50` mapped to `0.5..1.5`, and at minimum do contrast/brightness via per-pixel ImageData loop with sRGB→linear→adjust→linear→sRGB. For real quality match the Python pipeline's LAB-based adjustments.

4. **[HIGH] No EXIF orientation, no ICC profile, no color management** (lines 226–245, 471, 472).
   - `FileReader.readAsDataURL` + `new Image()` honors EXIF orientation on modern Chrome/Safari, but the rule is browser-version-dependent and HEIC-imported JPEGs often have a "rotate 90" tag that browsers handle inconsistently.
   - Once the image is drawn to a `<canvas>`, the ICC profile is **stripped** and the canvas is forced to sRGB. iPhone photos default to Display P3 (~25% wider gamut than sRGB) — those vivid greens and oranges in a painting will visibly shift toward dull after the canvas roundtrip. The Python pipeline preserves the original color space via Pillow's ICC pass-through.
   - There is no `image-orientation` CSS hint and no manual orientation check.
   - **Fix:** read EXIF before drawing (use `exifr` or a 50-line manual parser), apply rotation explicitly, and warn the user when source is P3/AdobeRGB ("colors may shift on save").

5. **[HIGH] Every transform roundtrips through `toDataURL('image/png')` and a new `Image()`** (lines 388, 398, 438, 471, 472).
   - After perspective: line 460 returns `gc.toDataURL('image/png')`, then line 438 builds a new Image from it; same in `runAutoPersp` line 388.
   - After crop: line 471 does `tmp.toDataURL()` (defaults to PNG).
   - After resize: line 472 does `tmp.toDataURL()`.
   - PNG is lossless so per-op pixel data survives, but each roundtrip:
     - Allocates ~4× the image size as a base64 string (a 4000×3000 image → ~50 MB base64) and parks it in memory until the Image decodes.
     - Forces an async `Image.onload` race — the user's next button click can fire before `workingImg` is updated.
     - On Safari, `toDataURL` on large canvases (>16 MP) silently fails or returns `data:,` — which then becomes an Image with `width=0`, breaking everything downstream with no error.
   - **Fix:** use `OffscreenCanvas` references directly; never go through dataURL except at export. If you must materialize, use `canvas.transferToImageBitmap()` or `createImageBitmap(canvas)` — both stay in GPU/binary form and don't allocate the base64 detour.

## Browser vs Python pipeline

| Capability | Python pipeline | Browser version | Verdict |
|---|---|---|---|
| Crop accuracy | 3-strategy auto: Hough-intersection corners → quad contour → content bbox + frame-border trim (run_pipeline.py:776–849). ~19/20 on test set. | Manual rectangle only, or AI-suggested `{x,y,w,h}` from Claude vision call (line 391). No geometric auto-crop, and the Claude API call is broken (no `x-api-key`, would CORS-fail anyway, line 335). | **Python much better** — and stays better even if the AI call is fixed, because AI vision at 800px preview can't match Hough's pixel-accurate edges. |
| Perspective correction | `cv2.getPerspectiveTransform` + `warpPerspective` — true 3×3 homography with bilinear interpolation in source space (run_pipeline.py:465–467). Auto-detected via Hough line intersections (run_pipeline.py:585–678). | Affine-per-triangle WebGL warp (line 449) — **mathematically wrong** for any non-rectangular quad. Corners come from manual drag or broken AI call. No automatic geometric detection. | **Python much better** — the browser version doesn't produce a real perspective warp at all. |
| Straightening | Multi-signal: Hough intersection (preferred) → contour quad (fallback). Verifies edges with cross-edge LAB contrast > 65 (run_pipeline.py:670–676). | Same path as perspective — manual or AI-driven. No automatic. | **Python much better.** Browser has no concept of "this painting is tilted, let me fix it" without AI. |
| Color/exposure adjustment | Currently none (pipeline doesn't touch tone). When added, would naturally use LAB or linear space via OpenCV. | `ctx.filter` in sRGB, range `-100..+100` that clips to black/grey at extremes (line 263). | **Browser is the only one shipping anything**, but the implementation is amateur-grade. Could match by switching to ImageData loops in linear space. |
| Watermark quality | Not in pipeline. | Rasterized onto display canvas with `font:${canvas.width/24}px sans-serif` (line 266). Stroke + fill, no font fallback, anti-aliased by browser. | **Browser-only feature**, but currently sized to the *display* canvas so the watermark would be tiny if exports were at full res (~16px on a 4000px export). The font is "sans-serif" — platform-default, will differ Mac vs Windows. |
| EXIF/color preservation | Pillow reads EXIF orientation, preserves ICC profile through JPEG round-trip (run_pipeline.py:205–216). | EXIF is browser-dependent for the initial decode and **discarded** after canvas draw. ICC profile is **always lost** — canvas forces sRGB. No P3/AdobeRGB awareness. | **Python much better.** This is non-recoverable in browser without a WASM JPEG decoder (jpeg-js, libjpeg-turbo). |
| Large-image throughput | Streams files; opencv operations are in-place on numpy arrays. Tested up to ~24MP iPhone HEIC. | Every transform roundtrips through PNG dataURL (lines 388, 398, 438, 471, 472). A 4000×3000 image hits ~50 MB base64 per op. Safari fails silently on canvases > 16MP via toDataURL. | **Python much better.** Browser will OOM or silently truncate on real artwork sizes. |

## Critical pixel-quality risks

Specific places where the current code degrades the image without telling the user:

1. **Export saves at preview resolution** (lines 479, 517–521). The single biggest issue — Sveta clicks "Save to Drive" expecting her full painting and gets a ~600×460 thumbnail. No warning, no indication. The Sheets row says "4000×3000" (line 532) while the actual file is 600×460. Silent data destruction.
2. **Resize is destructive and forgets to remember** (line 472). `applyResize` overwrites `workingImg` with a downsized copy. A user who types 800×600 to test, then changes their mind, has already lost the original pixels (technically `origImg` still holds them, but `resetToOriginal` line 473 also resets all sliders and tool state — so it's "lose your work to recover resolution"). No "preview at this size, export at full" distinction.
3. **`tmp.toDataURL()` defaults to PNG with no quality flag** but is then read back via `new Image()` (lines 471–472). On Safari with >16MP canvases this silently returns `data:,` and the next Image has `width=0`. No try/catch around it, no size check.
4. **Affine-per-triangle perspective warp** (line 449). Even when the user drags corners perfectly, the warp itself bends straight lines along the triangle diagonal. Sveta will see her crisp vertical canvas edge become subtly kinked. She may attribute it to her photo, not the tool.
5. **`ctx.filter` brightness/contrast at extremes destroys data without limit** (line 263). Slider at -100 = `brightness(0)` = pure black. There is no clamp, no warning, no soft-knee. One accidental drag to the left edge of the slider and the image is gone (recoverable via reset, but the user may not realize).
6. **No `imageSmoothingQuality = 'high'`** anywhere (lines 252, 260, 397, 472). Default canvas downscale uses cheap bilinear, which produces visible moire on detailed brushwork. Setting `ctx.imageSmoothingQuality = 'high'` (a one-line fix) gets Lanczos-ish quality on Chrome/Safari.
7. **Watermark rasterizes at preview size**, would look 6× too small at full-res export if the export bug were fixed (line 266: `fs = max(12, canvas.width/24)`).
8. **HEIC inputs**: drop zone says "JPG, PNG, WEBP" but `accept="image/*"` (line 75) accepts HEIC. Desktop Chrome/Firefox cannot decode HEIC — file will silently fail to load (no `onerror` handler on the Image, line 229). iOS Safari can decode, so behavior diverges by platform. Python pipeline handles HEIC via pillow-heif.

## Out-of-scope concerns

- The Claude API call (line 335) is broken as-shipped — no `x-api-key`, hard CORS block from a `file://` page. The AI Integration Manager is the right owner. From an image-processing angle, even if the call worked, sending an 800px-wide downscaled JPEG to Claude (line 480) means AI-derived corners are only accurate to ±15 source pixels — fine for "is the painting tilted?" decisions, not fine as the actual transform input.
- Drive multipart upload sends a JPEG at the chosen quality (line 517), but the JPEG is the display canvas — the Google Integration Expert should know that "fixing the upload" doesn't fix what's being uploaded.
- A11y: range sliders have no `aria-label`, value spans aren't `aria-live` — Frontend Developer's lane.
- Test coverage: zero test images with mixed-orientation HEIC, no test for >24MP input, no test for non-sRGB source — QA's lane.
- No undo/redo history. Every destructive op is permanent until the user hits Reset, which loses everything else too. Frontend/UX lane but worth flagging because pixel-quality issues compound: a user who crops, resizes, then realizes the crop was off has no recovery path.
