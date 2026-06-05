# QA Tester — Round 1 Audit

## Verdict
`rewrite` — the core flow has too many silent-failure paths (no `img.onerror` anywhere, no multi-file feedback, AI analysis broken-as-shipped at line 335, formula injection into Sheets, no atomicity between Drive upload and Sheet write) for "improve" to be safe in a single round.

## Top 3 strengths
1. **Double-click guard on Save All / Auto-correct (lines 274, 496):** button is disabled synchronously before any `await`, so a second click during the run is a no-op. Clean implementation.
2. **Empty / undersized selections handled cleanly:** `applyCrop` rejects rects < 4 px (line 471), `runAutoCrop` rejects < 10 px (line 395), `drawWatermark` exits on empty trimmed text (line 265), `applyResize` exits on `!w||!h` (line 472), `perspWarpWebGL` returns `null` if WebGL is unavailable and both callers check (lines 387, 437). The “press Apply with nothing selected” class of bugs is largely covered.
3. **Tools are mutually exclusive:** `toggleCrop` cancels perspective mode and vice versa (lines 403, 464), so the two overlays cannot fight over input events.

## Top 5 issues (blockers + highest-severity bugs)

### 1. Formula injection via Title (and any text field) into Google Sheets — **HIGH**
- **Lines:** 547, 550, 553 (all use `valueInputOption=USER_ENTERED`)
- **Repro:**
  1. Upload any image.
  2. Set Title to `=HYPERLINK("https://evil.example/x","click")` (or `=IMPORTDATA(...)` to exfiltrate, or just `=1+1`).
  3. Click Save all.
  4. Open the linked Google Sheet.
- **Expected:** Title cell stores the literal string the user typed.
- **Actual:** Sheets evaluates it as a formula. With `IMPORTDATA`/`IMPORTXML` an attacker-supplied filename could make Sveta's Sheet pull data from a third party on every open. The same risk applies to Description, Notes, Tags, Price.
- Fix direction: use `valueInputOption=RAW`, or prefix any value starting with `=`, `+`, `-`, `@` with a leading apostrophe.

### 2. `Image.onerror` is never wired anywhere — auto-correct and crop can hang forever — **HIGH**
- **Lines:** 228–243 (`handleFile`), 388 (`runAutoPersp`), 398 (`runAutoCrop`), 438 (`applyPersp`), 471 (`applyCrop`), 472 (`applyResize`)
- **Repro:**
  1. Rename a small `.txt` file to `painting.jpg` (or use a truncated/corrupted JPEG).
  2. Drop it on the upload zone — `file.type` is `image/jpeg` so the type guard passes.
  3. `FileReader` reads it, `new Image()` fires neither `onload` nor (caught) `onerror`.
- **Expected:** Clear "Could not read image" error, return to drop zone.
- **Actual:** Drop zone vanishes, work area stays blank, no message. Same hang pattern occurs if `runAutoPersp`/`runAutoCrop`'s intermediate Image fails to decode — the `await new Promise()` never resolves, Auto-correct's button stays in `Working…` forever.

### 3. Multi-file drag, folder drag, and non-image drop are all silent — **HIGH**
- **Lines:** 203–204, 224
- **Repro:**
  - Drop a folder of 30 paintings on the zone.
  - Drop two `.jpg` files in one gesture.
  - Drop a `.pdf`.
- **Expected:** Either accept all files in turn, or show "Please drop a single image" with the count.
- **Actual:**
  - Folder: `dataTransfer.files[0]` is the directory entry with `file.type === ''` → `handleFile` returns silently. Zone just sits there. Sveta has no idea why nothing happened.
  - Two files: file 1 loads, file 2 is discarded with no notice.
  - `.pdf`: silent reject.

### 4. AI auto-fill clobbers fields the user has already typed — **HIGH**
- **Lines:** 320–325 (`fillFields`), 369
- **Repro:**
  1. Upload a painting. `analyzeWithClaude` starts.
  2. Immediately type a Title (so the auto-fill condition `title && !titleFilled` becomes true the moment AI returns) and start writing a Description in your own words.
  3. AI response arrives ~3–10 s later.
- **Expected:** Your typed Description is preserved; AI suggestions appear in a separate “suggestions” area for you to accept.
- **Actual:** `fillFields` overwrites `fDesc.value` with the AI description (line 321 only checks if AI returned a value, not if the field is empty). Same for Medium and Tags. Sveta's typing silently vanishes. Considering AI fetch latency, this is very easy to hit.

### 5. Drive upload + Sheets append is non-atomic — orphan files on partial failure — **HIGH**
- **Lines:** 494–507 (`saveAll`), and the dedup race in `upsertSheetRow`
- **Repro A (orphan file):**
  1. Click Save all on a new artwork.
  2. Drive upload succeeds → `driveLink` returned.
  3. Block Sheets API in DevTools (Network → block `sheets.googleapis.com`) or revoke the spreadsheets scope between calls.
  4. Status reads "Error: …".
- **Expected:** Either a transaction-like rollback (delete the just-uploaded Drive file) or a clear "Image uploaded but row not added — retry?" with a button that only re-runs the Sheets step.
- **Actual:** File is in Drive forever with no row in the sheet. User has to find and delete manually. The button text on retry says "Save all" — clicking it re-uploads a *second* copy to Drive.
- **Repro B (TOCTOU dedup race, two tabs):**
  1. Open the page in tab A and tab B with the same title.
  2. Click Save all in A, then B before A finishes.
  3. Both read the sheet, neither sees the row, both append.
- **Expected:** One row.
- **Actual:** Two duplicate rows. (Sveta is a single user but two tabs is a stated realistic scenario.)

## Bug log (everything else)

- **HIGH | save flow | OAuth popup closed by user leaves Save All button stuck in "Saving…" forever.** `getAccessToken` (line 484) never resolves if the GIS token client never invokes its callback; the promise leaks and the spinner never clears.
- **HIGH | save flow | `cachedToken` is set in a `setTimeout` that may fire while a save is mid-flight.** Line 488 schedules `cachedToken=null` after ~58 min, but if the user keeps the tab open the next `saveAll` will re-prompt mid-session even though the underlying token is still valid for Sheets API requests already in flight.
- **HIGH | sheets | `upsertSheetRow` writes data unconditionally to columns A–M assuming `COLS` order, with zero validation that the sheet's actual header row matches.** If Sveta reorders columns in the Sheet, every future save writes into wrong cells with no warning.
- **HIGH | sheets | Title match is case-insensitive but not trimmed.** Line 531: `rows[i][1].toLowerCase()===title.toLowerCase()`. Existing row `"Autumn Light"` and new title `" Autumn Light"` (leading space, easy to paste in) → no match → duplicate row inserted. Also vulnerable to NBSP vs space.
- **HIGH | auto-correct | Re-running Auto-correct after Apply Crop applies the same fractional crop again.** `aiData.crop` is fractions of the *original* image, but after one crop `workingImg` is the cropped image. `runAutoCrop` uses `workingImg.width` (line 393), so the same fractions are reapplied to a smaller frame, cropping again. Same issue for perspective: `aiData.perspective_corners` are fractions of the original.
- **HIGH | auto-correct | Reset while Auto-correct is in flight produces a corrupted state.** `resetToOriginal` (line 473) sets `workingImg=origImg`, but `runAutoPersp` / `runAutoCrop` are still awaiting; when their `img.onload` fires, it overwrites `workingImg` again with the corrected image, silently undoing Reset.
- **MEDIUM | upload | No file-size guard.** Drop a 50 MB photo from a DSLR: `FileReader.readAsDataURL` holds the base64 (~67 MB string) in memory, then `canvasToBase64` builds another, then Claude payload, then `canvas.toBlob` for Drive. Page can OOM on Safari especially. No "this is too large" message.
- **MEDIUM | upload | `accept="image/*"` allows HEIC but Chrome/Safari can't decode HEIC to a canvas-drawable Image.** Drop-zone text says "JPG, PNG, WEBP" but iPhone photos are HEIC by default. Same silent-fail path as bug #2.
- **MEDIUM | crop / perspective | No global `window.mouseup` listener.** If user mousedowns on the overlay then releases outside it: `cropStart` stays non-null (line 466) and `dragIdx` may stay set (line 421). Next mousemove inside the canvas resumes a "drag" the user thinks they ended.
- **MEDIUM | layout | `setupCanvas` runs once per image; window resize is not handled.** Resize the browser while in crop mode and the canvas+overlay sizes are stale, so crop rectangle coordinates point at the wrong pixels.
- **MEDIUM | resize | `max="8000"` on the input is HTML5 advisory only; not enforced in JS.** Type `40000` and `applyResize` creates a 40000×40000 canvas → most browsers will throw or return a blank canvas (max canvas area is ~16384² on Chrome, lower on Safari). Silent corruption of `workingImg`.
- **MEDIUM | resize | Negative values bypass the guard.** `-5` → `+value === -5`, which is truthy, so `if(!w||!h)` does not return. `tmp.width = -5` is then coerced to 0 or rejected, depending on browser → blank `workingImg`.
- **MEDIUM | ai | Auto-correct triggers `analyzeWithClaude(null,true)` if no analysis yet; with the broken API this fails silently (silent=true) and the user is left with three green ✓ check marks but no actual changes.** Steps end with "Perspective ✓ Crop ✓ Adjustments — review" implying success.
- **MEDIUM | ai | aiBox stays at the red "API error" string forever even after user manually fixes the image.** No way to clear it; persists into Save and looks like a permanent failure.
- **MEDIUM | sheet write | Header row is only added on a *completely empty* sheet (`rows.length===0`, line 547).** If the sheet had headers manually deleted but had stray content somewhere, the read returns rows but the header check is skipped. Data appends with no header above it.
- **MEDIUM | save flow | "Save all" success label `'Saved to Drive + Sheets ✓'` is shown even if `upsertSheetRow` silently no-ops (e.g. an `await` rejection swallowed).** The catch on line 505 catches actual throws but not Sheets-level partial success.
- **MEDIUM | save flow | `saveToDrive` (the standalone "Drive" button) never updates the Sheet but uses the same Title, so the user can think the painting is "saved" while no row exists.** Two adjacent buttons with overlapping verbs.
- **MEDIUM | title field | Required asterisk in label but no client-side block on Save All with empty title.** Drive upload gets `photo.jpg`, sheet gets row with empty B column. The case-insensitive dedup then matches every future empty-title save → all overwrite the same first row.
- **MEDIUM | watermark | Very long text overflows: anchored right, no width constraint, no truncation; on a 320 px-wide thumbnail it just runs off the left edge.** No error, just clipped output.
- **LOW | upload | `fileInput` reuses the same `<input>` element; uploading the same filename twice in a row triggers no `change` event.** Standard browser behavior, but the page doesn't clear `fi.value` after handling.
- **LOW | adjustments | `applyAll` re-renders synchronously on every `input` event of a slider; on a 5000 × 5000 image the scrubbing is laggy.** No `requestAnimationFrame` throttle.
- **LOW | export | Quality slider value is read at moment of download, but applied even when PNG is selected (`canvas.toBlob` ignores quality for PNG).** Harmless but the row is hidden via JS for png only on a `change` event — initial load + format=jpeg means the quality row is correctly visible. Fine, just noting.
- **LOW | ai | Analysis sent to Claude uses a max-800 px JPEG at 0.85 quality (line 480).** AI quality detection / crop fractions may be off for textured paintings; not a blocker but degrades correctness of the "needs_crop" suggestion.
- **LOW | a11y | Most buttons are icon-only with only an `<i>` glyph + visible text; the `aria-hidden="true"` on icons is correct, but the "Apply crop"/"Cancel"/"Apply" buttons are positionally context-dependent and no `aria-label` distinguishes "Apply" for crop vs perspective.** Screen reader hears "Apply" twice with no context.
- **LOW | a11y | Drop zone is a `<div>` with click handler but no `role="button"`, no `tabindex`, no keyboard activation.** Keyboard-only upload is impossible.
- **LOW | a11y | Range sliders have no associated `<label for="">`; only an adjacent `.ctrl-label` div.** Screen readers will not announce "Brightness".
- **LOW | css | The file uses `var(--color-*)` and `var(--font-sans)` tokens with no fallbacks for body color, border, etc.** Opened standalone in a browser without the parent theme stylesheet (which is the stated deployment — `file://`), most UI is unstyled / invisible (white text on white background risk for text-primary).
- **LOW | css | Background is `transparent` on `body` (line 7).** On `file://` with no parent, the browser's default white/dark applies and the cards' `var(--color-background-primary)` may not exist → cards have no visible background.
- **LOW | drive | Default fallback link `https://drive.google.com/drive/folders/${DRIVE_FOLDER}` is written into the Drive-link column when standalone "Save to Drive" was used or when `driveLink` is falsy.** The Sheet then has a row pointing to the folder, not the file, with no indication.
- **LOW | reset | Reset does not restore the watermark text (only opacity).** If user changed watermark, Reset keeps the new text.
- **LOW | crop | Toggling Crop while a selection exists silently discards it (line 464).** No "are you sure".

## Out-of-scope concerns

- Whether the AI feature is the right shape at all — it's broken as shipped (no `x-api-key`, plus CORS is forbidden for direct browser calls to `api.anthropic.com`). The AI-integration manager owns the choice of provider, but my testing assumes this stays broken in any "improve" path.
- The watermark logic, perspective-warp math, and crop-fraction interpretation are mechanically correct in isolation; whether they're the right algorithm vs. the existing Python pipeline is the image-processing expert's call.
- OAuth scopes (`drive.file` + `spreadsheets`) and the multipart upload format itself look correct on a quick read; the Google-integration expert should sanity-check the exact request body shape.
- All design / theming questions (using `var(--color-*)` from an outside system; no light/dark fallback) are the frontend developer's lane — I noted them only where they cause a *functional* failure (invisible UI on `file://`).
- Persistence (page refresh wipes state) is a UX question, not a bug per se — flagged for the frontend developer.
