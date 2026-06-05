# Google Integration Expert — Round 2 Critique

## Updated verdict
`improve` — still. The Drive/Sheets request shapes are right; the failures live in *what gets sent* and *how state is sequenced*, not in the OAuth/upsert architecture itself. Three rewriters voted on bugs that mostly live outside my lane (preview canvas as truth, no styles at `file://`, silent-failure paths), and within my lane the fixes remain targeted: `RAW` instead of `USER_ENTERED`, await the header write, normalize the dedup key, honor `expires_in`, dedup by Drive name. None of that requires throwing the file out.

## Strongest points I agree with from the others
- **Image: "Drive upload sends the display canvas" (`audit-r1-image-processing.md` issue #1, lines 15–19 + line 81).** This is a Google-lane bug I underweighted — my `uploadToDrive` is the function physically transmitting a thumbnail-sized JPEG with honest `dims` metadata pointing at the original resolution. The Sheet row becomes a lie about a file. The fix is on the image side (full-res offscreen canvas), but I'm the one shipping the wrong bytes.
- **QA: "Drive upload + Sheets append is non-atomic — orphan files on partial failure" (`audit-r1-tester.md` issue #5).** I called out the dedup race but didn't articulate the orphan-on-partial-failure case. A failed Sheets write after a successful Drive upload leaves a file with no row, and clicking retry uploads a *second* copy. This is squarely my lane and I missed it.
- **QA: "OAuth popup closed leaves Save All stuck in Saving… forever" (bug log, line 72).** GIS's `requestAccessToken` callback simply never fires if the user closes the popup. My `getAccessToken` promise leaks. Needs a timeout or a `cancel` handler — I didn't flag this.
- **QA: "Header row only added on completely empty sheet" (bug log line 86).** Sharper framing of my own issue #2 — the race is real but the *trigger condition* is also wrong. Stray content with no headers passes the `rows.length===0` check and data appends below empty cells.
- **AI manager: "drive.file scope only grants access to files this app created — re-running in a different browser profile cannot update files saved earlier" (out-of-scope note, line 39).** I noted the folder-ownership wrinkle but not the *per-OAuth-client-installation* angle. Two browsers = two file sets, neither can edit the other's. Worth documenting.

## Points I push back on
- **Frontend's "BLOCKER: page renders unstyled standalone" rests on a deployment assumption that contradicts mine.** Frontend tested `file://` and found undefined tokens; *my* finding is that GIS will not authenticate at `file://` *at all* — Sveta cannot use this file from `file://` regardless of styling. Both findings collapse into the same requirement: she must serve it. So Frontend's blocker is real, but it's the same blocker as mine, not an independent one.
- **Image's "WebGL warp is affine-per-triangle" is correct but not a rewrite trigger by itself.** It's a ~20-line shader fix, not an architectural redesign. Same for `imageSmoothingQuality='high'` (one line). The export-resolution bug is the only one that genuinely warps the whole data flow.
- **QA's "rewrite" verdict over-rotates on count of bugs.** Most of QA's bug log is bounded-fix territory (input validation, `img.onerror` handlers, trim/normalize, atomicity wrapper, `window.mouseup` listener). That's a thorough audit, not a structural condemnation.
- **AI manager's Gemini-via-`file://` recommendation conflicts with my GIS finding.** AI proposes a `file://` workflow with `localStorage` API keys; GIS forbids `file://`. If Sveta serves the page (which she must, for Google auth), Gemini works too — but the "no server needed" framing in `audit-r1-ai-manager.md` is wrong.
- **Image is right that I "should know what's being uploaded" — but the *fix* is in the image lane.** My upload code is correctly multipart-encoding whatever blob is handed to it; the bug is upstream. Coordinated fix, not a Google-lane rewrite.

## What I missed in Round 1
- **Atomicity between Drive POST and Sheets PUT/append.** I flagged duplicates from re-saves, not orphans from partial failure. Needs either Sheets-first-with-pending-link, or a rollback that `DELETE`s the Drive file on Sheets failure.
- **OAuth popup cancellation leaks the promise.** GIS callback never fires on user-closed popup; `await getAccessToken()` never resolves. Needs a timeout race or an explicit cancel path.
- **The upload payload is wrong.** I audited *how* I upload, not *what*. The blob coming out of `canvas.toBlob` is preview-sized. Even with my multipart upload perfect, the file in Drive is a thumbnail. Image processing flagged this; it belongs in any fix list I sign off on.
- **`drive.file` scope behavior across browser profiles.** Files created by this OAuth client in browser A are invisible to the same client in browser B (different installation). My folder-scope note didn't capture this.
- **`canvasToBase64`'s 800px downscale also flows into the upload path indirectly** — if any auto-correct uses the AI's corners/crop computed against an 800px image and then writes back to `workingImg` before save, the saved file's geometry inherits 800px-quantized decisions. Subtle, but downstream of my upload.

## Deployment story implication
The two findings collapse into one hard requirement: **Sveta cannot use this page from `file://` at all.** Frontend's "no styles" and my "GIS won't auth at `file://`" point at the same fix.

**Minimum bootstrap:**
1. From repo root: `python3 -m http.server 8000` (already installed on macOS).
2. Open `http://localhost:8000/tools/photo_manager.html` (not `127.0.0.1` unless that origin is also registered).
3. In Google Cloud Console for the existing `CLIENT_ID`: add `http://localhost:8000` to Authorized JavaScript origins.
4. Frontend defines a `:root { … }` token block so the page is actually styled when served from that origin.
5. Document the three steps above in `tools/photo_manager.html`'s file header or a sibling `README.md` — none of this is currently discoverable, and Sveta will double-click the file, see a blank page, and assume it's broken.

If we want a true zero-bootstrap option, we'd need to move Google integration to a desktop-OAuth flow (a small local helper script), which is a rewrite. For now: serve it, register the origin, define the tokens. That's the cheapest path to a working page.
