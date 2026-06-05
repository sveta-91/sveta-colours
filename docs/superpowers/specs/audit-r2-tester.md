# QA Tester — Round 2 Critique

## Updated verdict
`rewrite` — I'm holding. Image-processing's downscale-on-export finding actually *strengthens* the case: the file is producing wrong output even on the happy path, not just the edge cases I catalogued.

## Strongest points I agree with from the others
- **Image-processing's "export saves the display canvas" (lines 479, 517).** This trumps my formula-injection bug for user impact — Sveta will quietly accumulate ~600×460 thumbnails in Drive, with Sheet metadata that *lies* about dimensions (`workingImg.width/height` on line 532). Silent data destruction on the primary action. I should have caught this in my own pass and didn't.
- **Frontend's "all design tokens are undefined" (lines 7, 12–62).** This is QA-critical: at `file://` the page is *visually broken*, which means none of us could have tested it as a user would. My own audit was code-read only. That has to be treated as a finding, not a footnote.
- **Google's header-init race (line 547).** Concrete repro: open a brand-new empty sheet, click Save all → `upsertSheetRow` reads `A:B` (empty), fires `values:append` for `COLS`, then *without awaiting* falls through to `rowIdx>0 ? PUT : append` and fires a second `append` for the data row. The two appends race; on a slow connection the data row can land at A1 and the header row at A2. Reload the page and dedup is now permanently broken because row 1 is no longer headers.
- **AI manager's Gemini Flash recommendation.** Correct provider choice and the `responseSchema` option also fixes my "JSON parse blows up on prose preamble" concern (LOW in my list).
- **Image-processing's affine-per-triangle warp (line 449).** I had marked perspective math as in-scope for them; their finding is sharper than I expected — even a *correctly dragged* quad produces bent lines. The "Apply" button is a lie on any non-trivial tilt.

## Points I push back on
- **AI manager calling this "improve."** Gemini swap is fine, but it adds *new* QA scenarios I now have to write: rate-limit 429 (15 RPM free tier — easy to hit on a batch import day), malformed JSON despite `responseSchema` (still possible on safety-blocked outputs), slow-network timeout, key-not-yet-pasted state, key revoked mid-session. None of those are handled in the current code shape. That's rewrite territory, not improve.
- **Google's "improve" verdict.** The header-init race + drive.file scope + filename-collision-creates-duplicates combination means a long-running user accumulates orphan Drive files *and* corrupt Sheet rows. That's not three small fixes, it's a redesign of the save path.
- **Frontend's "no `<title>`" framing as a11y-only.** It's also a QA signal — there is literally no smoke test that this file ever rendered. I'm escalating the CSS-tokens issue from "LOW" in my Round 1 to BLOCKER.
- **My own #1 (formula injection).** Still real, but lower-frequency than image downscale. Demoting.
- **Image-processing dismissing AI corners as "±15 source pixels."** True at 800px preview, but their proposed full-res send blows up Gemini's input-token budget on a 24MP painting. Trade-off, not a fix.

## Re-ranked top bugs (combined, by user-impact severity)
1. **Export saves preview-resolution canvas, not `workingImg`** — image-processing. Every save corrupts the asset.
2. **Page renders unstyled at `file://` (undefined CSS tokens)** — frontend. Audit was code-only because nobody could actually use it.
3. **Claude API call cannot work from browser (no key, CORS-blocked)** — AI manager. The AI button is dead on arrival.
4. **`Image.onerror` never wired → corrupt/HEIC/`.txt`-as-`.jpg` hangs Auto-correct forever** — me.
5. **Affine-per-triangle WebGL warp bends straight lines** — image-processing. Tool actively damages the photo.
6. **Formula injection via Title/Description into Sheets (`USER_ENTERED`)** — me + Google.
7. **Header-init race on first-save empty sheet** — Google. Repro above; corrupts the Sheet permanently.
8. **Drive upload + Sheets append non-atomic → orphan files** — me. Retry re-uploads a second copy.
9. **AI auto-fill clobbers fields the user has already typed** — me. Latency-dependent race, easy to hit.
10. **Multi-file / folder / non-image drop silently rejected** — me + frontend. Sveta's most likely "what just happened" moment.

## What 'safe to ship' looks like
A next iteration is safe when *all* of these hold on real artwork:
- **Round-trip pixel test:** drop a 4000×3000 JPEG, click Save to Drive with no edits — the file in Drive must be 4000×3000, byte-for-byte the same image content, with Sheet `dimensions` cell matching.
- **`file://` smoke test:** double-click the file in Finder, no local server. Every section visible (no white-on-white), every button reachable by keyboard, focus ring visible.
- **Network-failure matrix:** block `googleapis.com` → user sees actionable error, button re-enables, no orphan Drive file. Block `generativelanguage.googleapis.com` → manual fields work, status reads "AI offline."
- **Corrupt input:** rename `notes.txt` to `art.jpg`, drop it → "Could not read image," drop zone returns. Drop a folder → "Please drop one image at a time."
- **Sheets safety:** Title `=1+1` lands in the cell as literal text `=1+1`. Title `" Autumn Light"` (leading space) dedups to existing `Autumn Light`.
- **Two-save idempotence:** click Save all twice on the same artwork (network is slow) → exactly one Drive file, exactly one Sheet row.
- **AI race:** start typing Description before AI returns → AI suggestions go into a separate "Suggestions" affordance, never overwrite.
- **Token expiry:** advance system clock 1h, click Save → re-prompt happens cleanly, save resumes; no stuck spinner.
- **EXIF orientation:** drop an iPhone HEIC photographed in portrait → exports portrait, not sideways.
