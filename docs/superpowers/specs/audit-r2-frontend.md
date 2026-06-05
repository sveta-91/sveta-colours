# Frontend Developer — Round 2 Critique

## Updated verdict
`rewrite` — softened slightly. I'll keep voting rewrite, but a narrower rewrite than I implied in R1: re-do the HTML/CSS shell, keyboard layer, and token system; **preserve** the JS scaffolding the AI manager defends (JSON schema, lazy `aiData` cache, `needs_*` gating, `cachedToken` shape). It's a structural rewrite, not a from-scratch.

## Strongest points I agree with from the others

- **Image-processing: export saves the *display* canvas, not full-res** (`audit-r1-image-processing.md:15-19`, lines 248-254/479/517). This is the single bug that flips my "design is bad" reading into "the artifact is *functionally* lying to Sveta." Sheet says 4000×3000, Drive file is 600×460. I missed this entirely in my lane (see below).
- **AI manager: the JSON schema, `aiData` cache, and `needs_*` graceful-degradation are worth keeping** (`audit-r1-ai-manager.md:8-10`). Agreed — these are the only parts of the file that read like someone thought about a real workflow, and they're provider-agnostic. A rewrite that throws them out is wasteful.
- **QA: `Image.onerror` is unwired anywhere — the entire async chain can hang silently** (`audit-r1-tester.md:24-31`). Bigger than my "no responsive breakpoint" complaint. A keyboard-accessible drop zone (my R1 fix) is moot if dropping a corrupt JPEG produces a permanent blank screen with no message.
- **Google: formula injection via `valueInputOption=USER_ENTERED`** (`audit-r1-google.md:13`, `audit-r1-tester.md:13-23`). I flagged unsanitized `innerHTML` from Claude into `#aiBox`; I did not catch the much worse downstream path where Claude-generated `description`/`tags` get written *live as formulas* into Sveta's Sheet. Same root cause (untrusted text used unescaped), wider blast radius.
- **QA: AI auto-fill clobbers user-typed fields** (`audit-r1-tester.md:46-52`). Pure UX/frontend bug — `fillFields` only checks if AI returned a value, not whether the field is empty. I should have caught this in the editor card I reviewed.

## Points I push back on

- **AI manager calls this `improve`. With the export-resolution bug now visible, that verdict can't stand** — the file ships downscaled thumbnails to Drive while telling the Sheet they're full-res. That's data destruction, not a missing feature. Improve-tier fixes don't reach that.
- **Google's "targeted fixes" framing is defensible *for the Google layer alone*** (drive.file scope, GIS token client, multipart upload are all correct). But the Google layer is ~70 lines out of ~570. Five of those lines have blockers (formula injection, header race, dedup, expiry, folder ownership). "Targeted" stops being targeted when half the surrounding HTML/CSS/keyboard layer also has to be redone.
- **Image-processing's "rewrite the math" is correct but doesn't require rewriting the file** — swapping affine-per-triangle for a real homography is ~20 lines, and `OffscreenCanvas`-with-full-res-mirror is a localized refactor. Their *verdict* matches mine; their *scope* is narrower than they imply.
- **QA's drop-on-folder, multi-file silent reject, OAuth-popup-closed hang** (`audit-r1-tester.md:33-43, 71-72`) — these are bugs, not rewrite drivers. They'd be straightforward additions to the existing event handlers.
- **I disagree with my own R1 take that the visual language has "no relationship" to `index.html`.** Re-reading: the *information architecture* (top: image work area + metadata; bottom: AI + actions) is fine. It's tokens, type scale, and accent color that diverge. Those are a stylesheet swap, not a structure rewrite.

## What I missed in Round 1

Things firmly in the frontend/UX lane that other agents caught:

- **AI overwrites user input** (QA #4). The `fillFields` race is a frontend concern — it's about user intent and form state, not AI. I reviewed lines 320-325 for a11y and missed the clobber.
- **Export-resolution mismatch is a UX-visible lie** (image processing #1). Image-processing owns the *fix*, but the *symptom* is "user is told one size, gets another" — that's a frontend trust failure. I should have flagged that `setupCanvas` caps at 460 px and `downloadImg` uses the same canvas.
- **OAuth-popup-cancelled leaves Save All stuck on "Saving…" forever** (QA bug log). Pure frontend issue — promise never resolves, spinner never clears, button never re-enables. I praised the button-state coupling and missed the cancel path.
- **Resize input accepts negatives** (`-5` → blank canvas, QA medium). Input-validation gap I should have seen in the metadata card review.
- **Auto-correct success ✓✓✓ shown even when AI silently failed** (QA medium). The 3-step indicator I called a "genuine UX strength" lies when `analyzeWithClaude(null, true)` swallows the error. My praise was premature.
- **`new Date().toISOString().slice(0,10)` is UTC, wrong for a Canadian artist after 7-ish pm** (Google out-of-scope). Frontend/copy issue. Missed it.

## What 'rewrite' should mean concretely

**Minimum scope (~60-70% of file rewritten, ~30-40% preserved):**

Rewrite:
- All HTML (semantic landmarks, real labels, `<h1>`, `<main>`, `<form>`, `<fieldset>`s).
- All CSS (defined token block aliased to `index.html`'s bone/ink palette; Fraunces/Inter; no Tabler font; real responsive breakpoint at 720px; `:focus-visible` rings; `imageSmoothingQuality='high'`).
- Drop zone → `<button>` with full keyboard path; `Esc` to cancel modes; arrow-key nudge on perspective corners and crop edges.
- `aria-live` regions on `#saveStatus`, `#autoSteps`, `#aiBox`.
- Full-res offscreen canvas mirror; export path reads from *that*, not preview (image-processing's call but I'm signing it).
- Replace `innerHTML` writes of AI text with `textContent` (and switch Sheets to `valueInputOption=RAW`).
- Add `img.onerror` + file-size guard + multi-file feedback + drag-folder rejection message.

Preserve verbatim (per AI manager):
- AI JSON schema shape (lines 339-354).
- `aiData?.needs_*` gating in `runAutoCorrect` (lines 294-309).
- Lazy `analyzeWithClaude` + `aiData` cache (lines 241, 290, 426).
- `cachedToken` + `getAccessToken` structure (lines 484-491) — but fix the 58-min magic number to `expires_in`.
- The 3-step indicator data structure (lines 89, 281-288), once it's wired to honest status.

Cost estimate: ~6-8 hours focused. An "improve" path that fixes all 27+ items across all four lanes is more work than that and ships an incoherent file at the end. Rewrite wins on both quality and time.
