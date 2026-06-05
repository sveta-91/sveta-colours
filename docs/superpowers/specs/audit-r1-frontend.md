# Frontend Developer — Round 1 Audit

## Verdict
`rewrite` — markup is unsemantic, the entire design system depends on undefined host CSS variables (page renders nearly blank standalone), there is no keyboard path through the editor, and the visual language has no relationship to `index.html`'s editorial bone/ink aesthetic.

## Top 3 strengths
- **Compact, readable single-file layout.** The grid skeleton (`.top-layout` / `.bottom-layout`, lines 9–11) and 3-card right column are easy to scan and easy to evolve.
- **Honest progress affordance during auto-correct.** The 3-step indicator (`autoSteps`, lines 89, 281–288, 312) — Perspective → Crop → Adjustments with spinner / checkmark / pending states — is a genuinely good piece of UX feedback for a multi-stage AI operation.
- **Status line + button state coupling.** `setStatus()` (line 481) with `.status.ok` / `.status.err` (lines 51–53), plus consistent `btn.disabled=true` + spinner swap during async work (lines 274, 425, 496, 509), is a clean pattern. The `aria-hidden="true"` on every `<i class="ti …">` icon (lines 70–145) is correctly applied so screen readers don't read garbage glyphs.

## Top 5 issues

1. **[BLOCKER] All design tokens are undefined — page renders unstyled standalone.**
   Lines 7, 12–62 use `var(--color-text-primary)`, `var(--color-background-primary)`, `var(--color-border-tertiary)`, `var(--border-radius-md)`, `var(--font-sans)`, etc. None of these are declared in this file or imported. Opened as `file://` (the stated deployment mode), text inherits the UA default (black) on `background:transparent` (line 7), borders collapse to `0.5px solid` of nothing, cards have no fill, and the drop zone has no visible outline. The variables look copy-pasted from Claude.ai's chat-artifact host. **Fix:** define a `:root { … }` token block at the top of the `<style>` that mirrors `index.html`'s palette (`--bg:#EAE6DE; --surface:#F2EFE8; --text:#14130F; --border:#D6CFC1; --accent:#1A2640;` etc.) and rename references — or alias the chat-host names to those values. Without this, the page is unusable outside the artifact iframe.

2. **[BLOCKER] No semantic structure or landmarks.**
   No `<title>`, no `<main>`, no `<header>`, no `<form>`, no `<fieldset>`, no `<label for>` pairing. The only heading is a `.sr-only` `<h2>` (line 65) — so the document has no `<h1>`, fails the WCAG "page must have one main heading" rule, and a screen-reader rotor shows zero landmarks and one orphan h2. Every text input is sighted-only: `<label>Title *</label><input id="fTitle">` (lines 160–170) — the label is a visual `<div>`-style label not connected by `for`/`id`. **Fix:** wrap the editor in `<main>`, add a real `<h1>` ("Add artwork"), convert each card to `<section aria-labelledby="…">`, replace card-title divs with `<h2>`, and either use `<label for="fTitle">` or wrap inputs inside labels. Convert the metadata card into a `<form>`.

3. **[HIGH] Editor is mouse-only — no keyboard path through crop / perspective / drop zone.**
   The drop zone (line 71) is a `<div>` with `cursor:pointer` and a click handler (line 200); it has no `tabindex`, no `role="button"`, no Enter/Space handler. Keyboard users cannot trigger the file picker without tabbing into the hidden `<input type=file>` (which is `display:none`, line 19 — focus-unreachable in most browsers). Crop and perspective are pointer-only: mousedown/mousemove/mouseup on the canvas overlays (lines 421–423, 466–468). There is no way to nudge perspective corners with arrow keys, no `Esc` to cancel a mode, no `Enter` to apply. **Fix:** make the drop zone a `<button type="button">` (or add `role="button" tabindex="0"` and a keydown handler for Space/Enter); replace `display:none` on the file input with the standard visually-hidden pattern; add keyboard nudging for perspective corners (Tab between corners, arrows to move, Enter to apply, Esc to cancel); same for the crop rectangle.

4. **[HIGH] No status-message announcements for screen readers; long ops are invisible.**
   The status div (`#saveStatus`, line 147) is plain text — no `role="status"`, no `aria-live`. The auto-correct step indicator (`#autoSteps`, line 89) is the same. A non-sighted user clicks "Save all" and gets silence for 5–30 seconds with no announcement of "Uploading to Drive… / Updating Sheets… / Saved" (line 500–504). The Claude analysis box (`#aiBox`, line 155) also silently updates. **Fix:** add `role="status" aria-live="polite"` to `#saveStatus`, `#autoSteps`, and `#aiBox`. For the auto-correct flow, emit text-only progress phrases (the current HTML uses spinners and checkmarks that SR will read as "·" and "✓").

5. **[HIGH] Layout breaks at <600px and there is no responsive breakpoint at all.**
   `grid-template-columns:1fr 320px` (line 9) with `padding:1rem` (line 7) means on an iPhone (~390px viewport) the canvas column is squeezed to ~50px and the file becomes unusable on the device Sveta is most likely to upload artwork *from*. The bottom layout (line 11) has the same problem (`1fr 1fr`). The tool-bar (line 23) wraps but the canvas itself doesn't reflow because `setupCanvas` reads `area.clientWidth` once and never re-runs on resize (line 250). **Fix:** add a `@media (max-width: 720px)` block collapsing both grids to `grid-template-columns:1fr`; cap `.col-right` width to 100%; add a `window.addEventListener('resize', …)` (debounced) that calls `setupCanvas(workingImg)` so the canvas re-fits.

## A11y findings

- **No `<html lang>`** (line 2) — screen readers default to system locale, mis-pronouncing English text. `index.html` has `lang="en"` (its line 2); copy that.
- **No `<title>`** in `<head>` — browser tab shows the file path. Required by WCAG 2.4.2.
- **No `<h1>`.** Only an off-screen `<h2>` (line 65). Heading hierarchy is broken from the start.
- **Card-title elements are styled `<div>`s with an icon (`.card-title`, line 13)** — should be `<h2>` so the page has scannable structure. Currently the rotor shows zero headings.
- **No `<label for>` association** on any of ~12 form inputs (lines 104–170). Each field-row uses a sighted label only. Screen reader reads "edit text" with no context.
- **Required field is signalled visually with "Title *"** (line 160) and nowhere else; the input lacks `aria-required="true"` and there is no validation surfaced when the user hits Save with an empty title (line 526 silently uses empty string).
- **Slider values are read as numbers but not labelled with units or scale** — `<input type="range" id="brightness" min="-100" max="100">` (line 116) has no `<label>` and no `aria-valuetext`. SR announces "slider, -100 to 100" but never says "brightness".
- **Focus is invisible.** No `:focus-visible` styles anywhere. `index.html` has a proper focus ring (its lines 113–119); this file has none, so keyboard users (if they could navigate) wouldn't know where they are.
- **Color contrast not verifiable** because all colors come from undefined vars (see issue 1). On any sensible token set, the tag chip (`.tag`, line 36) at 10px is below the WCAG minimum of 12px / 16px for body text and probably below 4.5:1 anyway.
- **Spinner is a non-text indicator** (`.spinner`, line 54) with no SR-readable equivalent in the button (the original label is replaced by "Working…" on line 274, which is good — but `Save all` is replaced with "Saving…", line 496, also good; this is fine, just be consistent and never lose the text).
- **No `Esc` handler** to cancel crop / perspective mode (lines 412, 465) — pointer-only escape.
- **The hidden file input** uses `display:none` (line 19), which removes it from the tab order. Use the standard `.sr-only`-style absolute clip instead, so keyboard users can focus the input directly.
- **`oninput="onTitleInput()"` inline handler** (line 160) is fine functionally but mixed paradigms with the rest of the file's `addEventListener` style (lines 200–213). Minor.
- **AI analysis box** updates with `innerHTML` (lines 361–366) including user-derived text from Claude — if Claude returns angle brackets in the description, they render as markup. Sanitize or use `textContent` for the description text.
- **Modal-free destructive ops.** "Reset" (line 87) discards all edits with no confirm. "Save all" (line 145) writes to Sheets with no review step. Not strictly a11y but a UX/safety issue.

## Visual / brand alignment with index.html

It does **not** feel like the same product.

- `index.html` uses a hand-built editorial system: bone background `#EAE6DE`, ink-blue accent `#1A2640`, Fraunces (serif display) + Inter (body), 12px+ type, generous whitespace, hairline borders, no icon font. `photo_manager.html` uses: undeclared chat-host tokens, `font-family:var(--font-sans, sans-serif)` (line 7) which falls back to the UA default sans, Tabler icon font everywhere, 10–12px copy throughout, and a bright `#2563eb` blue accent (line 47) that has no relationship to Sveta's ink-blue `#1A2640`. The "active-tool" border `#3b82f6` (line 49) is a third, unrelated blue.
- The Tabler icon font (line 174) loaded from jsDelivr is fine functionally but stylistically reads as "developer dashboard," not "painter's gallery tool." Sveta's site uses zero icon fonts — an SVG-light approach would match better, or just type-led labels.
- Card-title pattern (`text-transform:uppercase;letter-spacing:.05em;font-size:11px`, line 13) is a SaaS-admin convention; `index.html` has nothing like it. The portfolio side uses Fraunces titles and quiet hairlines.
- Buttons: `primary` class (line 45) inverts text-on-background which would be black-on-bone (good — matches `index.html`'s `.btn` style), but the `accent` class is `#2563eb` (line 47) which clashes hard with the bone palette and Sveta's ink-blue.
- The drop zone illustration is a Tabler upload glyph; on `index.html` Sveta uses a hand-drawn SVG dot-circle favicon. A small custom upload mark (or no glyph, just type) would fit.

**One-paragraph fix sketch:** import Fraunces + Inter (or rely on system fallback for tooling), replace all tokens with `index.html`'s palette, drop the `accent` blue and use ink-blue `#1A2640` for the single primary CTA only, remove `text-transform:uppercase` card titles in favor of Fraunces h2s at 1rem, swap the Tabler icon font for ~6 inline SVGs (upload, check, x, refresh, crop, download), and bump base type to 13–14px to match the portfolio's restrained-but-readable feel.

## Out-of-scope concerns

- **CDN dependencies break offline.** Tabler icons (line 174) and Google GSI (line 175) are remote. On `file://` with no network, all icons render as `[?]` placeholder boxes and the entire OAuth flow fails silently (`window.google?.accounts?.oauth2` returns falsy, line 487, and `getAccessToken` resolves `null` without telling the user *why*). The error message "Auth failed" (line 499) is the same whether GSI didn't load, the user denied consent, or the token endpoint returned 4xx. (Google-integration / tester lanes.)
- **Claude API call from a `file://` page will be blocked by CORS** even with a key (no `x-api-key`, no `anthropic-version`, no `anthropic-dangerous-direct-browser-access` header — line 335). Spec already flags this; calling it out because the entire "AI pre-fills metadata" UX is dead. (AI-integration lane.)
- **`canvas.toDataURL` for perspective output** (line 460) hits Safari's ~few-MB string limit on large paintings and produces a lossy PNG round-trip; consider `toBlob` + `URL.createObjectURL`. (Image-processing lane.)
- **No image-size guard.** A 50MP HEIC will OOM the canvas on iPad without warning (line 223 reads any file). (Tester lane.)
- **`drag-over` styling** is added on `dragover` and removed on `dragleave` (lines 201–202), but `dragleave` fires when the cursor enters a child element — the drop zone flickers. Should also clear on `drop` (it does, line 203) and ideally track `relatedTarget`. (Tester lane.)
- **Multiple file drop** is silently dropped except for the first file (line 203: `e.dataTransfer.files[0]`). No message. Sveta might reasonably drag five photos. (Tester lane.)
- **Non-image file drop** is silently ignored (line 224). No error toast. (Tester lane.)
- **OAuth token cached for 58 minutes** (line 488, `3500000` ms) with no refresh path. If Sveta keeps the tab open longer the next save will silently re-prompt — or fail. (Google-integration lane.)
