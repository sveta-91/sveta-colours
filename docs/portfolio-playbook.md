# Portfolio Site Playbook

How to build a single-artist portfolio like sveta-colours.com from scratch.
Distilled from the design-and-debug history of this repo, not theory — every
pitfall called out here cost a debug cycle to find.

---

## Architecture in one paragraph

A **single-file `index.html`** with all CSS and JS inlined. Data lives in a
**Google Sheet** published as CSV; the site fetches the CSV, parses it client-
side, and renders. Images live in **Google Drive** (one public folder) and are
served via `lh3.googleusercontent.com/d/<file_id>=w<width>` thumbnails. Hosting
is **GitHub Pages** on the main branch (custom domain via `CNAME`). All write
operations (publishing a new painting, marking a hero, etc.) happen out-of-band
via a Python CLI (`push_painting.py`) talking to Sheets/Drive via OAuth.

Result: zero servers, zero build step, instant publishing from anywhere with
the Sheet open.

---

## Data layer

### Sheet schema that survived contact

| Column | Purpose | Notes |
| --- | --- | --- |
| `title` | Display title | Required |
| `medium` | "Acrylic", "Watercolor" | Required |
| `widthIn` / `heightIn` | Painting dimensions in inches | **Use these to derive `isLandscape`** — never infer from the photo |
| `price` | Number, USD | Empty = "Price on request" |
| `sold` | `TRUE`/`FALSE` | Renders `.is-sold` class + "Sold" badge |
| `image` | Drive file ID | Just the ID, not full URL |
| `orientation` | `portrait` / `landscape` / `square` | Used for near-square paintings where dimensions tie; derived for the rest |
| `hero` | `TRUE`/`FALSE` | Marks the home-page hero painting |
| `recent` | Integer 1..N | Order in the "Recent" home strip; null = not featured |
| `collections` | Comma-separated tags | Drives `/collections/<slug>` chapters |

**Why `hero` and `recent` are sheet-driven:** the artist controls curation
without touching code. Always keep safety-net constants in the JS
(`HERO_PIN_TITLE`, hand-picked recent list) so an accidental column delete
doesn't break the site.

### CSV cache invalidation

Google's published-CSV is cached at the edge for ~5 minutes. After updating
the sheet, the site won't see changes immediately. Two patterns we tested:

- **Bust on demand**: append `?v=<timestamp>` to the CSV URL. Cheap but
  ineffective — Google ignores arbitrary query params on the publish endpoint.
- **Wait it out**: just refresh in a few minutes. This is what we ship.

### Parsing pattern

`sheetRowToPainting()` reads each column with a `get('column_name')` helper
that's case-insensitive and trims whitespace. Every numeric field goes through
`parseInt` with `isNaN` checks; every boolean uses `yes()` which accepts
`TRUE`, `true`, `yes`, `1`. This sounds paranoid — it's not, because Sheets
exports `TRUE` and humans type `true`.

---

## Layout patterns (and what we tried first that didn't work)

### Masonry grid

CSS Grid with `grid-auto-rows: 8px` and a per-card computed `grid-row: span N`
based on the card's actual rendered height:

```js
function computeMasonry() {
  document.querySelectorAll('.painting-card').forEach(card => {
    const img = card.querySelector('.card-image');
    const info = card.querySelector('.card-info');
    const h = img.getBoundingClientRect().height
            + info.getBoundingClientRect().height;
    const rowGap = 64; // matches grid row-gap
    const rowHeight = 8;
    const span = Math.ceil((h + rowGap) / (rowHeight + rowGap));
    card.style.gridRow = `span ${span}`;
  });
}
```

**Pitfall:** call `computeMasonry()` *after* images load, not just on
`DOMContentLoaded`. The `aspect-ratio` CSS hint gives a placeholder ratio
(e.g. `4/5`); the actual rendered height shifts when the image's natural
aspect (e.g. `0.75`) refines it. We compute on `load` and on `resize`.

### Same-canvas paintings hang at the same on-page scale

Every gallery card locks its rendered aspect to the **physical canvas**
(`widthIn / heightIn`), never to the photo's natural pixel aspect.

**Why:** photos vary in framing — Sveta's photographer step (Grok web
UI) crops to the artwork edge, but the crop isn't pixel-perfect. A
12% photo-aspect spread across the same 16×20" group is normal. If
the card uses the photo aspect, paintings labeled "16 × 20 in" render
at visibly different heights, and the gallery wall reads as
inconsistent. Brainstorm panel converged on this 2026-06-09: 4 of 6
roles picked "force canvas aspect" over "honest photo aspect."

```js
// cardHTML: --ratio is canvas-derived, never refined from photo
<div class="card-image-inner" style="--ratio: ${widthIn}/${heightIn}">
```

The earlier `onLoad` handler that overrode `--ratio` to photo-natural
when off by >5% is **disabled on gallery cards** (still active on the
hero band, where a single large image benefits from edge-to-edge
fill).

Photos whose aspect doesn't quite match canvas show small letterbox
bars in `var(--surface-alt)` — reads as museum-matte, not as dead
space.

### Enforce photo↔canvas match at publish time

`push_painting.py` runs a `check_canvas_aspect()` guard right before
upload:

- Photo within **2%** of canvas aspect: passes as-is.
- 2–5% per-side trim: snap-crops centered to canvas aspect, writes
  to `<name>.canvas.<ext>`, uploads that.
- >5% per-side trim: hard-fails with a "reshoot or re-crop in Grok"
  message. The threshold protects painted pixels — beyond 5% the
  crop is likely to eat into the artwork itself.

Pre-2026-06-09 photos may still have larger drift. They render with
matte letterbox bars until republished.

### Landscape cards have transposed dimensions

A portrait card is 280×480 (image 280×365 + info ~115). A landscape card is
**350×480** — same total height, but the image area is 350×280 (the canvas
rotated 90°, so on-wall scale matches portraits). The 72px difference
between portrait image height and landscape image height is absorbed as
breathing room inside the landscape card.

```html
<div class="painting-card" data-orientation="landscape">…</div>
```

```css
.gallery-grid {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;        /* cards in a row match the tallest */
  gap: 4rem 2.75rem;
}
.painting-card {
  flex: 0 0 280px;
  display: flex;
  flex-direction: column;
}
.painting-card[data-orientation="landscape"] {
  flex: 0 0 350px;
}
.painting-card .card-info {
  margin-top: auto;            /* anchors title at card bottom */
}
```

**Why this works:** `align-items: stretch` makes the landscape card grow
to the row's tallest height. `margin-top: auto` on `.card-info` pushes
the title to the card's bottom, so the extra space sits as breathing
room *between the painting and its title*. Title positions line up
across the row.

**Don't** make landscape span 2 grid columns — that makes it dominate the
row and overstate its visual heft against the same physical canvas size.

**Side effect:** mixed rows fit fewer cards than portrait-only rows
(at 1280px container: 3 cards with a landscape vs. 4 portraits-only).
Accept this — it's the cost of transposed-dimension symmetry.

### Hero painting sizing (the trap)

**Don't** do this:
```css
.hero-image-wrap { width: 100%; max-height: 88vh; }
.hero-image-wrap img { object-fit: cover; }
```
This forces a portrait painting into a landscape band and crops the top and
bottom. We hit this and it looked terrible.

**Do** this:
```css
.hero-image-wrap {
  width: auto;
  height: 72vh;
  max-width: 100%;
  min-height: 380px;
  aspect-ratio: var(--hero-ratio, 4/5);
  margin: 0 auto;
}
.hero-image-wrap img { object-fit: contain; }
```
Then set `--hero-ratio` per-painting from the natural image dimensions.
The hero scales to the painting's actual aspect, never crops.

### Sold paintings

Show them. Don't hide them. The portfolio is a record.

- Add semantic class only: `.painting-card.is-sold`. Use it as a hook, but
  **don't** apply visual dimming (`opacity: 0.7` etc.) — it's inconsistent
  next to portraits where the painting itself is dark.
- Replace price with the text `"Sold"`. No strike-through SVG — they
  rendered as broken lines across the modal in our tests.
- Add a `Commission similar →` mailto button on sold cards so the painting
  still has a path to revenue.

```html
<button class="card-commission"
        onclick="event.stopPropagation();
                 window.location.href='mailto:…?subject=Commission similar to …'">
  Commission similar &rarr;
</button>
```

---

## Interaction patterns

### Custom cursor — scope it tightly

Easy to write a custom cursor that follows the mouse over `.painting-card`
and shows `View →`. Easy to ship a version that **obscures the title and
Commission button below the image**. Don't.

Scope the cursor handlers to `.card-image-inner` only:

```js
document.querySelectorAll('.painting-card .card-image-inner').forEach(t => {
  t.addEventListener('mouseenter', () => {
    cursorEl.textContent = `${resolveTitle(t)} →`;
    showCursor();
  });
  t.addEventListener('mouseleave', hideCursor);
});
```

Other rules:
- Pill shape (`border-radius: 999px`), not a circle — a magnifier shape
  *implies* zoom-on-click and we don't have that.
- `requestAnimationFrame` throttle the `mousemove` handler.
- `pointer-events: none` on the cursor element so it never blocks clicks.

### Focus styles need `transition: none`

`:focus-visible` with a `transition: opacity 0.6s` looks great to a human and
**fails Selenium tests intermittently** because the assertion fires mid-fade.
Add `transition: none !important` to the focus rule. The fade is
imperceptible anyway.

### Hash routing doesn't deep-link on first load

Our site renders SPA-style with `showPage(name)`. Loading `/#works` directly
shows the home page, then the hash handler fires. In Selenium tests, this
means waiting for `.hang-chapter` after `showPage('paintings')`, **not**
just hitting the URL.

### Modal class names

Modal element is `.modal-overlay.open`, not `.modal.open`. We wrote two
verify scripts with the wrong selector before catching it. Pick a class name
and grep for it everywhere.

---

## Verification process

We used Selenium 4.44 in `tools/image_pipeline/.venv` to verify every UI
change. The user does not run a visual review for each iteration — the
verify script either confirms the change or screenshots the failure.

### The standard verify pattern

See `scripts/templates/selenium_verify.py`. The pattern:

1. Open the deployed URL (GitHub Pages auto-deploys on push to main, takes
   ~30s).
2. Wait for the SPA root element.
3. Navigate via `driver.execute_script("showPage('paintings');")` — don't
   trust hash routing on first load.
4. Wait for the page-specific selector (`.hang-chapter`, `.modal-overlay.open`).
5. Sleep ~2s for masonry compute + image refinement.
6. Use `execute_script` to dump geometry (`getBoundingClientRect`) for every
   card — much faster than per-element `find_element` calls.
7. Save a screenshot to `/tmp/hover-test/<batch>-<n>.png`.

### Gotchas

- **`ActionChains.move_to_element_with_offset`** offsets from the element
  *center*, not top-left. To sample the bottom of an element at `frac=0.9`,
  use `offset_y = int((0.9 - 0.5) * box.h)`.
- **Hover state requires the mouse to actually move.** Selenium's
  `move_to_element` to the same target twice doesn't re-fire `mouseenter`.
  Move to `h1` first to reset, then to the target.
- **`save_screenshot` captures the viewport only**, not the full page. For
  full-page screenshots, use Chrome's DevTools protocol via
  `driver.execute_cdp_cmd("Page.captureScreenshot", {"captureBeyondViewport": True})`.

---

## Pitfalls that cost real time

A flat list, ordered by how long the fix took.

1. **Hero painting cropped as landscape band.** Cause: `width:100%` +
   `object-fit: cover`. Fix: see "Hero painting sizing" above. ~90 min.
2. **Sold opacity made cards look broken.** Some sold cards have dark
   paintings; the dimming was invisible. Others had pale paintings; the
   dimming was obvious. Inconsistent → looked like a bug. Fix: drop
   visual dimming, keep semantic class only. ~45 min.
3. **Modal "Sold" treatment.** Showed price + strike-through SVG that
   rendered as a line across the entire modal width. Fix: replace price
   with `"Sold"` text. ~30 min.
4. **Painting orientation inferred from photo aspect.** Phone photos of
   portrait paintings include landscape context around them. Don't guess;
   read `widthIn` / `heightIn` from the sheet. ~30 min.
5. **Custom cursor obscuring Commission button + title.** First fix
   carved out one element; second fix rescoped to image only. The
   broader fix is the right one. ~20 min.
6. **Masonry spans recompute too early.** Image natural aspect refines
   after `load`, changing card height. Compute on `load`, not
   `DOMContentLoaded`. ~20 min.
7. **CSV publish cache.** Sheet updated, site didn't change. Just waits
   the cache out. Set expectations. ~10 min.

---

## Reusable scripts

- `scripts/templates/selenium_verify.py` — generic UI verify pattern.
  Copy, rename to `/tmp/<intent>_verify.py`, edit the assertions.
- `tools/image_pipeline/push_painting.py` — publish-a-painting CLI.
  Loads OAuth creds, uploads to Drive, upserts sheet row, preserves
  `hero` and `recent` columns when updating.

## Process patterns worth keeping

- **Per-painting publishing > batch pipeline.** Started with a 9-stage
  batch pipeline (Drive inbox → straighten → crop → upload). Quit that
  for a per-painting CLI driven by Grok web UI (perspective+crop) +
  `trim_borders.py` (strip Grok padding) + `push_painting.py`. Faster
  feedback, fewer edge cases. Batch pipeline still in repo as
  `run_pipeline.py` for reference.
- **Brainstorm panels for design refinement.** A 6-role panel (designer,
  frontend, tester, art collector, buyer advocate, marketer) running in
  swarm → director-led mode generated the T1/T2/T3 polish lists. See
  `~/.claude/skills/brainstorm-panel/`. Worth the round-trip for any
  design-bearing batch with >3 simultaneous changes.
- **Sheet-driven config wins.** Every config knob we moved from JS
  constants to sheet columns paid off the first time the artist edited
  it solo. `hero`, `recent`, `orientation`, `collections` — all
  sheet-driven, all fallback-protected with constants.
