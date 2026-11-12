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

That was the starting point, not a rule: a Spring Boot backend is now on the
table. See "Extending the site" at the end.

---

## Data layer

### Sheet schema that survived contact

| Column | Purpose | Notes |
| --- | --- | --- |
| `title` | Display title | Required |
| `medium` | "Acrylic", "Watercolor" | Required |
| `widthIn` / `heightIn` | Painting dimensions in inches | **Use these to derive `isLandscape`** — never infer from the photo |
| `price` | Number, CAD | Empty = "Price on request" |
| `sold` | `TRUE`/`FALSE` | Renders `.is-sold` class + "Sold" badge |
| `image` | Drive file ID | Just the ID, not full URL |
| `orientation` | `portrait` / `landscape` / `square` | Used for near-square paintings where dimensions tie; derived for the rest |
| `hero` | `TRUE`/`FALSE` | Marks the home-page hero painting |
| `recent` | Integer 1..N | Order in the "Recent" home strip; null = not featured |
| `cover` | Collection name(s), comma-separated | This painting is the collection's button picture (rail tabs + Collections cards). Blank → first available in order |
| `order` | `3` or `Forest:2,Seasons:5` | Position on collection pages; numbered first, ties and blanks keep sheet order (`collectionItems()`) |
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

### Long texts in plain .txt files (`texts/`)

Sveta's choice (2026-11-09), after rejecting a key/text Google Sheet: the long
texts live in plain text files she can read and edit as text.

- `texts/home.txt` (`[headline]`, `[under headline]`, `[from the studio]`,
  `[on painting]` — the last shows on Home and About), `texts/about.txt`
  (`[about]`), `texts/faq.txt` (`[question]`/`[answer]` pairs, page order).
- A `[label]` line opens a block (case-insensitive). Blank line = new paragraph,
  single line break = `<br>` (under-headline: one `<span>` per line). Text is
  escaped — HTML typed in a file shows as typed.
- Missing/empty block → that spot is hidden. Unknown label, text before the first
  label, half a FAQ pair → skipped with a `console.warn` naming it.
- No copy of these texts in `index.html`. `loadTexts()` starts at boot, runs
  parallel to the sheet fetch, capped at 5s overall.
- Short copy (page titles/subtitles, section headings, contact, footer,
  postcards line, menu, buttons) stays in code — her call ("only the long texts").

**Step 2 (live): Google Docs are the source, `texts/` is the backup.**

- Three Google Docs in the sveta.mordkovich Drive (the .txt originals are in sveta.colours) — `home`, `about`, `faq` — made
  with "Open with Google Docs" from the .txt files, then File → Share → Publish
  to web. `TEXT_DOCS` in `index.html` holds the three `…/d/e/2PACX-…/pub` links.
  Same `[label]` format; each Doc paragraph = one line, empty paragraph = blank line
  (`docPageToText()` reads `#contents` p/li/h*).
- Per file: Doc first (2.5s timeout) → `texts/<name>.txt` → nothing.
  `TEXTS_SOURCE` records which one was used (`doc`/`backup`/`none`).
- **Why Docs, not the .txt in Drive:** Drive's download service answers a
  browser's cross-site read with 403 for every link form (`uc?export=download`,
  `drive.usercontent.google.com/download`, …), even when shared "anyone with the
  link". Only "Publish to web" copies (Sheets CSV, Docs) send
  `access-control-allow-origin`. Docs ignores `?output=txt` — the page is HTML,
  hence the paragraph extraction. Also: a .txt can't be edited in Drive's browser
  UI, a Doc can.
- **Publish lag:** an edit reaches the `/pub` link at once for some Google
  servers, but the copy a browser gets lagged ~5–6 min (measured 2026-11-09).
- The `texts/` backup is not synced automatically — after Sveta changes a Doc
  for good, copy the text into `texts/<name>.txt` too (or the backup shows old
  text if Google is down).
- Docs autocorrect may turn `'` into `’` and `--` into `—` as she types. Fine.
- Editing: open the Doc in Drive, change it; no Publish step needed again.
  Text changes go through the `sveta-writing` skill like any site copy.

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

**Do** this (home design 1, "Statement", 2026-11): the hero is a two-column
grid — the tagline as the `h1` on the left, the painting on the right.
```css
.hero-image-wrap {
  width: min(600px, 33vw, calc((100vh - 13rem) * var(--hero-ratio, 4/5)));
  max-width: 100%;
  aspect-ratio: var(--hero-ratio, 4/5);
}
.hero-image-wrap img { object-fit: contain; }
```
Then set `--hero-ratio` per-painting from the natural image dimensions.
The hero scales to the painting's actual aspect, never crops, and the height
term keeps painting + caption above the fold on short laptop windows
(verified 1280×800 to 1920×1080). At ≤1024px the grid stacks — words first,
then the painting at `min(100%, 72vh × ratio)`. The caption is
title · medium · size (inches). With the desktop rail the home page hides
its Collections block (the rail lists them); below 1025px it stays.

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

## Type (fonts)

Two fonts per page, chosen by Sveta on 2026-11-09 after a six-font prototype:

- **Marcellus** (`--font-brand`, one weight, upright only): the site name, the
  home headline and its two lede lines, page titles, section headings, FAQ
  questions, collection names (rail tabs and home cards), contact heading.
- **Inter** (`--font-body`): everything else: body text, menu, counts, notes,
  prices, buttons. Notes that used to be Fraunces italic are plain Inter now.
- **Fraunces is gone** (not even downloaded). `--font-display` survives only
  as an alias of `--font-brand`. Painting titles (cards, popup) are
  Marcellus; sizes, prices, "Commission similar", popup links are Inter.
- No italic, no bold, no flourish under the name. Marcellus has no italic or
  bold, so the browser would fake both badly.
- Name size: 28px in the desktop rail, 22.4px in the phone top bar. Keep it
  smaller than the page headline.
- Check: every visible text node on every page and in the popup computes to
  Marcellus or Inter, `font-style: normal`.

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

### Routing: every page has a hash URL

`showPage(page)` pushes a hash; a single `popstate` listener drives both pages
and the painting modal; boot reads the hash before the first `render()`.

| Page key | URL |
| --- | --- |
| `home` | no hash |
| `recent` | `#recent` — the sheet's `recent` paintings (positions 1–6), hero included |
| `paintings` | `#paintings` (legacy `#works` still resolves) |
| `collections` | `#collections` |
| `collection:<Name>` | `#collections/<slug>` (`Big City` → `big-city`) |
| `about`, `faq`, `postcards`, `past` | `#<page>` |
| painting modal | `#painting/<id>` — layered on top of whatever page is showing |

Traps, each one hit while building it:

- **Unknown hashes must be ignored, not routed home.** `pageFromHash()` returns
  `null` for `#main` (the skip link) and anything else it doesn't know.
- **Never `history.back()` and then `pushState` in the same tick.** `back()` is
  async and lands *after* the push, undoing it. The modal's collection chips use
  `closeModal(true)` + `showPage(…, { replace: true })` to swap the modal entry
  for the collection page instead.
- **The collection rail is a `div role="navigation"`, not `<nav>`.** The
  global `nav { position: fixed }` rule pins any `<nav>` over the header.
- An unknown collection slug falls back to the Collections overview, so a
  renamed collection never 404s into a blank page.

Primary nav (Sveta's order, 2026-11-08): Home · Recent · Collections (+ picture
tabs) · All Paintings · About · FAQ. Postcards is off the nav; `#postcards`
and `#past` stay reachable by URL and from the footer. Layout chosen 2026-11-08 from a 3-design
canvas (option D: catalogue rail + picture tabs), and it must follow the
prototype site-wide:

- **Desktop ≥1025px:** the primary `<nav>` is a fixed 272px left column on
  every page: logo, the links, one picture tab per collection nested under
  Collections (`#navCollections`, filled by `renderNavCollections()` on every
  `render()`), and a studio line at the bottom. `body` gets
  `padding-left: var(--rail-w)`. The rail must fit a 1280×900 window without
  scrolling; check this after adding a collection.
- **769–1024px:** top bar again. Collections pages show the picture tabs
  in-page (`.coll-rail`) to the left of the paintings.
- **≤768px:** hamburger menu (links only). Collections pages show the tabs as
  a 2×2 grid above the paintings, and the overview hides its text cards
  because the grid already is the overview.

A picture tab shows the collection's first unsold painting, its name and its
count. The active tab gets a paper fill and an ink outline.

The rail costs ~270px of width, so at laptop widths three 280/350px cards no
longer fit in a mixed portrait+landscape row. Between 1025 and 1599px, every
card is scaled by the same 0.86 (240/300px), which keeps the
portrait↔landscape scale rule intact. Don't fix this by shrinking only one
orientation.

Local Selenium runs can show every Drive image broken: `lh3.googleusercontent.com`
sometimes refuses requests from a `localhost` page. Confirm image loading
against the deployed site before calling it a bug.

### AI chat needs a proxy, never a key in the page

The site has no server, so an OpenAI key in `index.html` would be public.
The chat widget (`initChat()`) posts to a Cloudflare Worker (`worker/`) that
holds the key, builds the system prompt from studio facts plus the live sheet
CSV, and forwards to OpenAI. The conversation lives in the visitor's
`sessionStorage` and is sent whole each turn, and the Worker stores nothing.
`CHAT_ENDPOINT_PROD = ''` hides the widget completely, which is the safe default
until the Worker is deployed. On localhost, `?chat=<url>` points the widget at a
test Worker. The panel is a `div`, not a `<section>`, because the global
`section` padding blew up its layout.

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
3. Load the page's URL directly (`/#collections/flowers`), or drive it with
   `showPage('paintings')`. Load `about:blank` first — a `get()` that only
   changes the hash is a same-document navigation, not a reload.
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
- **The gallery boots asynchronously.** `loadFromSheet()` fetches the published
  CSV after first paint, so the page renders "Loading paintings…" before any
  card exists. Waiting on `.painting-card` happens to work only because the
  card can't appear until the fetch resolves — but any wait keyed to a
  *non-card* selector (chapter headings, the collection header, the hero) can
  fire against the pre-data DOM and measure nothing. Poll until the loading
  text is gone, then wait for the selector.
- **Favicons are not verifiable by Selenium at all.** They live in the browser
  tab — browser chrome, not the viewport — so no screenshot can show one. Use
  `curl` against `/icons/favicon*.{svg,png}` and grep the served HTML for the
  `<link rel="…icon">` tags instead.

---

## Spec-vs-renderer drift

The most expensive class of bug here isn't a crash — it's a rule agreed in one
place and silently contradicted in another, where nothing fails and nobody
looks.

**The instance (2026-08-23).** The voice spec locks painting descriptions at
2–4 sentences, three being the sweet spot, with the closing sentence carrying
the hedge. The modal's `truncateDescription()` returned *"first 2 sentences OR
240 chars, whichever is **shorter**"*. The sentence rule therefore won every
time, and the modal amputated the last sentence of **20 of the 21 live
descriptions** — precisely the sentence the spec cares most about. Only the one
2-sentence description ever displayed in full. It shipped that way for months
and was found by the artist noticing an ellipsis, not by any check.

**Why nothing caught it.** `audit_gallery.py` validates the *data* and the data
was fine. The renderer was fine in isolation too. The defect existed only in the
disagreement between them, and no artifact owned that relationship.

**The rule.** When a spec fixes a shape for content — length, sentence count,
structure — the renderer must not impose a second, tighter limit of its own. If
it needs a cap at all, make it a guard set well outside the spec's range
(`DESC_MAX_CHARS = 400` against a 60-word spec) and say so in a comment that
names the spec. Any cap inside the spec's range is a silent content editor.

**Check it this way:** run the spec's own bounds against live data through the
renderer's function, not by reading the code. Twenty of twenty-one is obvious in
one line of output and invisible on inspection.

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

---

## Extending the site (decided 2026-11-11)

**The constraint changed.** "No server, no build step" was how the site
happened, not a value. A **Java / Spring Boot backend is acceptable**, and so
is the extra admin that comes with it. This section maps the features already
queued up (the external review in `docs/review.txt`) onto the code: where each
one attaches and what it needs.

### What already exists: three half-backends

The site already needs a server and has grown one in three places:

| Piece | Where | Job | State |
| --- | --- | --- | --- |
| Inquiry form | `inquiry/Code.gs` (Apps Script) | form → private sheet + email | not wired: `FORM_ENDPOINT = ''` |
| AI chat proxy | `worker/` (Cloudflare Worker) | holds the OpenAI key | not wired: `CHAT_ENDPOINT_PROD = ''` |
| Publishing | `tools/image_pipeline/push_painting.py` | Drive upload + sheet upsert | live |

**Decision:** one Spring Boot app takes over the first two and anything new.
`push_painting.py` stays a local CLI (Grok is web-only anyway). It gains one
call to the backend to refresh its cache after a push. Don't build a third and
fourth serverless piece for payments and images.

### Two defects to fix before building on them

1. **Painting identity is the sheet row number.** `sheetRowToPainting()` sets
   `id: index + 1`, so `#painting/10` means "the 10th row right now".
   Inserting or reordering a row silently retargets every shared link. Real
   URLs, a Stripe order and a sitemap all need a **stable key**. Add a `slug`
   column (`noise-and-silence`) that `push_painting.py` writes once and never
   changes, and keep accepting `#painting/N` as a legacy redirect.
2. **Inquiry contract mismatch.** `Code.gs` says the site posts
   `Content-Type: text/plain` to avoid a CORS preflight (Apps Script can't answer
   `OPTIONS`). `sendInquiry()` posts `application/json`, so the form would
   fail the moment `FORM_ENDPOINT` is set. `Code.gs` also checks `elapsedMs`,
   which the form never sends, and points to an `inquiry/README.md` that
   doesn't exist. This is the same class of bug as "Spec-vs-renderer drift"
   above. It's moot if the Spring app replaces Apps Script, but fix it if Apps
   Script ships first.

### Target shape

```
Cloudflare (DNS + cache)
   └── Spring Boot app (one service)
         GET  /                       index.html shell, per-route <head> injected
         GET  /paintings/{slug}       same shell, painting's title/description/og:image/JSON-LD
         GET  /collections/{slug}     same shell
         GET  /sitemap.xml, /robots.txt
         GET  /img/{slug}-{w}.webp    resized from the Drive original, cached
         POST /api/inquiry            replaces inquiry/Code.gs
         POST /api/chat               replaces worker/
         POST /api/checkout/{slug}    Stripe Checkout session (holds the painting)
         POST /api/stripe/webhook     paid → mark sold in the sheet
         POST /api/admin/refresh      called by push_painting.py
   reads: Paintings sheet (Sheets API, cached in memory), Drive originals
```

**The key choice is a shell plus injected `<head>`, not a rewrite.**
`index.html` stays the client app it is now. The server only fills `<title>`,
meta, `og:*`, `JSON-LD` and a `<noscript>` text block for the route it serves.
Link previews (Telegram, WhatsApp, Facebook) don't run JS, so the **head** is
the part that must come from the server. Google does run JS, but slowly.
Server-rendering the whole UI (Thymeleaf etc.) is not needed for these goals.

**The sheet stays the source of truth.** "Sheet-driven config wins" still
holds: she edits it solo. The backend reads it through the Sheets API and
caches it. That also removes the published-CSV cache lag (see "CSV cache
invalidation"), because `push_painting.py` calls `/api/admin/refresh`. Move to
a database only when the backend writes state the sheet shouldn't hold (orders,
inquiries). Even then, the paintings stay in the sheet.

**Hosting** is outside GitHub Pages. GitHub Pages can't run Java. Any small
JVM host behind the existing Cloudflare DNS works. Cache the HTML and images at
Cloudflare so a backend restart doesn't take the gallery down. Keep the GitHub
Pages deploy as the fallback until the backend has run cleanly for a few weeks.

### Feature → seam map

| Feature (from review) | Seam in code today | Needs | Effort |
| --- | --- | --- | --- |
| On-site inquiry form | `inquiryHTML()`, `sendInquiry()`, `FORM_ENDPOINT` | `/api/inquiry` | small: the UI exists |
| Real URLs per painting/collection | router: `SIMPLE_ROUTES`, `hashForPage()`, `pageFromHash()`, `showPage()`, `hydrateFromHash()`, `closeModal()` | `slug` column; `pushState('/paintings/slug')` instead of hashes; server routes return the shell; old hashes redirect | **medium, the largest one.** Every `#…` href and the modal's open/close history logic changes together |
| Per-page title/description/og:image | `pageTitleFor()` sets the tab title client-side; static `<meta og:*>` in `<head>` | server-side head injection | small once routes exist |
| Structured data + sitemap | `injectPaintingsLD()` (client-side, first 12 only) | server emits per-page `VisualArtwork`/`Offer`; `/sitemap.xml` from the sheet | small |
| Web images (WebP, sizes, zoom) | `driveImageURL()`, `paintingImage()` srcset, `collectionThumb()`, hard-coded `lh3…` URL in `renderHero()` | `/img/{slug}-{w}.webp`; swap the URL builder in one place | small in code, medium in ops (cache, disk) |
| Buy / Reserve | `cardHTML()`, `openModal()` CTA, `formatPrice()` | Checkout + webhook + sold-sync | medium. See below |
| AI chat | `initChat()`, `CHAT_ENDPOINT_PROD` | `/api/chat` (port of `worker/chat-worker.js`) | small |
| Prices by size, shipping | `formatPrice()`, `SHIPPING_NOTE`, sheet `price` | data only, no code | none |

### The trap with selling originals

Every painting is **one unit**. A shop plugin assumes stock. The expert
question is: what happens when two people pay for the same original within the
same minute? A static Payment Link per painting can't prevent that, and a
manual "sold" flip in the sheet is too slow. **Rule:** creating a Checkout
session puts a short hold on the slug (e.g. 30 min, in memory or a tiny table).
A second buyer sees "Reserved". The paid webhook writes `sold=yes` to the sheet.
An expired session releases the hold. Until that exists, keep "Reserve" as an
inquiry, not a payment.

### Order of work

1. `slug` column + `push_painting.py` writes it (no visible change).
2. Spring app with `/api/inquiry` + `/api/chat` + `/api/admin/refresh`, deployed;
   the site still on GitHub Pages and pointing at it.
3. Routes + head injection + sitemap; Cloudflare switched to the app;
   hash URLs redirect.
4. Images.
5. Checkout with holds.

Each step ships alone, and the earlier ones are useful without the later ones.

### How to check a change against this section

- `scripts/verify_site.py` and `scripts/verify_collection_order.py` still pass.
- For each new route: `curl -s <url> | grep og:image` returns the painting's
  own image. Run it with curl, not a browser, because previews don't run JS.
- Every legacy `#painting/N` and `#collections/x` link from before the switch
  still lands on the same painting or collection.
