# Gallery / catalog grid — R1.5 plan (post-inventory)

> **Status:** updated 2026-06-08 after the panel was re-run against the actual inventory data (17 paintings, 2 size classes, all available priced $400, 6 sold / 11 available, mixed portrait/landscape, /postcards currently empty). The R1 synthesis was based on assumptions that broke against the real data; this document supersedes it. Where R1 has been overruled, it is called out inline.

**Target:** the `.gallery-grid` + `cardHTML()` system in `index.html`, in all 5 contexts (home featured, /works, /postcards, /collection/{name}, /past).

**The bar:** "A visitor scanning the grid feels each painting's distinctness, scrolls happily, and naturally lands on one to open or inquire about — without ever feeling they're shopping a feed."

**Out of scope:** modal internals, hero, nav layout, painting content/data, photo-manager backend.

---

## Inventory ground truth

- **17 paintings total** (not "50+").
- **11 available + 6 sold.**
- **2 size classes:** 16×20 in (12, ~70%) and 11×14 in (5, ~30%).
- **All available paintings priced identically at $400.** Sold pieces have no listed price.
- **Mix of portrait and landscape orientations** → ~4 discrete card shapes (16×20p, 20×16l, 11×14p, 14×11l).
- **All paintings acrylic on canvas.** No watercolors in inventory → **/postcards renders empty.**
- **5 collections:** Seasons, Flowers, Big City, Forest (and others), 1–2 tags per painting.
- **Year data sparse:** only 4 of 17 paintings have years (2024 or 2025).
- **`dominantColor` column** referenced in code but not populated in the sheet — color-aware curation is data-gated.

---

## Panel

Six roles, swarm → director-led, 2 rounds run (R1 cold + R1.5 with inventory facts). 41 R1 notes → 6 revised critiques → user decisions on 4 contested calls.

| Role | Lane |
|---|---|
| Gallery curator (director) | Editorial spine; final synthesis |
| Visual designer | Type, rhythm, breathing, image:meta balance |
| Frontend developer | Masonry mechanics, CLS, lazy-load, perf, mobile |
| Collector / buyer advocate | "Can I find the one for me without giving up?" |
| Adversarial UX skeptic | Attacks premises (catalog №, masonry, filters, hover) |
| UX/UI design auditor | Heuristics, affordance, a11y, cross-context consistency |

---

## Conflicts resolved (post-inventory; bold = changed from R1)

### 1. Catalog № — **DELETE everywhere** ⚠️ *overrides R1*

- **R1 stance:** keep, demote to corner overlay, hide on home + /past.
- **R1.5 stance:** Buyer + Skeptic + Auditor strengthened delete (small N makes № navigationally useless and editorially overreaching — "like numbering chairs in your living room"); Curator + Designer still argued keep.
- **User call:** delete everywhere.
- **Resolution:** remove `.card-number` CSS block (`index.html:441-451`); remove `${num}` slot from `cardHTML` (`:1932-1937`); remove `heroNum` rendering from `renderHero` (~`:2270-2271`); replace cursor `View №XX` labels (~`:2928-2934`) with `View "${title}"`. Keep `catalogNumber` as data only — surface in modal metadata if useful.

### 2. Price on card — **keep, move below meta** ⚠️ *user overrode R1.5 push*

- **R1.5 consensus** (Buyer + Skeptic + Designer + Auditor): drop price from card, state once in /works header (`All paintings $400 · 16×20 or 11×14 in · acrylic on canvas`). 11 identical $400 cards reads as commodity.
- **User call:** keep both meta AND price on card; lean on typography (R1 position).
- **Resolution:** stay with R1 — `.card-info { display: block }`, `.card-price { display: block; margin-top: 0.4rem; font-style: italic; color: var(--text-secondary); }`. `Sold` renders in the same slot via `.card-price.card-price--sold`. Remove inline `soldTag` from `.card-title`. **Designer's R1.5 refinement applies:** drop price weight (`font-size: 0.88rem`, color `var(--text-faint)`) since uniform price = non-information; the slot exists for layout consistency, not signal.

### 3. Masonry — **keep + Designer's interleave + 85% inset for 11×14** (tentative; revisable) ⚠️ *new from R1.5*

- **R1 stance:** keep masonry, fix FOUC.
- **R1.5:** Skeptic strengthened the uniform-tile case ("Tetris autopilot — 4 shapes never quite resolve"); Designer proposed saving masonry by interleaving portrait/landscape order and inset 11×14 cards to 85% column width (real-world scale cue without a chip); Curator + Dev held the line on keeping real proportions.
- **User call:** Designer's variant for now, not final.
- **Resolution:**
  - Keep masonry. Address FOUC via the R1 fix (seed `grid-row: span N` at render time from `aspectRatioFor(p)`; start in `grid-auto-rows: 8px` from page-load; no `.masonry-ready` swap). With 4 known aspect ratios, the span values can be a small lookup table.
  - At the data layer (where `PAINTINGS` is sorted before rendering), interleave portrait/landscape so no more than 2 consecutive cards share orientation. Where Sveta's curatorial order matters, that wins — but as a default ordering function, interleave.
  - Render 11×14 cards at ~85% column width, centered, via `.painting-card[data-size="small"] .card-image { width: 85%; margin: 0 auto; }`. Title / meta / price stay at full column width. Subtle inset = real-world scale cue without a `S/M` chip.
  - **Revisable:** if the result still reads as "uncanny valley" (Skeptic's risk), the fallback is the chapter-by-size strict-grid (move #2 below).

### 4. Sort / filter UI on /works — **no UI; chapter by size** (revised from chapter-by-year)

- **R1.5:** Buyer dropped their own R1 sort-strip ask ("11 cards, all $400, sort UI is performative at this scale"); Curator's year-chapter died (sparse year data); replaced with chapter-by-size (Large 16×20 / Small 11×14). Auditor adds: this also gives /works the `<h2>` substructure WCAG 2.4.10 wants.
- **Resolution:** in `renderPaintings` (~`:2498`), partition `works` by size class; render two `<section aria-labelledby="grp-N"><h2 id="grp-N">Large works — 16 × 20 in</h2><div class="gallery-grid">…</div></section>` blocks. Visually subtle (display-italic heading, hairline above, ~3rem margin-top). Same `cardHTML` in both grids.

### 5. Card affordance — **`<button>` → `<a href="#painting/${id}">`** (unchanged)

- Unanimous; Buyer's load-bearing ask.
- **Resolution:** render the card as `<a href="#painting/${p.id}" onclick="event.preventDefault(); openModal(${p.id});">`. Push history state in `openModal`. Add `popstate` listener.
- **Popstate semantics (proposed default):**
  - Click card → `history.pushState({ paintingId: id }, '', '#painting/' + id)`; open modal.
  - Close modal → `history.back()` if `state.paintingId` matches; else `history.replaceState({}, '', window.location.pathname + window.location.search)`.
  - `popstate` → if hash matches painting id, open that modal; if no painting hash, close any open modal.
  - Direct-link arrival (`/#painting/5`) → opens modal on initial render; closing replaces hash without leaving the site.

### 6. Home featured grid — **cut from 8 to 6** ⚠️ *new from R1.5*

- **R1.5:** Skeptic flagged 8-of-17 (47%) reads as preview, not curation. Curator's softer move: drop to 6 so /works has more reveal value; Skeptic's harder move: drop the dual-grid pretense entirely.
- **User call:** 6.
- **Resolution:** `selectFeatured(PAINTINGS, 6, …)` at `:2312`. Also drop the padding branch in `selectFeatured` (~`:2196-2199`) — available pool is only 11; padding with sold pieces fires too often. Cap at `Math.min(6, availablePool.length)`.

### 7. /postcards page — **hide nav entry while empty** ⚠️ *new from R1.5*

- **R1.5:** Curator + Buyer + Skeptic + Auditor all flagged this as Nielsen #1/#5 failure — nav entry resolves to an apologetic empty state with no recovery path.
- **Resolution:** in the nav render (~`:1529, :2857`), gate the Postcards `<li>` on `PAINTINGS.some(p => p.medium === 'watercolor' || p.medium === 'watercolour' || (p.collections||[]).includes('postcards'))`. Keep the route reachable by URL for when inventory exists. One conditional; removes a dead end.

---

## Tier 1 — load-bearing (revised)

| # | Move | File:line | Driver |
|---|---|---|---|
| T1.1 | Convert card from `<button>` to `<a href="#painting/${p.id}">`; intercept click → `openModal`; push history state; handle `popstate` | `index.html:1936`, `openModal` ~1957, new `popstate` listener | Buyer's advocate (unanimous) |
| T1.2 | **DELETE catalog №** — CSS block at `:441-451`, `num` slot in `cardHTML` at `:1932-1937`, `heroNum` in `renderHero` at `:2270-2271`, cursor labels at `:2928-2934` (replace `№ XX` with title). Keep `catalogNumber` as data only | multi-site | Skeptic + Buyer + user call |
| T1.3 | Chapter /works by size: partition `works` into 16×20 and 11×14 in `renderPaintings`; render as two `<section><h2>…</h2><div class="gallery-grid">…</div></section>` blocks. Heading: display italic, ~3rem margin-top, hairline above | `index.html:2498-2524`, new `.hang-chapter` CSS | Curator + Auditor (WCAG 2.4.10) |
| T1.4 | Price below meta in card-info: `.card-info { display: block }`; `.card-price { display: block; margin-top: 0.4rem; font-size: 0.88rem; font-style: italic; color: var(--text-faint); }`. `Sold` in same slot via `.card-price.card-price--sold` (existing `.sold-stroke`). Remove inline `soldTag` from `.card-title` | `index.html:479-516, 1925-1948` | Visual designer + Buyer + Auditor |
| T1.5 | Hide Postcards nav entry while no watercolor inventory: gate `<li>` on `PAINTINGS.some(...)` | `index.html:1529, 2857` | Curator + Buyer + Skeptic + Auditor |
| T1.6 | Home featured count: 8 → 6 at `:2312`; drop sold-padding branch in `selectFeatured` at `:2196-2199`; cap at `Math.min(6, availablePool.length)` | `index.html:2196-2199, 2312` | Curator + user call |
| T1.7 | Seed `grid-row: span N` at render time from `aspectRatioFor(p)`; start `.gallery-grid` in `grid-auto-rows: 8px` from page-load; remove `.masonry-ready` swap. With 4 known aspect ratios, the span values are a small lookup table | `index.html:415, 419`, `cardHTML:1935`, JS masonry path | Frontend dev (resolves Skeptic's FOUC concern) |
| T1.8 | Add `width`/`height` attrs to `<img>` from `widthIn`/`heightIn`; gate `onload` ratio-rewrite to `requestAnimationFrame` (low priority cleanup — `widthIn`/`heightIn` populated for all 17, so `onload` is effectively dead code) | `index.html:1900-1915` | Frontend dev |
| T1.9 | `srcset`/`sizes` via `driveImageURL()` using `lh3.googleusercontent.com` `=w400/=w800/=w1400` variants; `sizes="(max-width:600px) 100vw, (max-width:1024px) 50vw, 33vw"` | `driveImageURL()` + `paintingImage()` ~`:1915` | Frontend dev |
| T1.10 | Designer's 11×14 inset: `.painting-card[data-size="small"] .card-image { width: 85%; margin: 0 auto; }`; add `data-size` attribute to the card based on `p.widthIn × p.heightIn` | `cardHTML:1935`, new CSS rule near `.card-image` | Designer (tentative — revisable per #3) |
| T1.11 | Interleave portrait/landscape ordering in `PAINTINGS` before render (no more than 2 consecutive same-orientation cards) — implement as a stable post-load reorder where curatorial order isn't already set | data layer in `loadFromSheet`/render | Designer (tentative) |

## Tier 2 — high-value polish

| # | Move | Driver |
|---|---|---|
| T2.1 | Hover refactor: remove `transform: scale(1.02)` at `:477`; add `.painting-card:hover .card-image { transform: translateY(-2px); }`; title color/underline on hover via `.painting-card:hover .card-title { color: var(--accent); text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 0.2em; }` | Designer + Skeptic + Auditor |
| T2.2 | Grid gap: `gap: 4rem 2.75rem` desktop (less airy than R1 5rem at 17-card scale), `3.25rem 2rem` at 1024px, `3.5rem 0` at 600px | Designer R1.5 |
| T2.3 | `.card-info { padding: 1.35rem 0 0 }`; add `.card-image { border-bottom: 1px solid var(--hairline); padding-bottom: 0.9rem; }` — caption-shelf hairline, also a chromatic-insulation move | Designer |
| T2.4 | `.card-image-inner` background → `var(--bg)`; keep `--surface-alt` only on `<img>` as pre-load placeholder (upgraded to [med] in R1.5 — wider color variation than R1 assumed) | Designer R1.5 |
| T2.5 | `aria-label` includes status: `"Open ${title}, ${sold ? 'sold' : '$'+price}"` | Auditor |
| T2.6 | `.painting-card:focus-visible { outline: 2px solid var(--text); outline-offset: 4px; opacity: 1 !important; transform: none !important; }` so keyboard tab during stagger is visible | Frontend dev |
| T2.7 | Title typography: `1.4rem`, `line-height: 1.2`; meta `0.86rem`, `letter-spacing: 0.005em` | Designer |
| T2.8 | Drop `year` from card meta entirely (only 4/17 have it; asymmetric right-edge of meta strings is visually messier than missing data). Meta becomes `medium · size`. Year stays in modal | Designer R1.5 |
| T2.9 | Add `fetchpriority="high"` to the first `<img>` in `.gallery-grid` after render (LCP hint) | Frontend dev R1.5 |
| T2.10 | Drop `${PAINTINGS.length} pieces in the studio` from home (`:2323`) — accidentally diminishing at 17. Replace with editorial subtitle naming through-line of selected 6 (display italic, ≤120ch). Keep `<span class="eyebrow">Selected works</span>` since featured-count is now 6 (real selection) | Skeptic |
| T2.11 | **Derive `dominantColor` + `brightness` from image bytes at upload** in `push_painting.py` (one-time OpenCV sample, write back to the Sheet's existing `dominantColor` column plus a new `brightness` 0–1). Unlocks: (a) chromatic-collision check in `interleaveByOrientation` so adjacent cards don't both saturate / both go dark, (b) color-aware `selectFeatured` pick for the home 6 (cohere or contrast deliberately, not by accident), (c) possible "by mood" chapter on /works as a third axis alongside size. Avoids manual `brightness`/`color-scheme` columns that decay as inventory grows | Designer (chromatic-insulation R1.5 ask) + Curator (color-aware curation gap) |

## Tier 3 — nice-to-have

| # | Move | Driver |
|---|---|---|
| T3.1 | Meta separator: `·` → ` – ` (thin-space en-dash) in `paintingMeta()` | Designer |
| T3.2 | `.painting-card:active .card-image img { transform: scale(0.995); transition: transform .1s ease; }` | Auditor |
| T3.3 | `.painting-card.is-sold` opacity/desaturate **only on collection-detail pages with mixed sold/available** (not on /past, where every card is sold) | Curator R1.5 |
| T3.4 | Empty-state recovery link on `renderCollectionDetail` empty branch (~`:2593`): `→ See all available works` | Auditor |
| T3.5 | Sold cards on home or collection-detail get a `Commission similar →` link below price slot (Nielsen #1 — recovery path from sold to inquiry) | Auditor |

---

## R1 moves removed (no longer applies post-inventory)

| Move | Why killed |
|---|---|
| Per-card size chip (T1.6 in R1) | Only 2 size values + 70/30 skew — per-card chip is repetitive noise. Replaced by 85% inset for 11×14 (T1.10) + chapter-by-size (T1.3). |
| Editorial chapter dividers every ~16 cards on /works | 11 cards; dividers don't apply. |
| Sold-card desaturation on /past | Whole page is sold — desaturating everything just looks washed out. Restricted to collection-detail (T3.3). |
| Bump column min to 300px for clean 4×2 featured | 6 featured doesn't tile to 4-cols cleanly either; orphan-row concern was scale-dependent. Leave `260px` (`:409`). |
| rAF-coalesced masonry scheduler | 11 cards × ~ms layout cost — not worth the complexity. Dev says "current code as-is." |
| Image-load skeleton pulse | At 17 images max per page on modern devices, perceived delay is negligible. |
| Help/doc legend for № convention | Killed alongside № itself (T1.2). |
| Inline `onload` cleanup for CSP | Out of grid scope; security-reviewer concern. |
| `ResizeObserver` swap for window resize | Existing 100ms debounce handles iOS URL-bar jitter at 11 cards. |
| Mobile early-return in `computeMasonry` | Microsecond savings on 11 cards; not worth a dedicated edit. |

## Rejected moves (unchanged from R1, confirmed by R1.5)

| Move | Why (tied to the bar) |
|---|---|
| Replace masonry with uniform 4/5 letterbox tiles | Letterbox + `object-fit: contain` = the Saatchi/Etsy thumbnail wall the bar forbids. **Note:** Skeptic's case strengthened post-inventory; remains revisable if T1.10 + T1.11 don't resolve the "uncanny valley" feel. |
| Move price off card entirely | User call — kept on card (below meta, de-emphasized) despite R1.5 consensus push to header-only. |
| Sort dropdown on /works | Buyer killed it themselves in R1.5 — 11 cards, all $400, sort UI is performative. |
| Filter UI on /works | Same. /postcards (when populated) and Collections already filter by medium/theme. |
| Drop home featured grid entirely | User chose Curator's 6-cut, not Skeptic's eliminate. |

---

## Open questions still on the table

1. **Popstate semantics (T1.1) — confirm proposed default?** Close → `history.back()` if pushed, else `replaceState`. Direct-link arrival opens modal; closing replaces hash without leaving site.
2. **Featured-6 sequencing.** With 6 featured on a `minmax(260px, 1fr)` grid: 4 cols at 1440px = 4+2 (one orphan row), 3 cols at 1024px = 3+3. The 4+2 case may look off-balance vs. 6 = 3+3 or 2+2+2. Acceptable as-is, or do you want the featured grid forced to 3 cols × 2 rows on desktop via a `.gallery-grid--featured` modifier?
3. **Interleaving (T1.11)** — should it apply only to /works and home, or also to collection-detail? Where Sveta has hand-ordered the array, interleave should not reshuffle.
4. **Masonry escape valve.** If T1.10 + T1.11 don't fix the "Tetris autopilot" feel, the Skeptic's chapter-by-orientation grid (uniform mini-grids per orientation chunk) is the fallback. Flag for the R2 review.
5. **Russian copy** at `.../r1.ru.md` — update to match this revision?

---

## Execution sequencing

T1.1 (deeplinks), T1.2 (delete №), T1.5 (hide postcards), T1.6 (featured cut) are each isolated and can land in any order. T1.3 (size chaptering), T1.4 (price below meta), T1.10 (size inset), T1.11 (interleave) all interact with the rendering pipeline — group them in one edit pass. T1.7/T1.8/T1.9 (masonry FOUC + img attrs + srcset) are the perf bundle, can land last.

Total Tier 1 = ~250-350 lines changed across CSS + cardHTML + renderPaintings + selectFeatured + loadFromSheet + driveImageURL + nav render + openModal + new popstate listener.

---

## Reversibility

All changes are CSS + JS in `index.html`. Rely on git diff. If T1.10 + T1.11 (Designer's masonry refinements) look wrong in practice, they revert cleanly — neither breaks anything if reverted alone.
