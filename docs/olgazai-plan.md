# olgazai.com: second gallery site plan

Status: **Phases 1, 3 and 4 done; Phase 2 built and tested locally** (2026-11-12).
olgazai.com waits on Olga's Cloudflare invite and her links (see `sites/README.md`).
Source: brainstorm panel of 2026-11-12 (log: `.claude/brainstorm-panel/log.md`),
then a working prototype of both proposed looks, checked with Selenium at
390 / 1440 / 1920 / 2560 px.

## Goal

A portfolio site for Olga Zai (watercolor and acrylic, about as many paintings
as sveta-colours.com) built from the same code: same sheet format, same Drive
image approach, same publishing workflow. Different paintings, collections,
texts and look. The look must read as Olga's own, not a recolored copy.

The bar:

- Publishing to either site is the same five-minute workflow.
- A visitor who sees both sites doesn't suspect a template.
- A fix in shared code reaches both sites in one commit.

## Decisions

1. **One repo, one shared `index.html`.** A fork would drift within weeks:
   `index.html` had 35 commits in the month before this plan. Not a templating
   engine and not a runtime config fetch either.
2. **All artist-specific values in one marked block** of `index.html`
   (`<!-- SITE:BEGIN -->` … `<!-- SITE:END -->`): head meta, og tags, canonical,
   JSON-LD, the font link, the `:root` theme tokens and a `const SITE = {…}`
   object. Values are rendered into static HTML, never fetched, because link
   previews and crawlers don't run JS.
3. **Hosting.** sveta-colours.com stays on GitHub Pages exactly as now, with no
   DNS change. The root `index.html` keeps Sveta's block already filled in, so it
   still has no build step. olgazai.com is a **Cloudflare Pages project on the
   same repo**: build command `python scripts/build_site.py olga` copies
   `index.html`, swaps in the block rendered from `sites/olga/`, and copies
   `sites/olga/texts` and `images`. Moving Sveta to Cloudflare Pages too is
   optional and belongs with the Worker decision (`docs/painting-pages-plan.md`).
4. **Per-site config:** `sites/<name>/site.json` (identity, links, constants,
   tooling IDs), `sites/<name>/theme.css` (tokens), `sites/<name>/texts/`.
5. **First commit changes nothing visible.** It only moves values into the block.
   It merges only with zero visual difference, proven by Selenium screenshots at
   360 / 390 / 1280 px (home, all paintings, a collection, the modal, About, FAQ)
   plus a computed-CSS diff against the live site.
6. **Tooling:** `--site` is **required, with no default**, in `push_painting.py`,
   `audit_gallery.py`, `verify_site.py` and `verify_collection_order.py`.
   `photo_manager.html` takes `?site=olga`. The publish-painting skill asks
   "Sveta or Olga?" when it isn't clear and echoes the target sheet and URL
   before any real push. A bare "publish this" must never go to the wrong sheet.
7. **Access:** Olga owns her Sheet, Drive folder, Docs and form, and shares the
   Sheet, folder and Docs with Sveta's Google account as Editor. Sveta's existing
   OAuth client and token then work for both sites. Check that the Google Cloud
   app is "In production", not "Testing" (Testing tokens expire every 7 days).
   Olga publishing on her own comes later and needs a test with her account.
8. **Sheet schema unchanged.** `medium` is free text per row
   ("watercolor on Arches cold-press, 300 lb"). `size` is always the size of
   the painted image, so the canvas-aspect lock still works. Photos are cropped
   to the painted area, and the page draws the mat. One optional new column,
   `framing` (e.g. "matted to 16×20, framed under glass, frame included"),
   after `order`. It stays blank on Sveta's sheet; `push_painting.py` already
   preserves extra columns.
9. **Voice:** Olga's mode uses a neutral AI prompt in the photo manager, so
   Sveta's voice can't leak into her descriptions. An `olga-writing` skill is
   built from Olga's own sentences once she has 5 paintings published.
10. **Copy is never shared.** Every short string (nav, buttons, form wording,
    footer, terms) can be overridden in `SITE.strings`. Olga's FAQ is written
    from scratch for paper: framing, glass, flat shipping, image vs frame size,
    lightfastness. Sveta drafts Olga's shipping and return terms from her own
    FAQ, adapted for paper; Olga edits them.
11. **The sites acknowledge each other** with one human line in each About page.
    No shared footer, no badges.

## Look (B is live in sites/olga since 2026-11-12; Olga still confirms)

Both looks keep the same layout. Watercolors hang on a white mat drawn by CSS
(white ground, 9% padding around the image, hairline edge, soft shadow).
Acrylics get a thin edge. Both are keyed off a `data-medium` attribute on the
card. The home painting and the modal picture are matted the same way.

| | A · White paper (panel pick) | B · Grey wall |
|---|---|---|
| `--bg` / `--surface` / `--surface-alt` | `#F8F7F4` / `#FFFFFF` / `#EFEEEA` | `#D9DCDA` / `#E6E8E6` / `#CED2CF` |
| `--paper` / mat | `#FFFFFF` / `#FFFFFF` | `#F3F3F0` / `#FBFBF8` |
| `--border` / `--border-strong` | `#E2E0DA` / `#C9C6BD` | `#BEC3C0` / `#A2A8A5` |
| `--text` / `--text-secondary` / `--text-muted` | `#1F2326` / `#444A4E` / `#61686D` | `#1A1E1D` / `#323837` / `#4A5250` |
| `--accent` / hover | sea-teal `#2F6F73` / `#3F858A` | madder `#7A2E2A` / `#93403A` |
| Fonts (brand / body) | Gilda Display / Work Sans | Cormorant Garamond 500–600 / Karla |
| Logo | lowercase "olga zai", 0.04em tracking, small teal dot | "OLGA ZAI" in capitals, 0.24em tracking |
| Contrast (WCAG AA ≥ 4.5) | muted on bg 5.28, accent on bg 5.38 | muted on bg 5.82, accent on bg 6.75 |

Why not Sveta's bone ground: on warm bone, white watercolor paper looks dirty.

## Big screens (both sites)

Today the layout stops growing on wide monitors. At 2560 px the grid is
1,280 px wide with 3 columns, the home painting is 600 px, the modal is
1,080 px, and the page floats in empty space. Sveta's site does the same. The
prototype adds two steps, and nothing changes below 1800 px:

```css
@media (min-width: 1800px) {
  :root { --max-w: 1600px; --rail-w: 300px; }
  .painting-card { flex-basis: 330px; }
  .painting-card[data-orientation="landscape"] { flex-basis: 412px; }
  .hero { max-width: 1900px; }
  .hero-image-wrap { width: min(860px, 38vw, calc((100vh - 13rem) * var(--hero-ratio, 4/5))); }
  .hero-portrait { width: min(340px, 16vw, 30vh); }
  .hero-title { font-size: clamp(2.75rem, 2.4vw, 3.6rem); }
  .modal { max-width: 1480px; }
  .modal-content { grid-template-columns: minmax(0, 1.6fr) minmax(320px, 440px); }
  .modal-image { min-height: 640px; }
  .modal-image img { max-height: 84vh; }
}
@media (min-width: 2300px) {
  :root { --max-w: 1960px; --rail-w: 320px; }
  html { font-size: 112.5%; }
  .painting-card { flex-basis: 400px; }
  .painting-card[data-orientation="landscape"] { flex-basis: 500px; }
  .hero-image-wrap { width: min(1040px, 40vw, calc((100vh - 13rem) * var(--hero-ratio, 4/5))); }
  .modal { max-width: 1800px; }
}
```

Result at 2560 px: grid 1,960 px with 4 columns, home painting 1,024 px, modal
picture 1,360 px, no horizontal scroll.

As built (Phase 3), the painting window differs from the block above: the
prototype only had landscape watercolors, and on Sveta's portrait canvases the
wide image column left empty bands. The image column now takes its width from
the painting (`min(1040px, 84vh × --ratio)`, 1360px from 2300px) and the window
shrinks to fit. This goes into the shared code, so
Sveta's site gets it too. Check the masonry and the "same canvas, same scale"
rule (playbook) on Sveta's site before merging.

## Hard-coded Sveta values the prototype found

Besides the obvious ones (title, meta, og, JSON-LD, `SHEET_CSV_URL`, `TEXT_DOCS`,
`FORM_ENDPOINT`, `CONTACT_EMAIL`, `HERO_PIN_TITLE`, `FEATURED_TITLES`, the
portrait URL, alt texts, `document.title` suffix):

- **All paintings chapters** are hard-coded to Sveta's two canvas sizes:
  "Large works — 16 × 20 in" and "Small works — 11 × 14 in" (`renderWorks`
  area, `sizeChapterFor()`). Must become a per-site setting: by size for Sveta,
  by medium (Watercolors / Acrylics) for Olga.
- **Rail note** "Original paintings / Thornhill, Ontario" is literal HTML in the
  nav, not read from `SITE_DATA.artist.location`.
- **Modal meta values** are `white-space: nowrap`, so a long medium
  ("Watercolor on cold-press paper") is cut off. Allow wrapping.
- **Sveta's wording** in code: "commission requests welcome" (All paintings
  intro), "Replies within a day or two", "Shipping quoted to your address.
  Insured, from Canada."
- **28 hex colors outside `:root`**, all warm bone: the 9 placeholder gradients
  around line 2467, the hero placeholder gradient, the nav background
  `rgba(234, 230, 222, …)` (two places), the `var(--surface-alt, #E0DBD1)`
  fallback, and the bone RGB `234, 230, 222` in the `--accent-tint` mixing JS.
  All become tokens.
- **Font rules** "two fonts, no italic, upright logo" are Sveta's taste, kept as
  per-site options, not site law.

## Phases

| Phase | Work | Estimate |
|---|---|---|
| 0 | Olga's checklist (below), in parallel | — |
| 1 ✓ | Zero-visual-diff move of all artist values into the SITE block; Selenium + computed-CSS proof; commit + push | 5–7 h |
| 2 (code ✓; Cloudflare waits for Olga) | `sites/{sveta,olga}/`, `build_site.py`, Cloudflare Pages project, olgazai.com DNS; per-site chapters (`chapterBy`) and an optional portrait came along; first version shows every painting Olga has entered in her sheet (test the setup on one 11×15 watercolor first) | 3–4 h |
| 3 ✓ | Olga's theme, strings and mat variant; per-site chapters; big-screen rules; contrast-check script (AA 4.5 text, 3:1 focus ring); `verify_site.py --site`; white-image card-edge test at 360 px | 6–7 h |
| 4 ✓ | `--site` in `push_painting.py`, audit and verify scripts, the publish skill; photo manager `?site=`; watermark "© olgazai"; neutral AI prompt | 3–4 h |
| **v1** | | **17–22 h** |
| 5 | After Olga has 5 paintings: `olga-writing` skill, og-image and JSON-LD polish, collection covers, `health_check.py` for both sites, Worker keyed by hostname, Olga publishing on her own | later |

## What Olga provides

- A Google Sheet with Sveta's header row (plus the optional `framing` column),
  published to the web as CSV: the CSV link.
- A Drive folder for painting photos, shared as "anyone with the link can view".
- The Sheet, the folder and the three Docs shared with Sveta's Google account as Editor.
- Three Google Docs (home, about, FAQ), each "Published to the web": the links.
- The inquiry form: the Apps Script from `inquiry/Code.gs` deployed under her
  account (the web app URL), plus the contact email that receives inquiries.
- Domain access: olgazai.com is registered and already on Cloudflare, in
  **Olga's** Cloudflare account (confirmed 2026-11-12). The Pages project must
  live in the same account as the zone, so Olga adds Sveta as a member
  (Manage Account → Members → Invite, role Administrator) and the
  olgazai.com Pages project is created in Olga's account.
- A portrait photo; optionally a studio photo.
- Which painting leads the home page. Collection names come with the sheet.
- Three to five sentences about her work in her own words.
- Her edits to the drafted terms (shipping from Thornhill, flat and insured;
  returns; whether the frame is in the price). The draft is in the letter.
- The sheet filled in by her first: title, medium, size, price, sold,
  collection, framing, description. The first version shows every row.
- Her OK on look B (now in `sites/olga`; A stays in git, commit 595ac3a), the fonts and the logo.
- Photos taken to the guide: diffuse daylight, no glass, white balance set off
  the paper margin or a gray card, crop to the painted area.

## Open on purpose

- **A different home-page structure for Olga** (the buyer seat's idea). It waits
  until she reacts to v1 with her own paintings; a `data-variant` hook makes it
  cheap later.
- **Moving Sveta to Cloudflare Pages.** It goes with the Worker decision.
- **When Olga publishes on her own.** It depends on an OAuth test with her account.
- **"Sveta or Olga?" in the publish skill** is asked only once `tools.sheetId`
  is filled in for Olga; until then sveta is the only possible target.
- **A separate sheet-size column.** It's folded into `framing` for now; it becomes
  a column only if buyers actually need to filter on it.

## How to check this plan was followed

- `scripts/verify_site.py` and `scripts/verify_collection_order.py` pass for
  `--site sveta` and `--site olga`.
- Phase 1: screenshots and computed CSS identical to the live site before merge.
- Zero "Sveta" in Olga's rendered DOM, title, meta and alt texts, and zero "Olga"
  in Sveta's (except the one About line).
- Both themes pass the contrast check; the focus ring is visible on the first
  10 tab stops; no horizontal scroll at 360 / 390 px; a pure-white image keeps a
  visible edge.
- `curl -s https://olgazai.com | grep og:title` shows Olga's title (static, no JS).
