# Per-painting pages: Cloudflare Worker plan

Status: **planned, not started** (written 2026-11-12). Nothing here is live.

## Goal

Every painting gets a real URL, `https://www.sveta-colours.com/paintings/<slug>`,
whose HTML already carries that painting's title, description and picture in
`<head>`. Two things need that:

- **Link previews.** Telegram, WhatsApp and Facebook don't run JS. Today every
  shared link shows the generic site card, because `index.html`'s `<meta og:*>`
  are static.
- **Search.** Google runs JS, but slowly. A per-painting URL with its own
  `<title>`, description, `og:image` and JSON-LD, listed in `/sitemap.xml`,
  is indexed as its own page.

Not in scope: changing how the site looks, server-rendering the UI, images,
checkout.

## Decision: a Worker, not Spring Boot

Decided 2026-11-12 after comparing the two for this site (`/expert-lens`):

- Practising or showing Spring Boot is **not** a goal.
- Online checkout may come later, but it's low priority.
- The job is small: rewrite one `<head>` per request and emit a sitemap from 22 sheet rows.
  A Worker does it in about 150 lines of JS, with no host to run, patch or pay for,
  and GitHub Pages stays the origin.
- Spring Boot would need a JVM host, a deploy pipeline, and a way to keep
  it up. That cost buys nothing here until there's real server state
  (orders, holds). Even then Workers have storage (KV, D1), so checkout doesn't
  force Spring either. Revisit only if checkout turns out to need it.

This replaces the "one Spring Boot app" target in `docs/portfolio-playbook.md`
→ "Extending the site" for the routes, head and sitemap parts. That section's
seam map, the slug rules and the checkout-hold rule still apply.

## Cost (confirmed 2026-11-12 on Cloudflare's docs)

Sources: https://developers.cloudflare.com/workers/platform/pricing/ and
https://developers.cloudflare.com/workers/platform/limits/

| Free plan | Value |
| --- | --- |
| Requests | 100,000 per day, reset at midnight UTC |
| CPU time | 10 ms per request (computing only; waiting on fetches doesn't count) |
| Subrequests | 50 per request (Cache API calls count toward it) |
| Memory | 128 MB |
| Price | $0 |

- Paid plan, if it is ever needed: $5/month minimum, which includes 10M
  requests and 30M CPU-ms a month. Extra requests cost $0.30 per million.
- Over the free limit, Cloudflare returns error 1027. With the route set to
  **fail open**, requests skip the Worker and go to GitHub Pages instead.
- HTMLRewriter has no separate price; it uses the request's CPU time.

Only `/paintings/*` and `/sitemap.xml` go through the Worker. Home, gallery,
images and everything else don't count toward the limit.

## Shape

```
visitor / crawler
   │
Cloudflare (www proxied, orange cloud)
   ├── /paintings/*  ──► Worker ──► fetch "/" from GitHub Pages (the shell)
   │                       │      ──► published CSV of the Paintings sheet (cached ~5 min)
   │                       └── HTMLRewriter fills <head> for that slug → 200
   │                           unknown slug → shell with noindex → 404
   ├── /sitemap.xml  ──► Worker → one <url> per slug + the fixed pages
   └── everything else ─────────► GitHub Pages, untouched
```

- **The sheet stays the source of truth.** The Worker reads the same published
  CSV the site reads. There's no second copy of painting data, and nothing is in code.
- **The shell is `index.html` as GitHub Pages serves it.** Fetching `/` from the
  Worker reaches the origin directly, because `/` isn't on a Worker route.

## Steps

Each step can be shipped alone. Steps 1–4 are local only; step 5 changes the live site.

### 1. `worker/`: the Worker

- `worker/src/index.js`, `worker/wrangler.toml`, `worker/README.md`.
- `/paintings/<slug>`:
  - Find the row by `slug`, case-insensitive. Use the same `slugify(title)`
    fallback as the site when the column is blank.
  - In `<head>`, set `<title>`, `meta description`, `og:title`,
    `og:description`, `og:image`, `og:url`, `twitter:*`, and
    `<link rel="canonical">`.
  - Add a `VisualArtwork` JSON-LD block with name, image, size, medium, and an
    `Offer` with price, or SoldOut when sold.
  - Add a `<noscript>` block with the title and description, for crawlers.
  - HTML-escape every value. Titles and descriptions come from the sheet.
- Unknown slug: the shell with `<meta name="robots" content="noindex">` and
  status 404. The page still loads, so the visitor sees the site.
- `/sitemap.xml`: home, the fixed pages, and one URL per painting.
- CSV parsing: a real parser for quoted fields, commas and newlines inside
  descriptions. Look columns up by header, case-insensitive, the same way the
  site's `get('column')` does.

### 2. Data and caching

- Fetch the published CSV through the Cache API with a synthetic cache key and
  a TTL of about 5 minutes. The published CSV already lags about 5 minutes
  behind the sheet (playbook → "CSV cache invalidation"), so a shorter TTL
  gains nothing.
- Cache the fetched shell the same way.
- If a fetch fails, serve the shell with the generic head rather than an error.

### 3. `index.html`: open a painting from the path

- **3a.** On load, if `location.pathname` is `/paintings/<slug>`, open that
  painting, exactly as `hydrateFromHash` does for `#painting/<slug>`. Hash
  links keep working.
- **3b.** Card links become `href="/paintings/<slug>"`, a crawlable link. The
  click is intercepted to open the modal, and `openModal` does
  `pushState('/paintings/<slug>')`. Closing the modal returns to the previous
  URL. This makes the URL visitors copy the one with the preview. It's the
  "medium" router change from the playbook's seam map, so `popstate` and
  `closeModal()` change together.
- **Fallback for when the Worker isn't in front:** `404.html` on GitHub Pages
  redirects `/paintings/<slug>` to `/#painting/<slug>`. That covers error 1027
  and the Worker being off or down, and lets 3a/3b ship before step 5.

### 4. Local checks (`npx wrangler dev`)

- `curl -s http://localhost:8787/paintings/<slug> | grep og:image` returns
  that painting's own image, for **all 22 slugs**. Check it with curl, not a
  browser, because previews don't run JS.
- An unknown slug returns 404 and has `noindex`.
- `/sitemap.xml` is valid XML and has 22 painting URLs plus the fixed pages.
- A title with `"`, `<`, `&` in it comes out escaped. Test with a synthetic CSV row.
- Selenium: `/paintings/<slug>` opens the modal; closing and Back behave; the
  old `#painting/<slug>` link still works.
- `scripts/verify_site.py` and `scripts/verify_collection_order.py` still pass.

### 5. Switch live (ask Sveta first)

1. Cloudflare DNS: `www` from DNS-only (grey) to proxied (orange).
2. SSL/TLS mode **Full**, not Flexible, to avoid a redirect loop with GitHub Pages' HTTPS.
3. `npx wrangler deploy`, with routes `www.sveta-colours.com/paintings/*` and
   `www.sveta-colours.com/sitemap.xml`, route mode **fail open**.
4. Check the live site:
   - the curl checks from step 4, against www;
   - Facebook Sharing Debugger;
   - Telegram (paste a link in Saved Messages);
   - a WhatsApp message to herself;
   - Search Console: submit `/sitemap.xml`, then inspect one painting URL.
5. Keep watching GitHub Pages' "Enforce HTTPS" certificate. See the risks below.

**Rollback:** delete the routes, or set `www` back to DNS-only. The site is
then exactly what it is today, and the `404.html` fallback keeps
`/paintings/` links working.

## What Sveta needs to do

- `brew install node` (wrangler needs Node; it isn't installed).
- `npx wrangler login`. It opens a Cloudflare sign-in in the browser; she
  approves it herself.
- Step 5: confirm before the switch. Some Cloudflare dashboard toggles may be
  hers to click.

## Risks to check, not yet verified

- **GitHub Pages behind the Cloudflare proxy.** The GitHub Pages settings page
  sometimes reports the custom domain as misconfigured once it's proxied, and
  certificate renewal may need the proxy briefly off. This is from memory, not
  checked: read GitHub's and Cloudflare's current docs before step 5.
- **`og:image` from `lh3.googleusercontent.com`.** Check that Facebook and
  Telegram accept it: content type, size, and no redirect to a sign-in page.
  If they don't, the Worker can proxy the image under `/og/<slug>.jpg`.
- **The apex `sveta-colours.com`.** How it's set up today hasn't been checked.
  It should redirect to `www`, so the canonical URL is the only one.
- **Facebook caches previews.** Use "Scrape Again" in the Sharing Debugger
  after the switch.

## Later (not part of this plan)

- `/collections/<slug>` pages, the same way.
- Checkout:
  - a Worker with Stripe Checkout;
  - the hold rule from the playbook ("The trap with selling originals") using
    KV or D1;
  - the webhook writes `sold=yes` to the sheet.
- Image resizing and WebP.
