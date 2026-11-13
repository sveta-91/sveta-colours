# sites/

One folder per artist. `index.html` is shared code plus one generated block
(`<!-- SITE:BEGIN -->` … `<!-- SITE:END -->`) rendered from a folder here by
`scripts/build_site.py`. Plan and reasons: `docs/olgazai-plan.md`.

| Site | Domain | Hosting | How it's built |
| --- | --- | --- | --- |
| `sveta` | www.sveta-colours.com | GitHub Pages, off `main` | root `index.html` carries this block; no build |
| `olga` | olgazai.com | Cloudflare Pages, Olga's account | `python3 scripts/build_site.py olga --strict` → `dist/` |

## Files in a site folder

- **`site.json`**
  - `head`: title, description, Open Graph / Twitter text, canonical URL, og image
    (empty → no og:image tags), Google Fonts URL, the Person JSON-LD.
  - `files`: what the build copies next to `index.html` (`texts`, `images`,
    `icons`, `favicon`), as repo paths.
  - `site`: becomes `const SITE` in the page. Name, domain, `artist`, contact email,
    currency, `formEndpoint`, `sheetCsvUrl`, `textDocs` (published Google Docs),
    `heroPinTitle` / `featuredTitles` (safety nets when the sheet's `hero` / `recent`
    are empty), `chapterBy` (`size` or `medium` for the All paintings chapters),
    `portrait` (null → none), `strings` (the artist's own short wording),
    `fallbackPaintings` (shown only when the sheet can't be read).
- **`theme.css`**: the `:root` colour and font tokens.
- **`texts/`** (Olga): backup copies of the long texts, `[label]` format as in the root `texts/`.
- **`images/`** (Olga): portrait fallback and og image, once she sends them.

## Commands

```bash
python3 scripts/build_site.py sveta --root    # after editing sites/sveta/*: rewrite the root block
python3 scripts/build_site.py --check         # root block == sites/sveta? (run before committing)
python3 scripts/build_site.py olga            # → dist/ (warns about empty required links)
python3 scripts/build_site.py olga --strict   # fails while contactEmail / sheetCsvUrl / formEndpoint are empty
python3 scripts/build_site.py olga --out /tmp/x --dev-csv some.csv   # local test with a stand-in sheet
```

Never edit the block in `index.html` by hand: `--check` fails and the next
`--root` overwrites it.

## olgazai.com on Cloudflare Pages (once Olga has invited Sveta)

1. Cloudflare dashboard, **Olga's account** → Workers & Pages → Create → Pages →
   Connect to Git → this GitHub repo, branch `main`.
2. Build settings: framework preset **None**; build command
   `python3 scripts/build_site.py olga --strict`; output directory `dist`.
3. Save and deploy; check the `*.pages.dev` address first.
4. Custom domains → add `olgazai.com` (and `www.olgazai.com`). The zone is in the
   same account, so Cloudflare creates the DNS records itself.
5. Every push to `main` rebuilds olgazai.com. sveta-colours.com keeps deploying
   from GitHub Pages as before.

Until Olga's links are in `sites/olga/site.json`, `--strict` makes the Cloudflare
build fail on purpose, so a half-filled site can't go live.
