# Verify-script templates

Selenium scaffolds for UI verification of the portfolio site.

**Reach for `scripts/verify_site.py` first.** Copy `selenium_verify.py` only
when the assertion genuinely can't be expressed as flags.

## The split

| | `../verify_site.py` | `selenium_verify.py` (this dir) |
| --- | --- | --- |
| What it is | A parameterized CLI, committed and maintained | A copy-and-edit template |
| Use it for | The 80% case: open → navigate → measure → hover/click → screenshot | Novel, one-off assertions with real logic |
| Lifetime | Permanent | Copy to `/tmp/<intent>_verify.py`, throw away |

If you find yourself hand-editing the template for something you've now done
twice, that's a sign it should become a flag on `verify_site.py` instead.

## Files

- `selenium_verify.py` — generic UI verify template. Loads the deployed site,
  navigates SPA-style via `showPage()`, dumps card geometry, hover-tests the
  custom cursor, screenshots the result.

## Everyday flow (`verify_site.py`)

```bash
V=tools/image_pipeline/.venv/bin/python

# what's on the /works page right now?
$V scripts/verify_site.py --site sveta --page paintings --dump-geometry

# wait out the GitHub Pages lag after a push, then measure
$V scripts/verify_site.py --site sveta --wait-for-local --page paintings --dump-geometry

# prove a hover rule actually fired (before/after diff + two screenshots)
$V scripts/verify_site.py --site sveta --page paintings \
    --selector '.painting-card' --computed-style transform \
    --hover '.painting-card .card-image-inner' --screenshot card-hover

# read a custom property (note the `=`, or argparse eats the dashes)
$V scripts/verify_site.py --site sveta --page paintings \
    --selector '.card-image-inner' --computed-style=--ratio

# against the local server from scripts/dev.sh
$V scripts/verify_site.py --site sveta --url http://localhost:8000/ --page about \
    --selector '.about-full p' --computed-style font-size --dump-geometry
```

`--list-pages` prints the valid `--page` names (`home`, `paintings`,
`postcards`, `collections`, `about`, `faq`, `past`, `collection:<Name>`, plus
the `works → paintings` alias). Screenshots land in `/tmp/hover-test/`.
`--json` puts the dumps on stdout and the log on stderr. Exit codes: `2` bad
usage or a favicon target, `3` selector never matched, `4` deploy wait timed
out, `5` a `--hover`/`--click` selector wasn't found.

### Waiting for a deploy

There is no `--wait-for-deploy SHA`, because there is nothing to match it
against: `index.html` carries no build/version meta tag, and the GitHub Pages
response returns only `server` / `etag` / `last-modified` — the ETag is a
content hash, not a commit SHA. Two honest substitutes:

- `--wait-for-local` — polls until the served page byte-matches the repo's
  local `index.html`. This is the "did my push actually land" check.
- `--wait-for-text STRING` — polls until `STRING` appears in the served HTML.
  Use when the working tree has moved on past what you pushed.

Both cache-bust the Fastly edge (`?_cb=<ms>` + `no-cache`), so a 600s
`max-age` won't hand you the pre-push body.

## Falling back to the template

```bash
cd tools/image_pipeline
source .venv/bin/activate
cp ../../scripts/templates/selenium_verify.py /tmp/<intent>_verify.py
# Edit the ASSERTIONS section for the specific change you're verifying.
python /tmp/<intent>_verify.py
```

One script per change batch, named per intent (`/tmp/cursor_scope_verify.py`,
`/tmp/landscape_span_verify.py`). The template lives here in the repo; the
per-batch copies live in `/tmp/` and get thrown away after the batch ships.

## Gotchas — both files handle these

Documented in `docs/portfolio-playbook.md`; `verify_site.py` labels each one
`G1`…`G10` in its module docstring and at the call site.

- **Hash routing doesn't deep-link on first load** — navigate via
  `showPage('paintings')`, not `/#works`.
- **Paintings load async from the published Sheet CSV** — the app shows
  "Loading paintings…" before any `.painting-card` exists.
- **Card heights refine on image `load`**, not `DOMContentLoaded` — measure
  after images settle or you're reading the placeholder aspect.
- **`move_to_element` twice on the same target doesn't re-fire `mouseenter`**
  — reset via `<h1>` between hovers.
- **`move_to_element_with_offset` offsets from the element *center***, not its
  top-left (`--hover-frac` does the arithmetic).
- **The custom cursor** is `.custom-cursor` with a `.visible` class toggle, and
  is wired only to `.painting-card .card-image-inner` and `.hero-image-wrap`.
- **The modal is `.modal-overlay.open`**, not `.modal.open`.
- **Cards lock rendered aspect to the canvas** (`--ratio` = `widthIn/heightIn`),
  never the photo's natural aspect — `--dump-geometry` prints canvas vs
  natural vs rendered plus the drift.
- **`save_screenshot` is viewport-only** — `--full-page` uses the CDP
  `Page.captureScreenshot{captureBeyondViewport:true}` path instead.
- **Favicons are browser chrome and are not in any screenshot.** Point a
  selector at one and `verify_site.py` refuses and prints the `curl` checks.
