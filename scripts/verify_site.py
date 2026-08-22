#!/usr/bin/env python3
"""Flag-driven Selenium verifier for sveta-colours.com.

This is the *first* thing to reach for after a UI change to index.html.
It covers the 80% case end to end:

    open the page -> wait for boot + images -> navigate via showPage()
    -> dump geometry + computed CSS -> hover/click -> dump again
    -> screenshot before/after

Every gotcha from docs/portfolio-playbook.md ("Verification process" +
"Pitfalls that cost real time") is baked in — see GOTCHAS below. If the
assertion you need can't be expressed as flags, copy
scripts/templates/selenium_verify.py instead and hand-write it.

Run it from the pipeline venv (Selenium 4 lives there):

    tools/image_pipeline/.venv/bin/python scripts/verify_site.py \
        --page paintings --dump-geometry \
        --computed-style flex-basis --computed-style=--ratio \
        --hover '.painting-card .card-image-inner' \
        --screenshot cursor-scope

GOTCHAS ENCODED HERE (do not "simplify" these away)
---------------------------------------------------
G1  Navigate with showPage('<name>'), never hash routing — /#works does not
    deep-link on first load.                             -> goto_page()
G2  Paintings arrive async from the published Google Sheet CSV; the app shows
    "Loading paintings…" first.                          -> wait_for_boot()
G3  Card heights refine on image `load`, not DOMContentLoaded — measuring too
    early reports the aspect-ratio placeholder height.   -> wait_for_images()
                                                            + --settle
G4  ActionChains.move_to_element on the same target twice does not re-fire
    mouseenter — reset via <h1> between hovers.          -> hover()
G5  The custom cursor is a `.custom-cursor` element with a `.visible` class
    toggle, and is wired ONLY to `.painting-card .card-image-inner` and
    `.hero-image-wrap`.                                  -> cursor_state(),
                                                            hover() warning
G6  Modal element is `.modal-overlay.open` (id="modal"), NOT `.modal.open`.
                                                         -> MODAL_SELECTOR
G7  save_screenshot captures the viewport only. Full page needs the CDP call
    Page.captureScreenshot{captureBeyondViewport:true}.  -> --full-page
G8  Favicons live in browser chrome and are NOT in any Selenium screenshot.
                                                         -> favicon_guard()
G9  Gallery cards lock rendered aspect to the *canvas* (--ratio = widthIn /
    heightIn), never the photo's natural aspect — so --dump-geometry reports
    canvas vs natural vs rendered side by side.          -> GEOMETRY_JS
G10 move_to_element_with_offset offsets from the element CENTER, not top-left.
                                                         -> --hover-frac
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_INDEX = os.path.join(REPO_ROOT, "index.html")

DEFAULT_URL = "https://www.sveta-colours.com/"
DEFAULT_SELECTOR = ".painting-card"
DEFAULT_WINDOW = "1440x1800"
SCREENSHOT_DIR = "/tmp/hover-test"

# G6 — the modal really is .modal-overlay.open. Two verify scripts were
# written against `.modal.open` before this was pinned down.
MODAL_SELECTOR = ".modal-overlay.open"
CURSOR_SELECTOR = ".custom-cursor"
# G5 — the only two things the custom cursor is wired to (wireCustomCursor()).
CURSOR_TARGETS = (".card-image-inner", ".hero-image-wrap")

# Page names accepted by showPage() (index.html render() switch) mapped to a
# selector that only exists once that page has rendered. Verified against
# index.html — do not invent entries here.
PAGES: dict[str, str] = {
    "home":        ".hero-title",
    "paintings":   ".hang-chapter, .gallery-grid",
    "postcards":   ".collection-header, .gallery-grid",
    "collections": ".category-grid",
    "about":       ".about-full",
    "faq":         ".faq-list",
    "past":        ".gallery-grid--all-sold, .collection-header",
}
# The nav labels these pages by different words than showPage() uses.
PAGE_ALIASES = {"works": "paintings", "pastwork": "past", "past-work": "past"}
# collection:<Name> is also valid (render() -> renderCollectionDetail).
COLLECTION_WAIT = ".collection-header, .gallery-grid"


# --------------------------------------------------------------------------
# Output plumbing: with --json, humans read stderr and machines read stdout.
# --------------------------------------------------------------------------

class Out:
    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        self.stream = sys.stderr if as_json else sys.stdout

    def say(self, msg: str = "") -> None:
        print(msg, file=self.stream, flush=True)

    def warn(self, msg: str) -> None:
        print(f"!! {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# G8 — favicon guard. Runs before Chrome is even started.
# --------------------------------------------------------------------------

FAVICON_PAT = re.compile(r"favicon|apple-touch-icon|link\[rel[^]]*icon", re.I)


def favicon_guard(args, out: Out) -> None:
    """Selenium screenshots are viewport-only; the favicon is browser chrome.

    Refuse to produce a screenshot that could never contain the thing being
    verified, and point at the check that actually works.
    """
    targeted = FAVICON_PAT.search(args.selector or "") or any(
        FAVICON_PAT.search(x or "") for x in (args.hover, args.click)
    )
    named = FAVICON_PAT.search(args.screenshot or "")
    if not (targeted or named):
        return

    base = args.url.rstrip("/")
    out.warn(
        "Favicons live in the browser tab (browser chrome). Selenium screenshots "
        "capture the VIEWPORT ONLY, so no screenshot from this script can ever "
        "show a favicon. Verify the served bytes instead:\n"
        f"    curl -sSI {base}/icons/favicon.svg\n"
        f"    curl -sS  {base}/icons/favicon.svg | head -40\n"
        f"    curl -sS  {base}/icons/favicon-32.png | shasum -a 256\n"
        f"    curl -sS  {base}/ | grep -o '<link rel=\"[^\"]*icon[^>]*>'\n"
        "Then open the site in a real browser and look at the tab."
    )
    if targeted:
        out.warn("Selector targets the favicon — refusing to run. Exit 2.")
        sys.exit(2)


# --------------------------------------------------------------------------
# Deploy wait. See the module docstring in scripts/templates/README.md for why
# this is text/content based and not SHA based.
# --------------------------------------------------------------------------

def _fetch(url: str) -> tuple[bytes, dict]:
    # Cache-bust: GitHub Pages sits behind Fastly with max-age=600, so a plain
    # re-request can serve the pre-push body for ten minutes.
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        f"{url}{sep}_cb={int(time.time() * 1000)}",
        headers={
            "Cache-Control": "no-cache, max-age=0",
            "Pragma": "no-cache",
            "User-Agent": "verify_site.py",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(), dict(resp.headers)


def wait_for_text(url: str, marker: str, timeout: float, out: Out) -> bool:
    out.say(f"Polling {url} for marker {marker!r} (timeout {timeout:.0f}s)…")
    deadline = time.time() + timeout
    last_etag = None
    while time.time() < deadline:
        try:
            body, headers = _fetch(url)
        except (urllib.error.URLError, OSError) as exc:
            out.say(f"  fetch failed ({exc}) — retrying")
            time.sleep(5)
            continue
        etag = headers.get("ETag") or headers.get("Last-Modified")
        if etag != last_etag:
            out.say(f"  served version changed: {etag}")
            last_etag = etag
        if marker.encode("utf-8", "replace") in body:
            out.say("  marker found — deploy is live.")
            return True
        time.sleep(5)
    out.warn(f"Timed out waiting for {marker!r} at {url}")
    return False


def wait_for_local(url: str, timeout: float, out: Out) -> bool:
    """Poll until the served page byte-matches the local index.html."""
    if not os.path.exists(LOCAL_INDEX):
        out.warn(f"No local index.html at {LOCAL_INDEX}")
        return False
    with open(LOCAL_INDEX, "rb") as fh:
        want = hashlib.sha256(fh.read()).hexdigest()
    out.say(f"Polling {url} until it matches local index.html (sha256 {want[:12]}…)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            body, headers = _fetch(url)
        except (urllib.error.URLError, OSError) as exc:
            out.say(f"  fetch failed ({exc}) — retrying")
            time.sleep(5)
            continue
        got = hashlib.sha256(body).hexdigest()
        if got == want:
            out.say("  served bytes match local index.html — deploy is live.")
            return True
        out.say(f"  served sha256 {got[:12]}… (etag {headers.get('ETag')}) — waiting")
        time.sleep(5)
    out.warn("Timed out waiting for the served page to match local index.html")
    return False


# --------------------------------------------------------------------------
# Page-side JS. One execute_script round-trip per dump — per-element
# find_element calls are an order of magnitude slower.
# --------------------------------------------------------------------------

GEOMETRY_JS = r"""
const sel = arguments[0];
const limit = arguments[1];
const props = arguments[2] || [];
const nodes = Array.from(document.querySelectorAll(sel));
const out = nodes.slice(0, limit > 0 ? limit : nodes.length).map((el, i) => {
  const r = el.getBoundingClientRect();
  const cs = getComputedStyle(el);
  // The painting card nests: .painting-card > .card-image > .card-image-inner > img
  const inner = el.matches('.card-image-inner')
    ? el
    : el.querySelector('.card-image-inner');
  const innerR = inner ? inner.getBoundingClientRect() : null;
  const img = el.tagName === 'IMG' ? el : el.querySelector('img');
  const card = el.closest('.painting-card');
  const titleEl = (card || el).querySelector('.card-title');
  // G9 — --ratio is the CANVAS aspect (widthIn/heightIn), set in cardHTML.
  const ratioVar = (inner ? getComputedStyle(inner).getPropertyValue('--ratio') : '').trim();
  let canvasAspect = null;
  if (ratioVar && ratioVar.indexOf('/') > -1) {
    const p = ratioVar.split('/').map(parseFloat);
    if (p[0] && p[1]) canvasAspect = p[0] / p[1];
  }
  const naturalAspect = (img && img.naturalWidth && img.naturalHeight)
    ? img.naturalWidth / img.naturalHeight : null;
  const box = innerR || r;
  const renderedAspect = box.height ? box.width / box.height : null;
  const parent = el.parentElement;
  const pcs = parent ? getComputedStyle(parent) : null;
  const styles = {};
  props.forEach(p => { styles[p] = cs.getPropertyValue(p).trim(); });
  return {
    i: i,
    tag: el.tagName.toLowerCase(),
    classes: el.getAttribute('class') || '',
    title: titleEl ? titleEl.textContent.trim() : '',
    card_id: card ? card.getAttribute('data-card-id') : null,
    orientation: card ? (card.getAttribute('data-orientation') || 'portrait') : null,
    size_class: card ? (card.getAttribute('data-size') || 'large') : null,
    sold: card ? card.classList.contains('is-sold') : null,
    x: Math.round(r.left), y: Math.round(r.top + window.scrollY),
    w: Math.round(r.width), h: Math.round(r.height),
    img_w: innerR ? Math.round(innerR.width) : null,
    img_h: innerR ? Math.round(innerR.height) : null,
    natural_w: img ? img.naturalWidth : null,
    natural_h: img ? img.naturalHeight : null,
    canvas_ratio: ratioVar || null,
    canvas_aspect: canvasAspect,
    natural_aspect: naturalAspect,
    rendered_aspect: renderedAspect,
    // Aspect drift is the number the canvas-lock rule exists to control.
    aspect_drift_pct: (canvasAspect && naturalAspect)
      ? ((naturalAspect - canvasAspect) / canvasAspect) * 100 : null,
    // Layout: the grid is flex now but cardHTML still seeds grid-row, and
    // computeMasonry() still refines it. Report both so neither can lie.
    grid_row_inline: el.style ? (el.style.gridRow || '') : '',
    grid_row: cs.gridRow,
    grid_column: cs.gridColumn,
    flex: cs.flex,
    flex_basis: cs.flexBasis,
    parent_display: pcs ? pcs.display : null,
    parent_gap: pcs ? pcs.gap : null,
    visible: r.width > 0 && r.height > 0,
    styles: styles,
  };
});
return {total: nodes.length, items: out};
"""

BOOT_JS = r"""
// G2 — loadFromSheet() paints "Loading paintings…" before PAINTINGS exist.
const app = document.getElementById('app');
if (!app) return false;
if (/Loading paintings/.test(app.textContent || '')) return false;
return !!document.querySelector('.hero-title, .painting-card, .collection-header');
"""

IMAGES_JS = r"""
// G3 — masonry/aspect refinement happens on image `load`.
const imgs = Array.from(document.querySelectorAll('#app img'));
if (!imgs.length) return true;
return imgs.every(i => i.complete);
"""

CURSOR_JS = r"""
const c = document.querySelector(arguments[0]);
if (!c) return {present: false};
return {
  present: true,
  visible: c.classList.contains('visible'),
  text: (c.textContent || '').trim(),
  body_cursor_hide: document.body.classList.contains('cursor-hide'),
};
"""

CURRENT_PAGE_JS = r"""
// `currentPage` is a top-level `let` in index.html — it lives in the global
// lexical scope, so it resolves here even though it is not on `window`.
try { return currentPage; } catch (e) { return null; }
"""


# --------------------------------------------------------------------------
# Driver session
# --------------------------------------------------------------------------

class Session:
    def __init__(self, args, out: Out) -> None:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        self.args = args
        self.out = out
        w, h = args.window.lower().replace("×", "x").split("x")
        opts = Options()
        opts.add_argument(f"--window-size={int(w)},{int(h)}")
        if args.headless:
            opts.add_argument("--headless=new")
        # Deterministic rendering across runs.
        opts.add_argument("--force-device-scale-factor=1")
        opts.add_argument("--hide-scrollbars")
        self.driver = webdriver.Chrome(options=opts)
        self.driver.set_window_size(int(w), int(h))

    # -- waiting -----------------------------------------------------------

    def _poll(self, script: str, timeout: float, label: str) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.driver.execute_script(script):
                    return True
            except Exception:  # noqa: BLE001 — mid-render DOM churn
                pass
            time.sleep(0.25)
        self.out.warn(f"timed out waiting for {label} ({timeout:.0f}s)")
        return False

    def wait_for_boot(self) -> None:
        """G2 — the CSV fetch is async; cards do not exist at DOMContentLoaded."""
        self._poll(BOOT_JS, self.args.timeout, "app boot (sheet CSV load)")

    def wait_for_images(self) -> None:
        """G3 — measure only once images have settled, or heights are wrong."""
        self._poll(IMAGES_JS, self.args.timeout, "images to finish loading")
        time.sleep(self.args.settle)

    # -- navigation --------------------------------------------------------

    def goto_page(self, page: str) -> None:
        """G1 — SPA nav goes through showPage(); hash URLs do not deep-link."""
        wait_sel = (
            COLLECTION_WAIT if page.startswith("collection:") else PAGES[page]
        )
        self.out.say(f"showPage({page!r}) — waiting for {wait_sel}")
        self.driver.execute_script("showPage(arguments[0]);", page)
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        WebDriverWait(self.driver, self.args.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, wait_sel))
        )
        landed = self.driver.execute_script(CURRENT_PAGE_JS)
        if landed and landed != page:
            self.out.warn(f"showPage({page!r}) but currentPage is {landed!r}")
        self.wait_for_images()

    # -- reading -----------------------------------------------------------

    def dump(self, label: str) -> dict:
        data = self.driver.execute_script(
            GEOMETRY_JS, self.args.selector, self.args.limit,
            self.args.computed_style,
        )
        data["label"] = label
        data["selector"] = self.args.selector
        data["cursor"] = self.driver.execute_script(CURSOR_JS, CURSOR_SELECTOR)
        data["page"] = self.driver.execute_script(CURRENT_PAGE_JS)
        data["title"] = self.driver.title
        data["modal_open"] = bool(
            self.driver.execute_script(
                "return !!document.querySelector(arguments[0]);", MODAL_SELECTOR
            )
        )
        return data

    # -- interacting -------------------------------------------------------

    def _find(self, selector: str):
        from selenium.webdriver.common.by import By

        els = self.driver.find_elements(By.CSS_SELECTOR, selector)
        return els[0] if els else None

    def _scroll_into_view(self, el) -> None:
        self.driver.execute_script(
            "arguments[0].scrollIntoView({block:'center', behavior:'instant'});", el
        )
        time.sleep(0.35)

    def hover(self, selector: str) -> bool:
        """G4 — reset the pointer via <h1> first, else mouseenter never re-fires."""
        from selenium.webdriver.common.action_chains import ActionChains
        from selenium.webdriver.common.by import By

        el = self._find(selector)
        if el is None:
            self.out.warn(f"--hover selector not found: {selector}")
            return False
        if not any(t in selector for t in CURSOR_TARGETS):
            # G5 — the cursor is wired to the image only; hovering the title
            # or the whole card is a valid test but must not be read as a
            # cursor regression.
            self.out.say(
                f"note: {selector} is outside the custom cursor's scope "
                f"({' / '.join(CURSOR_TARGETS)}) — cursor stays hidden by design"
            )
        self._scroll_into_view(el)
        heads = self.driver.find_elements(By.TAG_NAME, "h1")
        if heads:
            ActionChains(self.driver).move_to_element(heads[0]).perform()
            time.sleep(0.25)
        chain = ActionChains(self.driver)
        if self.args.hover_frac is None:
            chain.move_to_element(el)
        else:
            # G10 — offsets are measured from the element CENTER.
            box = el.rect
            dy = int((self.args.hover_frac - 0.5) * box["height"])
            chain.move_to_element_with_offset(el, 0, dy)
        chain.perform()
        time.sleep(0.45)
        return True

    def click(self, selector: str) -> bool:
        el = self._find(selector)
        if el is None:
            self.out.warn(f"--click selector not found: {selector}")
            return False
        self._scroll_into_view(el)
        try:
            el.click()
        except Exception:  # noqa: BLE001 — overlays / animation intercepts
            self.driver.execute_script("arguments[0].click();", el)
        # G6 — if this opened the modal it is .modal-overlay.open, never
        # .modal.open. Give it a beat to appear; absence is not an error
        # (plenty of clickable things don't open a modal).
        time.sleep(0.6)
        if self.driver.execute_script(
            "return !!document.querySelector(arguments[0]);", MODAL_SELECTOR
        ):
            self.out.say(f"{MODAL_SELECTOR} is open")
        self.wait_for_images()
        return True

    # -- output ------------------------------------------------------------

    def screenshot(self, name: str) -> str:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
        if self.args.full_page:
            # G7 — save_screenshot is viewport-only; CDP captures the full page.
            metrics = self.driver.execute_cdp_cmd("Page.getLayoutMetrics", {})
            css = metrics.get("cssContentSize") or metrics.get("contentSize")
            shot = self.driver.execute_cdp_cmd(
                "Page.captureScreenshot",
                {
                    "captureBeyondViewport": True,
                    "clip": {
                        "x": 0, "y": 0,
                        "width": css["width"], "height": css["height"],
                        "scale": 1,
                    },
                },
            )
            import base64

            with open(path, "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
        else:
            self.driver.save_screenshot(path)
        self.out.say(f"screenshot: {path}")
        return path

    def quit(self) -> None:
        try:
            self.driver.quit()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------

def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_dump(dump: dict, args, out: Out) -> None:
    items = dump["items"]
    out.say("")
    out.say(f"=== {dump['label']} — {dump['selector']} "
            f"({len(items)} of {dump['total']} shown) ===")
    out.say(f"page={dump['page']!r}  title={dump['title']!r}  "
            f"modal_open={dump['modal_open']}")
    cur = dump.get("cursor") or {}
    if cur.get("present"):
        out.say(f"custom cursor: visible={cur['visible']} text={cur['text']!r} "
                f"body.cursor-hide={cur['body_cursor_hide']}")
    else:
        out.say("custom cursor: not present (touch / reduced-motion / not booted)")

    if args.dump_geometry and items:
        out.say("")
        out.say(f"  {'#':<3} {'title':<26} {'w':>5} {'h':>5} {'img w×h':>11} "
                f"{'canvas':>7} {'natural':>8} {'render':>7} {'drift%':>7} "
                f"{'orient':<10} {'sold':<5} {'flex-basis':>11} {'grid-row':>10}")
        for it in items:
            imgwh = (f"{it['img_w']}×{it['img_h']}"
                     if it["img_w"] is not None else "-")
            out.say(
                f"  {it['i']:<3} {(it['title'] or it['tag'])[:25]:<26} "
                f"{it['w']:>5} {it['h']:>5} {imgwh:>11} "
                f"{_fmt(it['canvas_aspect']):>7} {_fmt(it['natural_aspect']):>8} "
                f"{_fmt(it['rendered_aspect']):>7} "
                f"{_fmt(it['aspect_drift_pct'], 1):>7} "
                f"{str(it['orientation'] or '-'):<10} "
                f"{str(it['sold']):<5} {it['flex_basis']:>11} "
                f"{(it['grid_row'] or '-')[:10]:>10}"
            )
        # Row grouping catches "the landscape card broke the row" bugs.
        rows: dict[int, list] = {}
        for it in items:
            rows.setdefault(round(it["y"] / 10) * 10, []).append(it)
        if len(rows) > 1:
            out.say("")
            out.say("  rows (by top offset):")
            for top in sorted(rows):
                names = ", ".join((r["title"] or r["tag"])[:18] for r in rows[top])
                out.say(f"    y={top:<6} {len(rows[top])} card(s): {names}")

    if args.computed_style and items:
        out.say("")
        out.say(f"  computed style on {dump['selector']}:")
        for it in items:
            head = f"{it['i']} {(it['title'] or it['tag'])[:24]}"
            vals = "  ".join(f"{p}={it['styles'].get(p, '')!r}"
                             for p in args.computed_style)
            out.say(f"    {head:<28} {vals}")


def print_diff(before: dict, after: dict, args, out: Out) -> None:
    out.say("")
    out.say("=== before → after diff ===")
    bcur, acur = before.get("cursor") or {}, after.get("cursor") or {}
    if bcur.get("visible") != acur.get("visible") or bcur.get("text") != acur.get("text"):
        out.say(f"  custom cursor: visible {bcur.get('visible')} → "
                f"{acur.get('visible')}, text {bcur.get('text')!r} → "
                f"{acur.get('text')!r}")
    if before["modal_open"] != after["modal_open"]:
        out.say(f"  {MODAL_SELECTOR}: {before['modal_open']} → {after['modal_open']}")
    bi = {it["i"]: it for it in before["items"]}
    changed = 0
    for it in after["items"]:
        prev = bi.get(it["i"])
        if not prev:
            continue
        deltas = []
        for key in ("x", "y", "w", "h", "img_w", "img_h"):
            if prev[key] != it[key] and prev[key] is not None and it[key] is not None:
                deltas.append(f"{key} {prev[key]}→{it[key]} "
                              f"({it[key] - prev[key]:+d})")
        for p in args.computed_style:
            if prev["styles"].get(p) != it["styles"].get(p):
                deltas.append(f"{p} {prev['styles'].get(p)!r}→{it['styles'].get(p)!r}")
        if deltas:
            changed += 1
            out.say(f"  [{it['i']}] {(it['title'] or it['tag'])[:24]}: "
                    + "; ".join(deltas))
    if not changed:
        out.say("  no per-element geometry or computed-style change")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

EPILOG = """\
pages (showPage() names, from index.html render()):
  home  paintings  postcards  collections  about  faq  past
  collection:<Name>          e.g. --page 'collection:Autumn'
  aliases: works -> paintings, past-work -> past

deploy waiting:
  GitHub Pages exposes NO commit SHA — not in index.html (there is no build
  meta tag) and not in the response headers (server/etag/last-modified only;
  the ETag is a content hash, not a SHA). So there is no honest
  --wait-for-deploy SHA. Use instead:
    --wait-for-local          poll until the served page byte-matches the
                              local index.html  (the usual "did my push land")
    --wait-for-text STRING    poll until STRING appears in the served HTML

exit codes:
  0 ok   2 bad usage / favicon target   3 selector never found
  4 deploy wait timed out               5 interaction selector not found

examples:
  # did the landscape transposition survive deploy?
  verify_site.py --wait-for-local --page paintings --dump-geometry

  # prove the hover rule actually fires
  verify_site.py --page paintings --selector '.painting-card' \\
      --computed-style transform --hover '.painting-card .card-image-inner' \\
      --screenshot card-hover

  # local dev server (scripts/dev.sh)
  verify_site.py --url http://localhost:8000/ --page about \\
      --selector '.about-full p' --computed-style font-size --dump-geometry
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_site.py",
        description="Flag-driven Selenium verification for sveta-colours.com. "
                    "Handles the playbook gotchas (showPage nav, async sheet "
                    "load, image-settle before measuring, hover reset, "
                    "custom-cursor scope, modal class name) so you don't.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"site URL (default {DEFAULT_URL}); use "
                        "http://localhost:8000/ with scripts/dev.sh")
    p.add_argument("--page", default=None,
                   help="page to navigate to via showPage() — see list below")
    p.add_argument("--list-pages", action="store_true",
                   help="print the valid --page names and exit")
    p.add_argument("--selector", default=DEFAULT_SELECTOR,
                   help=f"CSS selector under test (default {DEFAULT_SELECTOR})")
    p.add_argument("--limit", type=int, default=0,
                   help="only report the first N matches (0 = all)")
    p.add_argument("--dump-geometry", action="store_true",
                   help="per-element boxes, canvas vs natural vs rendered "
                        "aspect, flex/grid span info, row grouping")
    p.add_argument("--computed-style", action="append", default=[],
                   metavar="PROP",
                   help="repeatable; getComputedStyle property on --selector. "
                        "Custom properties work, but must use = so argparse "
                        "doesn't eat the leading dashes: --computed-style=--ratio")
    p.add_argument("--hover", metavar="CSS",
                   help="hover this selector, then re-dump (before/after diff)")
    p.add_argument("--hover-frac", type=float, default=None, metavar="F",
                   help="hover at vertical fraction F of the element "
                        "(0=top, 0.5=center, 1=bottom) instead of its center")
    p.add_argument("--click", metavar="CSS",
                   help="click this selector, then re-dump (before/after diff)")
    p.add_argument("--screenshot", metavar="NAME",
                   help=f"save {SCREENSHOT_DIR}/NAME.png; with --hover/--click "
                        "saves NAME-before.png and NAME-after.png")
    p.add_argument("--full-page", action="store_true",
                   help="full-page screenshot via CDP instead of viewport-only")
    p.add_argument("--window", default=DEFAULT_WINDOW, metavar="WxH",
                   help=f"browser window size (default {DEFAULT_WINDOW})")
    p.add_argument("--wait-for-text", metavar="STRING",
                   help="poll the URL until STRING appears in the served HTML")
    p.add_argument("--wait-for-local", action="store_true",
                   help="poll the URL until the served page byte-matches the "
                        "repo's local index.html")
    p.add_argument("--wait-for-deploy", metavar="SHA",
                   help=argparse.SUPPRESS)
    p.add_argument("--deploy-timeout", type=float, default=300.0, metavar="SEC",
                   help="how long to poll for the deploy (default 300)")
    p.add_argument("--settle", type=float, default=2.0, metavar="SEC",
                   help="extra settle time after images load, for layout "
                        "refinement (default 2.0)")
    p.add_argument("--timeout", type=float, default=25.0, metavar="SEC",
                   help="per-wait timeout (default 25)")
    p.add_argument("--headless", action="store_true",
                   help="run Chrome headless (no window)")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="emit the dumps as JSON on stdout (logs go to stderr)")
    return p


def resolve_page(page: str, out: Out) -> str:
    key = page.strip()
    if key.startswith("collection:"):
        return key
    key = PAGE_ALIASES.get(key.lower(), key.lower())
    if key not in PAGES:
        out.warn(f"unknown --page {page!r}. Valid: {', '.join(sorted(PAGES))}, "
                 f"collection:<Name> (aliases: {', '.join(sorted(PAGE_ALIASES))})")
        sys.exit(2)
    return key


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Out(args.as_json)

    if args.list_pages:
        for name, sel in sorted(PAGES.items()):
            out.say(f"{name:<12} waits for {sel}")
        out.say(f"{'collection:<Name>':<12} waits for {COLLECTION_WAIT}")
        out.say(f"aliases: {PAGE_ALIASES}")
        return 0

    if args.wait_for_deploy:
        out.warn(
            "--wait-for-deploy SHA is not implementable: GitHub Pages serves no "
            "commit SHA (index.html carries no build meta tag; the response has "
            "only server/etag/last-modified, and the ETag is a content hash). "
            "Use --wait-for-local (served bytes == local index.html) or "
            "--wait-for-text 'some string you just added'."
        )
        return 2

    favicon_guard(args, out)
    page = resolve_page(args.page, out) if args.page else None

    if args.wait_for_local and not wait_for_local(args.url, args.deploy_timeout, out):
        return 4
    if args.wait_for_text and not wait_for_text(
        args.url, args.wait_for_text, args.deploy_timeout, out
    ):
        return 4

    session = Session(args, out)
    result = {"url": args.url, "page": page, "selector": args.selector,
              "dumps": []}
    exit_code = 0
    try:
        out.say(f"GET {args.url}")
        session.driver.get(args.url)
        session.wait_for_boot()      # G2
        session.wait_for_images()    # G3
        if page:
            session.goto_page(page)  # G1

        before = session.dump("before" if (args.hover or args.click) else "state")
        result["dumps"].append(before)
        print_dump(before, args, out)

        if before["total"] == 0:
            out.warn(f"selector {args.selector!r} matched nothing — "
                     f"failing with exit 3")
            exit_code = 3

        shot = args.screenshot
        if shot:
            session.screenshot(f"{shot}-before" if (args.hover or args.click)
                               else shot)

        interacted = False
        if args.hover:
            if session.hover(args.hover):
                interacted = True
            elif exit_code == 0:
                exit_code = 5
        if args.click:
            if session.click(args.click):
                interacted = True
            elif exit_code == 0:
                exit_code = 5

        if interacted:
            after = session.dump("after")
            result["dumps"].append(after)
            print_dump(after, args, out)
            print_diff(before, after, args, out)
            if shot:
                session.screenshot(f"{shot}-after")
    finally:
        session.quit()

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
