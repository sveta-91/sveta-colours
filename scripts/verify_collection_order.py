#!/usr/bin/env python3
"""Collection order + cover test.

Usage (from the repo root):
    tools/image_pipeline/.venv/bin/python scripts/verify_collection_order.py --site sveta          # live site
    tools/image_pipeline/.venv/bin/python scripts/verify_collection_order.py --site sveta --url http://localhost:8765/
--site is required (no default); the URL comes from sites/<site>/site.json.
Run it live: the image CDN refuses localhost, but order checks work either way.
Exit code 0 = all pass, 1 = any failure.

Part 1 checks the site against Sveta's REAL sheet data, whatever she has
numbered: for every collection, the test works out the expected order and
button picture on its own (from the raw collection / order / cover fields)
and compares with what the page shows. Nothing assumes a collection is
un-numbered, so reordering in the sheet doesn't break it.

Part 2 wipes one collection's ordering in memory (never in the sheet) and
drives the rules with synthetic values: blank → sheet order, 1..n reverses,
cover picks the picture, "A:1,B:99" moves one painting in both. A and B are
tools.orderTestCollections in site.json (Sveta: Forest, Seasons); when that is
empty the test takes the largest collection and one that shares a painting.
"""
import argparse, sys, time, json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import site_config  # scripts/site_config.py
ap = argparse.ArgumentParser(description="Collection order + cover test.")
site_config.add_site_arg(ap)
ap.add_argument("--url", help="override the site URL from site.json")
args = ap.parse_args()
BASE = args.url or site_config.load(args.site)["head"]["url"]
PAIR = site_config.tools(args.site).get("orderTestCollections") or []
fails = []
def check(n, c, x=""):
    print(("PASS " if c else "FAIL ") + n + ("" if c else " :: " + json.dumps(x))); c or fails.append(n)
o = Options(); o.add_argument("--headless=new"); o.add_argument("--window-size=1700,900")
d = webdriver.Chrome(options=o); d.get(BASE + "#collections")
for _ in range(80):
    if d.execute_script("return typeof PAINTINGS!=='undefined' && PAINTINGS.length>0 && !!document.querySelector('#navCollections .coll-tab')"): break
    time.sleep(0.25)
time.sleep(1)

# Independent re-statement of the documented rule (CLAUDE.md sheet schema):
# inline "Name:N" in collection wins, else `order` ("3" or "Forest:2,..."),
# numbered first ascending, then the rest in sheet order.
EXPECT_JS = """
const pos = (p, name) => {
  const k = name.toLowerCase();
  if (p.collOrder && typeof p.collOrder[k] === 'number') return p.collOrder[k];
  const raw = String(p.order || '').trim();
  if (!raw) return Infinity;
  if (/^[0-9.]+$/.test(raw)) return +raw;
  for (const part of raw.split(',')) { const [n, v] = part.split(':'); if (v && n.trim().toLowerCase() === k) return +v; }
  return Infinity;
};
const expectOrder = name => PAINTINGS.map((p, i) => ({p, i}))
  .filter(x => (x.p.collections || []).includes(name))
  .sort((a, b) => (pos(a.p, name) - pos(b.p, name)) || (a.i - b.i)).map(x => x.p);
const expectCover = name => { const l = expectOrder(name).filter(p => p.image);
  return (l.find(p => (p.cover || []).includes(name.toLowerCase())) || l.find(p => !p.sold) || l[0]).image; };
const titles = () => [...document.querySelectorAll('.coll-main .painting-card .card-title')].map(x => x.textContent);
const thumbSrc = n => [...document.querySelectorAll('#navCollections .coll-tab')]
  .find(x => x.querySelector('.name').textContent === n).querySelector('img').getAttribute('src');
"""

# ---- Part 1: real sheet data, every collection
real = d.execute_script(EXPECT_JS + """
  return getAllCollections().map(name => {
    showPage('collection:' + name);
    return { name, shown: titles(), expected: expectOrder(name).map(p => p.title),
             numbered: expectOrder(name).filter(p => pos(p, name) !== Infinity).map(p => p.title + '=' + pos(p, name)),
             cover: thumbSrc(name), expectCover: expectCover(name) };
  });""")
for c in real:
    check(f"sheet data: {c['name']} page order", c["shown"] == c["expected"], c)
    check(f"sheet data: {c['name']} button picture", c["cover"].startswith(c["expectCover"]), [c["cover"], c["expectCover"]])
    print(f"   {c['name']}: numbered {c['numbered'] or 'none'}")

# ---- Part 2: synthetic rules on collection A (in memory only)
r = d.execute_script(EXPECT_JS + """
  const out = {};
  let [A, B] = arguments[0];
  if (!A) {   // no pair configured: largest collection + one sharing a painting with it
    const count = {}; PAINTINGS.forEach(p => p.collections.forEach(c => count[c] = (count[c] || 0) + 1));
    A = Object.keys(count).sort((x, y) => count[y] - count[x])[0];
    const shared = PAINTINGS.find(p => p.collections.includes(A) && p.collections.length > 1);
    B = shared ? shared.collections.find(c => c !== A) : null;
  }
  out.A = A; out.B = B;
  const forest = PAINTINGS.filter(p => p.collections.includes(A));
  const wipe = () => forest.forEach(p => { p.order = ''; p.cover = []; p.collOrder = {}; });
  const go = n => { showPage('collection:' + n); return titles(); };
  out.sheetOrder = forest.map(p => p.title);
  wipe(); render(); out.blank = go(A);
  out.blankCover = thumbSrc(A); out.blankExpect = (forest.find(p => p.image && !p.sold) || forest[0]).image;
  forest.forEach((p, i) => { p.order = String(forest.length - i); });
  const pick = forest[forest.length - 1]; pick.cover = [A.toLowerCase()];
  render(); out.reversed = go(A); out.coverSrc = thumbSrc(A); out.pickImg = pick.image;
  wipe();
  const both = B && PAINTINGS.find(p => p.collections.includes(A) && p.collections.includes(B));
  if (both) { both.collOrder = {}; both.order = A + ':1,' + B + ':99'; }
  if (B) PAINTINGS.filter(p => p.collections.includes(B) && p !== both).forEach(p => { p.order = ''; p.collOrder = {}; });
  render();
  out.both = both ? both.title : null; out.firstForest = go(A)[0]; out.firstSeasons = B ? go(B)[0] : null;
  return out;""", PAIR)
print(f"   synthetic part on {r['A']}" + (f" + {r['B']}" if r['B'] else ""))
check("blank: page keeps sheet order", r["blank"] == r["sheetOrder"], r)
check("blank: picture = first available painting", r["blankCover"].startswith(r["blankExpect"]), r)
check("order 1..n reverses the collection", r["reversed"] == list(reversed(r["sheetOrder"])), r["reversed"])
check("cover column picks the button picture", r["coverSrc"].startswith(r["pickImg"]), [r["coverSrc"], r["pickImg"]])
if r["both"]:
    check(f"'{r['A']}:1' puts it first in {r['A']}", r["firstForest"] == r["both"], r)
    check(f"'{r['B']}:99' still before unnumbered {r['B']} paintings", r["firstSeasons"] == r["both"], r)
d.quit(); print("\n%d failure(s)" % len(fails)); sys.exit(1 if fails else 0)
