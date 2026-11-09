#!/usr/bin/env python3
"""Collection order + cover test.

Usage (from the repo root):
    tools/image_pipeline/.venv/bin/python scripts/verify_collection_order.py            # live site
    tools/image_pipeline/.venv/bin/python scripts/verify_collection_order.py http://localhost:8765/
Run it live: the image CDN refuses localhost, but order checks work either way.
Exit code 0 = all pass, 1 = any failure.

Part 1 checks the site against Sveta's REAL sheet data, whatever she has
numbered: for every collection, the test works out the expected order and
button picture on its own (from the raw collection / order / cover fields)
and compares with what the page shows. Nothing assumes a collection is
un-numbered, so reordering in the sheet doesn't break it.

Part 2 wipes the Forest ordering in memory (never in the sheet) and drives
the rules with synthetic values: blank → sheet order, 1..n reverses, cover
picks the picture, "Forest:1,Seasons:99" moves one painting in both.
"""
import sys, time, json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
BASE = sys.argv[1] if len(sys.argv) > 1 else "https://www.sveta-colours.com/"
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

# ---- Part 2: synthetic rules on Forest (in memory only)
r = d.execute_script(EXPECT_JS + """
  const out = {};
  const forest = PAINTINGS.filter(p => p.collections.includes('Forest'));
  const wipe = () => forest.forEach(p => { p.order = ''; p.cover = []; p.collOrder = {}; });
  const go = n => { showPage('collection:' + n); return titles(); };
  out.sheetOrder = forest.map(p => p.title);
  wipe(); render(); out.blank = go('Forest');
  out.blankCover = thumbSrc('Forest'); out.blankExpect = (forest.find(p => p.image && !p.sold) || forest[0]).image;
  forest.forEach((p, i) => { p.order = String(forest.length - i); });
  const pick = forest[forest.length - 1]; pick.cover = ['forest'];
  render(); out.reversed = go('Forest'); out.coverSrc = thumbSrc('Forest'); out.pickImg = pick.image;
  wipe();
  const both = PAINTINGS.find(p => p.collections.includes('Forest') && p.collections.includes('Seasons'));
  if (both) { both.collOrder = {}; both.order = 'Forest:1,Seasons:99'; }
  PAINTINGS.filter(p => p.collections.includes('Seasons') && p !== both).forEach(p => { p.order = ''; p.collOrder = {}; });
  render();
  out.both = both ? both.title : null; out.firstForest = go('Forest')[0]; out.firstSeasons = go('Seasons')[0];
  return out;""")
check("blank: page keeps sheet order", r["blank"] == r["sheetOrder"], r)
check("blank: picture = first available painting", r["blankCover"].startswith(r["blankExpect"]), r)
check("order 1..n reverses the collection", r["reversed"] == list(reversed(r["sheetOrder"])), r["reversed"])
check("cover column picks the button picture", r["coverSrc"].startswith(r["pickImg"]), [r["coverSrc"], r["pickImg"]])
if r["both"]:
    check("'Forest:1' puts it first in Forest", r["firstForest"] == r["both"], r)
    check("'Seasons:99' still before unnumbered Seasons paintings", r["firstSeasons"] == r["both"], r)
d.quit(); print("\n%d failure(s)" % len(fails)); sys.exit(1 if fails else 0)
