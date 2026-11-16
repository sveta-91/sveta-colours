#!/usr/bin/env python3
"""Build the Google Drive starter kit for a new artist's site.

The artist uploads the kit to her own Drive, so she owns the sheet, photos
and texts; the publishing account (sveta.colours) gets Editor access later.
Runbook, in order, with who does each step: docs/new-artist-onboarding.md.

  python3 scripts/make_drive_kit.py --site olga --out /tmp/olga-kit
  python3 scripts/make_drive_kit.py --site olga --out /tmp/olga-kit \
      --csv paintings.csv --photos ~/Pictures/olga-cropped

Writes into --out (replaced if it already exists, after --force):

  Paintings.csv                   the sheet: header + one row per painting
                                  (image left empty; filled with Drive IDs
                                  once the photos are in her Drive)
  Paintings photos/<title>.<ext>  one per row, named by title so that
                                  `drive_tool.py ids` maps title -> Drive ID
  Site texts/<domain> site text: home.txt, about.txt, faq.txt (+ " (RU)")
                                  from sites/<site>/texts[/ru], [label] format

--csv: a CSV with the sheet's columns (title, medium, price, … at least
title). Without it the sheet is the header only. --photos: a folder with
one image per row, named by slug or title (any of .jpg .jpeg .png .webp).
In Drive: open each .csv/.txt with Google Sheets / Google Docs, so it can
be published to the web. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = ["title", "medium", "price", "sold", "size", "image", "description",
        "collection", "orientation", "hero", "recent", "slug"]
RU = ["title_ru", "description_ru", "ru_source"]
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def slugify(s: str) -> str:   # same rule as index.html slugify()
    return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", s.lower()))


def ru_source(title: str, description: str) -> str:   # = fill_ru.ru_source
    return hashlib.sha1(f"{title.strip()}\n{description.strip()}".encode("utf-8")).hexdigest()[:8]


def safe_name(s: str) -> str:
    return re.sub(r'[/\\:]+', "-", s).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sites = sorted(x.parent.name for x in (REPO / "sites").glob("*/site.json"))
    p.add_argument("--site", required=True, choices=sites)
    p.add_argument("--out", required=True, type=Path, help="kit folder to write")
    p.add_argument("--csv", type=Path, help="painting rows (sheet columns)")
    p.add_argument("--photos", type=Path, help="folder of photos named by slug or title")
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    a = p.parse_args()

    site = json.loads((REPO / "sites" / a.site / "site.json").read_text(encoding="utf-8"))
    domain = site.get("site", {}).get("domain") or a.site
    if a.out.exists():
        if not a.force:
            sys.exit(f"{a.out} exists; add --force to replace it")
        shutil.rmtree(a.out)
    photos_dir, texts_dir = a.out / "Paintings photos", a.out / "Site texts"
    photos_dir.mkdir(parents=True)
    texts_dir.mkdir()

    rows: list[dict] = []
    extra: list[str] = []
    if a.csv:
        with a.csv.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            cols = [c.strip().lower() for c in rd.fieldnames or []]
            if "title" not in cols:
                sys.exit(f"{a.csv}: no title column")
            extra = [c for c in cols if c not in BASE + RU]
            rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
                    for r in rd if (r.get("title") or "").strip()]
    header = BASE + extra + RU

    missing = []
    for r in rows:
        r.setdefault("slug", "")
        r["slug"] = r["slug"] or slugify(r["title"])
        r["image"] = ""                       # Drive ID, filled after upload
        if r.get("title_ru") or r.get("description_ru"):
            r["ru_source"] = ru_source(r["title"], r.get("description", ""))
        if a.photos:
            src = next((a.photos / f"{stem}{ext}" for stem in (r["slug"], r["title"])
                        for ext in EXTS if (a.photos / f"{stem}{ext}").is_file()), None)
            if src:
                shutil.copy2(src, photos_dir / f"{safe_name(r['title'])}{src.suffix.lower()}")
            else:
                missing.append(r["title"])

    with (a.out / "Paintings.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])

    n_texts = 0
    for suffix, d in (("", REPO / "sites" / a.site / "texts"),
                      (" (RU)", REPO / "sites" / a.site / "texts" / "ru")):
        for name in ("home", "about", "faq"):
            f = d / f"{name}.txt"
            if f.is_file():
                shutil.copy2(f, texts_dir / f"{domain} site text: {name}{suffix}.txt")
                n_texts += 1

    print(f"{a.out}/")
    print(f"  Paintings.csv       {len(rows)} rows, columns: {', '.join(header)}")
    print(f"  Paintings photos/   {len(list(photos_dir.iterdir()))} files")
    print(f"  Site texts/         {n_texts} files")
    if missing:
        print(f"  no photo for: {', '.join(missing)}")
    if not (REPO / "sites" / a.site / "texts").is_dir():
        print(f"  sites/{a.site}/texts/ is missing: write the backup texts first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
