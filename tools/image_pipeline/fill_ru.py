#!/usr/bin/env python3
"""Fill the Russian columns of a site's Paintings sheet.

Usage:
  fill_ru.py --site sveta|olga --from translations.json          # dry run: prints the plan
  fill_ru.py --site sveta|olga --from translations.json --write  # writes

translations.json maps a painting's slug to its Russian text:
  {"held-summer": {"title_ru": "Удержанное лето", "description_ru": "..."}, ...}

Rules (the EN·RU switch, sites/README.md → Languages):
  - Columns title_ru, description_ru and ru_source are added at the end of the
    header if missing. Nothing else moves.
  - Only EMPTY cells are filled. A Russian cell with text in it is never
    overwritten, so Sveta's own edits survive any re-run. To get a fresh
    translation, empty the cell first.
  - ru_source records which English the Russian was made from: the first 8
    hex digits of sha1(title + "\n" + description). audit_gallery.py compares
    it with the current English and reports rows whose English changed since.
  - Only the changed cells are written (one batch), never whole rows.

Same OAuth token as push_painting.py (push-painting-token.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "scripts"))
import site_config  # noqa: E402

RU_COLS = ["title_ru", "description_ru", "ru_source"]


def ru_source(title: str, description: str) -> str:
    return hashlib.sha1(f"{title.strip()}\n{description.strip()}".encode("utf-8")).hexdigest()[:8]


def col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    site_config.add_site_arg(p)
    p.add_argument("--from", dest="src", required=True, type=Path, help="translations JSON, keyed by slug")
    p.add_argument("--write", action="store_true", help="write to the sheet (default: dry run)")
    args = p.parse_args()

    t = site_config.require(args.site, "sheetId", "sheetTab")
    sheet_id, tab = t["sheetId"], t["sheetTab"]
    print(site_config.describe(args.site))
    tr = json.loads(args.src.read_text(encoding="utf-8"))

    import push_painting as pp  # noqa: E402  (OAuth helper)
    _, sheets = pp.get_services()
    rows = sheets.spreadsheets().values().get(spreadsheetId=sheet_id, range=tab).execute().get("values", [])
    if not rows:
        sys.exit("the sheet is empty")
    header = rows[0]
    low = [h.strip().lower() for h in header]
    for need in ("title", "description", "slug"):
        if need not in low:
            sys.exit(f"no '{need}' column in {tab}")

    updates = []  # (a1, value)
    new_header = list(header)
    for c in RU_COLS:
        if c not in low:
            new_header.append(c)
            updates.append((f"{tab}!{col_letter(len(new_header) - 1)}1", c))
    idx = {h.strip().lower(): i for i, h in enumerate(new_header)}

    filled = kept = missing = 0
    used = set()
    for r, row in enumerate(rows[1:], start=2):
        cell = lambda name: (row[idx[name]] if idx[name] < len(row) else "").strip()
        slug, title = cell("slug"), cell("title")
        if not title:
            continue
        src = tr.get(slug)
        if not src:
            if not (cell("title_ru") and cell("description_ru")):
                print(f"  row {r} {slug}: no translation given — left empty")
                missing += 1
            continue
        used.add(slug)
        changed = False
        for c in ("title_ru", "description_ru"):
            if cell(c):
                kept += 1
                print(f"  row {r} {slug}: {c} already has text — kept")
            elif src.get(c):
                updates.append((f"{tab}!{col_letter(idx[c])}{r}", src[c]))
                changed = True
                filled += 1
        if changed:
            updates.append((f"{tab}!{col_letter(idx['ru_source'])}{r}", ru_source(title, cell("description"))))
            print(f"  row {r} {slug}: {title} → {src.get('title_ru', '')}")
    for s in sorted(set(tr) - used):
        print(f"  translation for '{s}' matches no row — skipped")

    print(f"{filled} cells to fill, {kept} kept as they are, {missing} rows without a translation, "
          f"{len(updates)} cell writes in total")
    if not args.write:
        print("dry run: nothing written (add --write)")
        return 0
    if updates:
        sheets.spreadsheets().values().batchUpdate(spreadsheetId=sheet_id, body={
            "valueInputOption": "RAW",
            "data": [{"range": a1, "values": [[v]]} for a1, v in updates],
        }).execute()
    print("written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
