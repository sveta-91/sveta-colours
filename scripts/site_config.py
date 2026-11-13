"""Per-site settings for the Python tools (docs/olgazai-plan.md, phase 4).

Every tool that touches a sheet, a Drive folder or a deployed site takes a
required --site (no default), so a bare command can never publish to, audit
or test the wrong artist's site. Settings come from sites/<name>/site.json:

  head.url               the deployed site
  site.sheetCsvUrl       the published CSV the site reads
  tools.sheetId          the spreadsheet the publishing tools write
  tools.sheetTab         its tab (normally "Paintings")
  tools.driveFolder      the public Drive folder for painting images
  tools.watermark        text for the photo manager's watermark
  tools.voiceSkill       writing skill for descriptions (null: neutral prompt,
                         and the audit skips the voice checks)
  tools.collections      known collection names (empty: not checked)
  tools.orderTestCollections  two collections that share a painting, for the
                         synthetic part of verify_collection_order.py
                         (empty: the test picks them itself)

Standard library only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITES = REPO / "sites"


def site_names() -> list[str]:
    return sorted(p.parent.name for p in SITES.glob("*/site.json"))


def load(name: str) -> dict:
    path = SITES / name / "site.json"
    if not path.is_file():
        sys.exit(f"no sites/{name}/site.json (known: {', '.join(site_names())})")
    return json.loads(path.read_text(encoding="utf-8"))


def tools(name: str) -> dict:
    return load(name).get("tools", {})


def require(name: str, *keys: str) -> dict:
    """The tools block, exiting with a clear message if any key is empty."""
    t = tools(name)
    gaps = [k for k in keys if not t.get(k)]
    if gaps:
        sys.exit(f"sites/{name}/site.json: tools.{', tools.'.join(gaps)} still empty — "
                 f"fill it in before using this tool for {name}")
    return t


def add_site_arg(parser) -> None:
    parser.add_argument(
        "--site", required=True, choices=site_names(),
        help="which artist's site (sites/<name>/site.json); required, no default")


def describe(name: str) -> str:
    """One line naming the target, printed before any tool acts."""
    c = load(name)
    t = c.get("tools", {})
    return (f"site: {name} ({c['head']['url']}) · sheet {t.get('sheetId') or '—'}"
            f" tab {t.get('sheetTab') or '—'} · Drive folder {t.get('driveFolder') or '—'}")
