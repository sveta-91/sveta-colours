#!/usr/bin/env python3
"""Compare the published Google Docs with their backup .txt copies, and update the copies.

The site reads its long texts (home, about, faq) from published Google Docs
(site.json -> site.textDocs, and site.textDocsRu for the Russian version). The
plain-text files are the backup the site falls back to when a Doc can't be read
(files.texts in site.json; texts/ for Sveta, sites/olga/texts for Olga; Russian
in <that>/ru/). The backups don't update by themselves: run this after a lasting
change to a Doc.

  --check   (default) report which files differ, block by block; writes nothing.
            Exit code 1 if any file differs or a Doc can't be read.
  --write   update the backup files whose blocks differ from the Doc.

The Doc is read the way index.html reads it (docPageToText + parseTextFile):
the paragraphs inside #contents, one line each, then [label] blocks. Two texts
count as equal when their blocks are equal, so trailing empty paragraphs and
spaces don't matter.

A Doc edit reaches the published page within seconds for this script, but a
browser may show the old copy for ~5 minutes (docs/portfolio-playbook.md ->
"Long texts in plain .txt files").

Standard library only. Examples:
  python3 scripts/sync_texts.py --site sveta
  python3 scripts/sync_texts.py --site sveta --lang en --write
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_config  # noqa: E402

# Mirrors TEXT_LABELS in index.html. Keep the two in step.
TEXT_LABELS = {
    "home": ["headline", "under headline", "from the studio", "on painting"],
    "about": ["about"],
    "faq": ["question", "answer"],
}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}


class DocText(HTMLParser):
    """Text of the paragraph-like elements inside <div id="contents">, in order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0          # div nesting inside #contents (0 = outside)
        self.cur: list[str] | None = None
        self.block_depth = 0
        self.lines: list[str] = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        if self.depth:
            if tag == "div":
                self.depth += 1
            if tag in BLOCK_TAGS:
                if self.block_depth == 0:
                    self.cur = []
                self.block_depth += 1
        elif tag == "div" and dict(attrs).get("id") == "contents":
            self.depth, self.found = 1, True

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag in BLOCK_TAGS and self.block_depth:
            self.block_depth -= 1
            if self.block_depth == 0 and self.cur is not None:
                self.lines.append("".join(self.cur))
                self.cur = None
        elif tag == "div":
            self.depth -= 1

    def handle_data(self, data):
        if self.cur is not None:
            self.cur.append(data)


def doc_to_text(html: str) -> str:
    p = DocText()
    p.feed(html)
    if not p.found:
        raise ValueError("no #contents in the published page")
    return "\n".join(l.replace(" ", " ").replace("​", "").replace("﻿", "") for l in p.lines)


def parse_blocks(name: str, raw: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Same rules as parseTextFile in index.html. Returns (blocks, warnings)."""
    blocks, warns, cur = [], [], None
    for line in raw.lstrip("﻿").splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if m:
            label = re.sub(r"\s+", " ", m.group(1).strip().lower())
            if label in TEXT_LABELS[name]:
                cur = [label, []]
                blocks.append(cur)
            else:
                cur = None
                warns.append(f"unknown label [{m.group(1).strip()}] — the site skips this block")
        elif cur is not None:
            cur[1].append(line)
        elif line.strip():
            warns.append(f'text before the first [label] is skipped: "{line.strip()[:40]}"')
    out = [(lab, "\n".join(lines).strip()) for lab, lines in blocks]
    return [b for b in out if b[1]], warns


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "sveta-colours sync_texts"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8")


def file_text(blocks_raw: str) -> str:
    """What gets written: the Doc's lines, trailing blanks trimmed, one final newline."""
    lines = [l.rstrip() for l in blocks_raw.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    site_config.add_site_arg(ap)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report differences only (default)")
    mode.add_argument("--write", action="store_true", help="update backup files that differ")
    ap.add_argument("--lang", choices=["en", "ru", "all"], default="all")
    ap.add_argument("--only", choices=sorted(TEXT_LABELS), help="one file (home, about or faq)")
    args = ap.parse_args()

    cfg = site_config.load(args.site)
    base = site_config.REPO / cfg.get("files", {}).get("texts", "texts")
    sets = []
    if args.lang in ("en", "all"):
        sets.append(("en", cfg["site"].get("textDocs") or {}, base))
    if args.lang in ("ru", "all"):
        sets.append(("ru", cfg["site"].get("textDocsRu") or {}, base / "ru"))

    problems = 0
    for lang, docs, folder in sets:
        if not docs:
            print(f"[{lang}] no Docs in sites/{args.site}/site.json — skipped")
            continue
        for name in TEXT_LABELS:
            if args.only and name != args.only:
                continue
            path = folder / f"{name}.txt"
            rel = path.relative_to(site_config.REPO)
            url = docs.get(name)
            if not url:
                print(f"[{lang}] {name}: no Doc link — skipped")
                continue
            try:
                doc_raw = doc_to_text(fetch(url))
            except Exception as e:  # network, HTTP, page shape
                print(f"FAIL [{lang}] {name}: Doc can't be read ({e})")
                problems += 1
                continue
            doc_blocks, warns = parse_blocks(name, doc_raw)
            for w in warns:
                print(f"WARN [{lang}] {name} Doc: {w}")
            old = path.read_text(encoding="utf-8") if path.is_file() else ""
            file_blocks, _ = parse_blocks(name, old)
            if doc_blocks == file_blocks:
                print(f"SAME [{lang}] {name}: {rel} matches the Doc ({len(doc_blocks)} blocks)")
                continue
            if not doc_blocks:
                print(f"FAIL [{lang}] {name}: the Doc has no [label] blocks — not touching {rel}")
                problems += 1
                continue
            print(f"DIFF [{lang}] {name}: {rel} ({len(file_blocks)} blocks) vs Doc ({len(doc_blocks)} blocks)")
            fmt = lambda bs: [f"[{l}]\n{t}\n" for l, t in bs]
            for line in difflib.unified_diff("".join(fmt(file_blocks)).splitlines(),
                                             "".join(fmt(doc_blocks)).splitlines(),
                                             f"{rel}", "Doc", lineterm="", n=1):
                print("    " + line)
            if args.write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(file_text(doc_raw), encoding="utf-8")
                print(f"     wrote {rel}")
            else:
                problems += 1
    if not args.write and problems:
        print(f"\n{problems} file(s) differ or failed. Run with --write to update the backups.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
