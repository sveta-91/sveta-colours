#!/usr/bin/env python3
"""Read-only pre-deploy audit of the Paintings sheet behind sveta-colours.com.

The site (index.html) is a single-page app that fetches the published Google
Sheet as CSV and renders straight from it. There is no build step and no
schema validation anywhere in the chain, so a subtly wrong cell — a missing
`size`, two rows marked `hero`, an `orientation` that contradicts the canvas
dimensions — silently breaks layout with no error anywhere. Every such bug in
this repo's history cost 20-90 minutes to find by eye.

This script catches that class of bug mechanically. Run it before deploying
(i.e. after editing the sheet, before pushing / before telling Sveta the site
is updated):

    python3 scripts/audit_gallery.py                    # fetch live CSV
    python3 scripts/audit_gallery.py --csv paintings.csv  # offline
    python3 scripts/audit_gallery.py --json             # machine-readable
    python3 scripts/audit_gallery.py --check-drive      # + Drive cross-check

Default mode is CSV-only, standard library only, no authentication, and no
venv required. Google API usage is lazy and opt-in behind `--check-drive`.

The script never writes to the sheet, to Drive, or to disk. Output is stdout.

Ground truth for what is checked is `index.html` itself, not the docs — the
checks below validate what `sheetRowToPainting()` and `renderPaintings()`
actually depend on. Two places where index.html differs from the prose docs
are noted inline (`NOTE (index.html vs docs)`).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence

# --------------------------------------------------------------------------
# Constants — mirrored from the live site and the pipeline CLI.
# --------------------------------------------------------------------------

# index.html: const SHEET_CSV_URL = ...
CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQrJdmLlZeUwFysVsa6rmHi"
    "Enby2moRMSYcBEixMNb3AoLxdpnvrSaWlxC23TF68VMTV3UDuQbuWOXV/pub?output=csv"
)

# tools/image_pipeline/push_painting.py: COLS (11 columns, as of 2026-06-09)
EXPECTED_HEADER: tuple[str, ...] = (
    "title", "medium", "price", "sold", "size", "image",
    "description", "collection", "orientation", "hero", "recent",
)

# tools/image_pipeline/push_painting.py: DRIVE_FOLDER
DRIVE_FOLDER = "1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX"

# README "Constants" table + index.html sheetRowToPainting().
VALID_ORIENTATIONS = frozenset({"portrait", "landscape"})

# index.html `yes()` accepts TRUE/true/yes/1/y; anything else reads as false.
# push_painting.py writes 'yes' / 'no'. Accept both spellings plus empty.
VALID_SOLD = frozenset({"", "yes", "no", "true", "false", "y", "n", "1", "0"})

# README flags reference: existing collection values. Novel values are
# reported as INFO, not an error — she may legitimately add a collection.
KNOWN_COLLECTIONS = frozenset({"Big City", "Flowers", "Forest", "Seasons"})

# index.html selectFeatured(paintings, 6, ...) — the home strip holds 6.
RECENT_MIN, RECENT_MAX = 1, 6

# Description length window, from .claude/skills/sveta-writing/SKILL.md
# ("Length": 32-55 words, hard ceiling 60). The audit uses the outer bounds.
DESC_MIN_WORDS, DESC_MAX_WORDS = 32, 60

# Titles whose description Sveta wrote herself. The 32-word floor exists to
# stop a drafted description from reading like a caption — it has no business
# policing her own writing, which is the source the spec was derived FROM
# (ruling 2026-08-22: "нижняя граница снимается, когда текст пишу я").
# The 60-word ceiling still applies to everyone. Add a title here whenever she
# supplies the text; match is case-insensitive.
SVETA_AUTHORED = {
    "vancouver",
}

# Em-dash cap, from SKILL.md "Sentence shapes": 1-2 per description, never 3.
MAX_EM_DASHES = 2

# --------------------------------------------------------------------------
# Ban list.
#
# SOURCE OF TRUTH: .claude/skills/sveta-writing/SKILL.md, section
# "Forbidden moves (cut on sight)". This constant is a mechanical, literal
# subset of that section — only phrases a regex can decide. Judgement calls
# from the skill ("three-act arcs ending in feeling", "symbolism announced",
# "title-restating") are deliberately NOT encoded here; this script never
# tries to judge voice. When the skill's ban list changes, update this
# constant to match.
# --------------------------------------------------------------------------
BANNED_PHRASES: tuple[str, ...] = (
    # Invitation verbs
    "invites", "draws you in", "beckons",
    # Art-statement vocabulary
    "explores", "captures", "evokes", "conveys", "expresses", "renders",
    "juxtaposes", "interplay", "narrative", "journey", "essence",
    "themes of", "meditation on", "reflection on", "a study in",
    # AI-tell abstract nouns
    "interplay of", "play of light", "dance of", "symphony of",
    "tapestry of", "weaving", "a sense of", "a feeling of",
    "an atmosphere of", "a moment of grace", "a quiet hope",
    # Mood prescription
    "peaceful", "serene", "calm", "cozy", "contemplative", "nostalgic",
    "wistful", "bittersweet", "joyful", "vibrant", "luminous", "ethereal",
    "evocative",
    # Buyer projection / second person
    "you'll feel", "imagine yourself", "brings to mind", "reminds one of",
    "for the collector who", "perfect for",
    # Gallery-card distance
    "this piece", "this work", "the painting", "the artist",
    # Color filler
    "rich with", "imbued with", "suffused with", "awash in", "drenched in",
    "bathed in",
    # Domain clichés
    "golden hour", "kissed by sunlight", "whispers of autumn",
    "snow-kissed", "a riot of color", "solitary figure", "bustling city",
    # Sentence-fragment poetry / register
    "stillness. warmth.",
)

# Called out separately by the skill as a hard ban ("The word *beauty*.").
BEAUTY_WORDS: tuple[str, ...] = ("beauty", "beautiful", "stunning")

SECOND_PERSON: tuple[str, ...] = ("you", "your", "yours", "you're", "yourself")

SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One audit result. `row` is the 1-based sheet row number (header = 1)."""
    severity: str          # ERROR | WARN | INFO
    check: str             # short machine-stable id
    title: str             # painting title, or "" for sheet-level findings
    column: str            # sheet column the finding is about, or ""
    message: str
    row: int | None = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

EXIT_LOAD_FAILURE = 2


def die(message: str) -> NoReturn:
    """Print a load-failure message to stderr and exit 2 (never 0/1)."""
    print(message, file=sys.stderr)
    raise SystemExit(EXIT_LOAD_FAILURE)


def fetch_csv(url: str = CSV_URL, timeout: int = 30) -> str:
    """Fetch the published CSV with stdlib urllib. Exits on any failure."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "sveta-colours-audit/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        die(
            f"ERROR: sheet fetch failed with HTTP {exc.code} {exc.reason}.\n"
            f"       Is the sheet still published to the web?\n"
            f"       URL: {url}"
        )
    except (urllib.error.URLError, OSError) as exc:
        die(
            f"ERROR: could not reach the published sheet ({exc}).\n"
            f"       Check network access, or run offline with --csv PATH."
        )

    text = raw.decode("utf-8-sig", errors="replace")
    head = text.lstrip()[:200].lower()
    if "text/html" in ctype.lower() or head.startswith(("<!doctype", "<html")):
        die(
            "ERROR: the publish URL returned HTML, not CSV.\n"
            "       Google serves an HTML error/sign-in page when the sheet\n"
            "       is unpublished or access changed. Re-publish the\n"
            "       'Paintings' tab as CSV (File > Share > Publish to web)."
        )
    if not text.strip():
        die("ERROR: the published sheet returned an empty body.")
    return text


def read_local_csv(path: Path) -> str:
    """Read a local CSV file. Exits on failure."""
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        die(f"ERROR: could not read --csv {path}: {exc}")


def parse_csv(text: str) -> tuple[list[str], list[list[str]]]:
    """Parse CSV into (header, data_rows).

    Mirrors index.html parseCSV(): cells are trimmed and all-empty rows are
    dropped, so blank spacer rows in the sheet are not reported as errors.
    """
    rows: list[list[str]] = []
    for raw_row in csv.reader(io.StringIO(text)):
        row = [c.strip() for c in raw_row]
        if any(c != "" for c in row):
            rows.append(row)
    if not rows:
        die("ERROR: CSV contained no non-empty rows.")
    return rows[0], rows[1:]


def normalize_header(name: str) -> str:
    """index.html: h.toLowerCase().replace(/[^a-z]/g,'')"""
    return re.sub(r"[^a-z]", "", name.lower())


def row_to_dict(header: Sequence[str], row: Sequence[str]) -> dict[str, str]:
    """Map a row onto normalized header keys, index.html's `get()` semantics."""
    out: dict[str, str] = {}
    for i, name in enumerate(header):
        key = normalize_header(name)
        if key and key not in out:
            out[key] = row[i].strip() if i < len(row) else ""
    return out


# --------------------------------------------------------------------------
# Shared parsing helpers (kept aligned with index.html)
# --------------------------------------------------------------------------

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_size(size_str: str, orientation: str) -> tuple[float, float] | None:
    """Derive (widthIn, heightIn) exactly the way index.html does.

    index.html prefers explicit `widthin`/`heightin` columns, but the live
    sheet has no such columns — dimensions come from the `size` string, with
    the `orientation` column deciding which number is width. Defaults to
    portrait when orientation is blank. If this returns None, index.html's
    sizeChapterFor() reports 'unknown' and the painting drops into the
    "Other works" chapter (the commit-7561047 bug).
    """
    m = SIZE_RE.search(size_str or "")
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return None
    smaller, larger = min(a, b), max(a, b)
    orient = (orientation or "").lower().strip()
    # NOTE (index.html vs docs): index.html also accepts 'wide' and 'l' as
    # landscape aliases. The schema only sanctions portrait/landscape, so the
    # orientation check below still flags them — but dimension derivation
    # here matches the site so cross-field checks agree with what renders.
    if orient in ("landscape", "wide", "l"):
        return larger, smaller
    return smaller, larger


def truthy(value: str) -> bool:
    """index.html yes(): TRUE / true / yes / 1 / y."""
    return (value or "").strip().lower() in ("yes", "true", "1", "y")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’\-]+", text or ""))


def count_em_dashes(text: str) -> int:
    return (text or "").count("—")


def find_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    """Case-insensitive, word-boundary-anchored phrase match."""
    low = (text or "").lower()
    hits: list[str] = []
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+")
        # Only require a trailing boundary when the phrase ends in a word char.
        if re.match(r"[\w]", phrase[-1]):
            pattern += r"\b"
        if re.search(pattern, low):
            hits.append(phrase)
    return hits


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

# Optional columns allowed after the 11 required ones, in any order.
OPTIONAL_COLUMNS: tuple[str, ...] = ("cover", "order", "slug")


def check_header(header: Sequence[str]) -> list[Finding]:
    """ERROR: the first 11 columns are not the expected ones, in order, or an
    extra column is not one of OPTIONAL_COLUMNS."""
    actual = [normalize_header(h) for h in header]
    expected = [normalize_header(h) for h in EXPECTED_HEADER]
    extra = actual[len(expected):]
    if actual[:len(expected)] == expected and all(
            h in OPTIONAL_COLUMNS or h == "" for h in extra):
        return []
    return [Finding(
        "ERROR", "header-mismatch", "", "header",
        f"header is {header!r}; expected {list(EXPECTED_HEADER)!r} in order",
        row=1,
    )]


def check_rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[Finding]:
    """Per-row and cross-row checks. Returns findings in row order."""
    findings: list[Finding] = []
    records: list[tuple[int, str, dict[str, str]]] = []

    for offset, raw in enumerate(rows):
        sheet_row = offset + 2  # header is row 1
        d = row_to_dict(header, raw)
        title = d.get("title", "")
        label = title or f"<row {sheet_row}>"
        records.append((sheet_row, label, d))

        # ---------- schema / structural (ERROR) ----------
        if not title:
            findings.append(Finding(
                "ERROR", "missing-title", label, "title",
                "row has no title — index.html filters it out entirely "
                "(`.filter(p => p.title)`), so the painting never renders",
                sheet_row,
            ))
        if not d.get("image", ""):
            findings.append(Finding(
                "ERROR", "missing-image", label, "image",
                "no Drive file ID — card renders with a broken/empty image, "
                "and the row is excluded from hero and recent selection",
                sheet_row,
            ))

        size_str = d.get("size", "")
        orientation = d.get("orientation", "")
        dims = parse_size(size_str, orientation)
        if not size_str:
            findings.append(Finding(
                "ERROR", "missing-size", label, "size",
                "size is empty — index.html cannot derive widthIn/heightIn, "
                "so the painting falls into the 'Other works' chapter and "
                "its card loses canvas-aspect lock",
                sheet_row,
            ))
        elif dims is None:
            findings.append(Finding(
                "ERROR", "unparseable-size", label, "size",
                f"size {size_str!r} does not match the NNxNN pattern "
                "index.html parses — painting drops to 'Other works'",
                sheet_row,
            ))

        if orientation and orientation.strip().lower() not in VALID_ORIENTATIONS:
            findings.append(Finding(
                "ERROR", "bad-orientation", label, "orientation",
                f"orientation is {orientation!r}; expected exactly "
                "'portrait' or 'landscape'",
                sheet_row,
            ))
        elif orientation and orientation not in VALID_ORIENTATIONS:
            # index.html lowercases and trims, so this renders fine — but
            # push_painting.py writes the canonical lowercase form.
            findings.append(Finding(
                "INFO", "orientation-casing", label, "orientation",
                f"orientation is {orientation!r}; canonical form is "
                f"{orientation.strip().lower()!r}",
                sheet_row,
            ))

        sold = d.get("sold", "")
        if sold.strip().lower() not in VALID_SOLD:
            findings.append(Finding(
                "ERROR", "bad-sold", label, "sold",
                f"sold is {sold!r}; expected one of "
                f"{sorted(v for v in VALID_SOLD if v)} or empty. index.html "
                "treats anything unrecognised as NOT sold",
                sheet_row,
            ))

        # ---------- cross-field consistency (WARN) ----------
        # Canvas-aspect-lock invariant (commit d487adc): the card's rendered
        # ratio is widthIn/heightIn, so orientation and size must agree.
        #
        # NOTE (index.html vs docs): the playbook says to derive isLandscape
        # from widthIn/heightIn, implying size and orientation are
        # independent signals that can contradict. They are not. index.html
        # derives widthIn/heightIn FROM the orientation column (the live
        # sheet has no widthIn/heightIn columns), and push_painting.py
        # documents that `size` stores the two numbers "in whatever order
        # Sveta typed" — verified: all 18 live rows are smaller-first
        # regardless of orientation. So "size says landscape but orientation
        # says portrait" is not a detectable condition; comparing the raw
        # number order fires on every landscape row and is pure noise.
        #
        # What IS detectable, and is the real canvas-aspect-lock risk
        # (commit d487adc), is orientation being absent or square-ambiguous:
        # index.html then silently defaults to portrait, so a landscape
        # painting renders in a 280px portrait slot at the wrong ratio.
        m = SIZE_RE.search(size_str or "")
        orient_norm = orientation.strip().lower()
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if not orient_norm and a != b:
                findings.append(Finding(
                    "WARN", "orientation-missing", label, "orientation",
                    f"orientation is blank on a non-square canvas "
                    f"({size_str!r}) — index.html silently defaults to "
                    "portrait, so a landscape painting would render in a "
                    "280px portrait card at the wrong canvas ratio",
                    sheet_row,
                ))
            elif a == b and orient_norm == "landscape":
                findings.append(Finding(
                    "WARN", "orientation-size-mismatch", label, "orientation",
                    f"size {size_str!r} is square but orientation is "
                    "'landscape' — the card takes the 350px landscape slot "
                    "with a 1:1 canvas ratio, breaking row rhythm",
                    sheet_row,
                ))

        if truthy(sold) and d.get("price", "").strip():
            findings.append(Finding(
                "INFO", "price-on-sold", label, "price",
                f"sold=yes but price is still {d.get('price')!r} — "
                "index.html renders 'Sold' and ignores it; every other sold "
                "row leaves price empty",
                sheet_row,
            ))

        if not d.get("description", "").strip():
            findings.append(Finding(
                "WARN", "missing-description", label, "description",
                "no description — modal and card copy render empty",
                sheet_row,
            ))
        if not d.get("price", "").strip() and not truthy(sold):
            findings.append(Finding(
                "WARN", "missing-price", label, "price",
                "no price on an unsold painting — renders as "
                "'Price on request'",
                sheet_row,
            ))

        # ---------- collections ----------
        collection_raw = d.get("collection", "")
        if collection_raw:
            if re.search(r",\s+", collection_raw):
                findings.append(Finding(
                    "INFO", "collection-spacing", label, "collection",
                    f"{collection_raw!r} has a space after a comma — the "
                    "convention is comma-separated with no spaces "
                    "(index.html trims, so this is cosmetic in the sheet)",
                    sheet_row,
                ))
            # "Big City:1" = collection + position on its page (index.html orderIn)
            for name in [re.sub(r"\s*:\s*\d+(\.\d+)?$", "", c.strip()) for c in collection_raw.split(",") if c.strip()]:
                if name not in KNOWN_COLLECTIONS:
                    findings.append(Finding(
                        "INFO", "unknown-collection", label, "collection",
                        f"collection {name!r} is not one of "
                        f"{sorted(KNOWN_COLLECTIONS)} — fine if newly added, "
                        "but it creates its own /collections chapter",
                        sheet_row,
                    ))

        # ---------- description quality (INFO, mechanical only) ----------
        findings.extend(check_description(label, d.get("description", ""), sheet_row))

    findings.extend(check_duplicates(records))
    findings.extend(check_slugs(records))
    findings.extend(check_hero(records))
    findings.extend(check_recent(records))
    return findings


def check_description(label: str, desc: str, sheet_row: int) -> list[Finding]:
    """Mechanically decidable subset of the sveta-writing rules only.

    No attempt is made to judge voice — see the BANNED_PHRASES comment.
    """
    if not desc.strip():
        return []  # already reported as a WARN
    out: list[Finding] = []

    wc = word_count(desc)
    # The floor is waived for descriptions Sveta wrote herself; the ceiling
    # is not. See SVETA_AUTHORED.
    sveta_wrote_it = label.strip().lower() in SVETA_AUTHORED
    floor = 0 if sveta_wrote_it else DESC_MIN_WORDS
    if wc < floor or wc > DESC_MAX_WORDS:
        target = (
            f"at most {DESC_MAX_WORDS} (floor waived — Sveta's own text)"
            if sveta_wrote_it
            else f"{DESC_MIN_WORDS}-{DESC_MAX_WORDS}"
        )
        out.append(Finding(
            "INFO", "desc-length", label, "description",
            f"{wc} words — sveta-writing target is {target}",
            sheet_row,
        ))

    dashes = count_em_dashes(desc)
    if dashes > MAX_EM_DASHES:
        out.append(Finding(
            "INFO", "desc-em-dashes", label, "description",
            f"{dashes} em-dashes — sveta-writing caps at {MAX_EM_DASHES}",
            sheet_row,
        ))

    banned = find_phrases(desc, BANNED_PHRASES)
    if banned:
        out.append(Finding(
            "INFO", "desc-banned-phrase", label, "description",
            "forbidden phrase(s): " + ", ".join(repr(b) for b in banned),
            sheet_row,
        ))

    beauty = find_phrases(desc, BEAUTY_WORDS)
    if beauty:
        out.append(Finding(
            "INFO", "desc-beauty-word", label, "description",
            "hard-banned word(s): " + ", ".join(repr(b) for b in beauty),
            sheet_row,
        ))

    second = find_phrases(desc, SECOND_PERSON)
    if second:
        out.append(Finding(
            "INFO", "desc-second-person", label, "description",
            "second person: " + ", ".join(repr(s) for s in second)
            + " — sveta-writing forbids 'you'",
            sheet_row,
        ))

    return out


def check_duplicates(records: Sequence[tuple[int, str, dict[str, str]]]) -> list[Finding]:
    """ERROR: duplicate titles, case-insensitively (push_painting.py's match)."""
    seen: dict[str, int] = {}
    out: list[Finding] = []
    for sheet_row, label, d in records:
        title = d.get("title", "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen:
            out.append(Finding(
                "ERROR", "duplicate-title", label, "title",
                f"duplicate of row {seen[key]} (case-insensitive) — "
                "push_painting.py matches on lowercased title and would "
                "only ever update the first one",
                sheet_row,
            ))
        else:
            seen[key] = sheet_row
    return out


SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def check_slugs(records: Sequence[tuple[int, str, dict[str, str]]]) -> list[Finding]:
    """When the sheet has a `slug` column: every painting has one, it is
    lower-case-dashed, and no two are the same. Slugs are permanent link keys."""
    if not records or "slug" not in records[0][2]:
        return []
    seen: dict[str, int] = {}
    out: list[Finding] = []
    for sheet_row, label, d in records:
        if not d.get("title", "").strip():
            continue
        slug = d.get("slug", "").strip()
        if not slug:
            out.append(Finding("ERROR", "slug-missing", label, "slug",
                               "no slug: links and the future /paintings/<slug> URL need one",
                               sheet_row))
        elif not SLUG_RE.match(slug):
            out.append(Finding("ERROR", "slug-format", label, "slug",
                               f"{slug!r} must be lower-case letters, digits and single dashes",
                               sheet_row))
        elif slug in seen:
            out.append(Finding("ERROR", "slug-duplicate", label, "slug",
                               f"{slug!r} is also used on row {seen[slug]}", sheet_row))
        else:
            seen[slug] = sheet_row
    return out


def check_hero(records: Sequence[tuple[int, str, dict[str, str]]]) -> list[Finding]:
    """ERROR: zero or multiple hero rows."""
    heroes = [(r, lbl) for r, lbl, d in records if truthy(d.get("hero", ""))]
    if not heroes:
        return [Finding(
            "ERROR", "no-hero", "", "hero",
            "no row has hero set — the home page silently falls back to the "
            "hardcoded HERO_PIN_TITLE constant in index.html",
        )]
    if len(heroes) > 1:
        where = ", ".join(f"{lbl} (row {r})" for r, lbl in heroes)
        return [Finding(
            "ERROR", "multiple-heroes", "", "hero",
            f"{len(heroes)} rows have hero set: {where}. index.html takes the "
            "first showable one, so which painting fronts the site depends "
            "on sheet row order",
        )]
    return []


def check_recent(records: Sequence[tuple[int, str, dict[str, str]]]) -> list[Finding]:
    """ERROR: duplicate ranks, out-of-range ranks, non-contiguous ranks."""
    out: list[Finding] = []
    ranks: dict[int, list[tuple[int, str]]] = {}

    for sheet_row, label, d in records:
        raw = d.get("recent", "").strip()
        if not raw:
            continue
        try:
            n = int(raw)
        except ValueError:
            out.append(Finding(
                "ERROR", "bad-recent", label, "recent",
                f"recent is {raw!r}, not an integer — index.html parseInt "
                "yields NaN and the painting drops out of the home strip",
                sheet_row,
            ))
            continue
        if n < RECENT_MIN or n > RECENT_MAX:
            out.append(Finding(
                "ERROR", "recent-out-of-range", label, "recent",
                f"recent={n} is outside {RECENT_MIN}-{RECENT_MAX} — the home "
                "strip renders at most 6 positions",
                sheet_row,
            ))
        ranks.setdefault(n, []).append((sheet_row, label))

    for n in sorted(ranks):
        entries = ranks[n]
        if len(entries) > 1:
            where = ", ".join(f"{lbl} (row {r})" for r, lbl in entries)
            out.append(Finding(
                "ERROR", "duplicate-recent", "", "recent",
                f"recent={n} claimed by {len(entries)} rows: {where} — "
                "carousel order becomes row-order dependent",
            ))

    present = sorted(n for n in ranks if RECENT_MIN <= n <= RECENT_MAX)
    if present:
        expected = list(range(RECENT_MIN, RECENT_MIN + len(present)))
        if present != expected:
            missing = [n for n in expected if n not in present]
            out.append(Finding(
                "ERROR", "recent-not-contiguous", "", "recent",
                f"recent ranks are {present}; expected a contiguous run "
                f"starting at {RECENT_MIN} ({expected}). Missing: {missing}",
            ))
        elif len(present) < RECENT_MAX:
            # index.html selectFeatured() short-circuits on the sheet path as
            # soon as ONE row has recent set — the constant fallback never
            # runs, so the home strip renders exactly this many cards.
            out.append(Finding(
                "INFO", "recent-underfilled", "", "recent",
                f"only {len(present)} of {RECENT_MAX} recent slots are "
                f"filled ({present}) — the home strip renders "
                f"{len(present)} cards, and FEATURED_TITLES in index.html "
                "is NOT used as filler once any row sets recent",
            ))
    return out


# --------------------------------------------------------------------------
# Drive cross-check (opt-in — needs OAuth, may open a browser)
# --------------------------------------------------------------------------

def check_drive(records: Sequence[tuple[int, str, dict[str, str]]]) -> list[Finding]:
    """Cross-check sheet image IDs against the portfolio Drive folder.

    OPT-IN ONLY. Imports the Google client libraries lazily so that the
    default CSV path runs on a bare `python3` with no venv. Read-only: lists
    files, never mutates.
    """
    pipeline_dir = Path(__file__).resolve().parent.parent / "tools" / "image_pipeline"
    sys.path.insert(0, str(pipeline_dir))

    push_token = pipeline_dir / "push-painting-token.json"
    oauth_client = pipeline_dir / "oauth-client.json"

    try:
        if push_token.exists():
            # Preferred path: reuse push_painting.py's cached token so no
            # browser opens. drive_auth.get_user_drive_service() hardcodes a
            # different token file (oauth-token.json), so build directly here.
            from google.oauth2.credentials import Credentials  # noqa: PLC0415
            from googleapiclient.discovery import build  # noqa: PLC0415
            creds = Credentials.from_authorized_user_file(
                str(push_token),
                ["https://www.googleapis.com/auth/drive"],
            )
            drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        else:
            # Fallback: the shared helper. NOTE: this may open a browser.
            import drive_auth  # noqa: PLC0415
            if not oauth_client.exists():
                return [Finding(
                    "ERROR", "drive-auth-missing", "", "",
                    f"neither {push_token.name} nor {oauth_client.name} "
                    f"found in {pipeline_dir} — cannot run --check-drive",
                )]
            drive = drive_auth.get_user_drive_service(oauth_client)
    except ImportError as exc:
        return [Finding(
            "ERROR", "drive-import-failed", "", "",
            f"Google API libraries unavailable ({exc}). Activate the venv: "
            f"source {pipeline_dir}/.venv/bin/activate",
        )]
    except Exception as exc:  # noqa: BLE001 - surface as a finding, not a crash
        return [Finding(
            "ERROR", "drive-auth-failed", "", "", f"Drive auth failed: {exc}",
        )]

    try:
        drive_files: dict[str, str] = {}
        page_token = None
        while True:
            resp = drive.files().list(
                q=f"'{DRIVE_FOLDER}' in parents and trashed = false",
                fields="nextPageToken, files(id,name)",
                pageSize=200,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                drive_files[f["id"]] = f.get("name", "")
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:  # noqa: BLE001
        return [Finding(
            "ERROR", "drive-list-failed", "", "",
            f"could not list Drive folder {DRIVE_FOLDER}: {exc}",
        )]

    out: list[Finding] = []
    referenced: set[str] = set()
    for sheet_row, label, d in records:
        image_id = d.get("image", "").strip()
        if not image_id:
            continue
        referenced.add(image_id)
        if image_id not in drive_files:
            out.append(Finding(
                "ERROR", "drive-dead-pointer", label, "image",
                f"image ID {image_id} is not in the portfolio Drive folder "
                "(deleted, trashed, or moved) — the card renders broken",
                sheet_row,
            ))

    for file_id, name in sorted(drive_files.items(), key=lambda kv: kv[1]):
        if file_id not in referenced:
            out.append(Finding(
                "WARN", "drive-orphan", name, "image",
                f"Drive file {name!r} ({file_id}) has no sheet row pointing "
                "at it — orphan left behind by a re-push or a deleted row",
            ))
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def sort_findings(findings: Sequence[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.row or 0, f.check),
    )


def render_text(findings: Sequence[Finding], source: str, row_count: int) -> str:
    lines: list[str] = []
    lines.append(f"Gallery audit — {row_count} painting rows from {source}")
    lines.append("")

    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for sev in ("ERROR", "WARN", "INFO"):
        group = [f for f in sort_findings(findings) if f.severity == sev]
        counts[sev] = len(group)
        if not group:
            continue
        lines.append(f"{sev} ({len(group)})")
        for f in group:
            where = f.title or "(sheet)"
            col = f".{f.column}" if f.column else ""
            rowref = f" [row {f.row}]" if f.row else ""
            lines.append(f"  {sev}  {where}{col}{rowref}: {f.message}")
        lines.append("")

    if not findings:
        lines.append("No findings — sheet is clean.")
        lines.append("")

    lines.append(
        f"Summary: {counts['ERROR']} error(s), {counts['WARN']} warning(s), "
        f"{counts['INFO']} info across {row_count} rows."
    )
    return "\n".join(lines)


def render_json(findings: Sequence[Finding], source: str, row_count: int) -> str:
    payload: dict[str, Any] = {
        "source": source,
        "rows": row_count,
        "counts": {
            sev: sum(1 for f in findings if f.severity == sev)
            for sev in ("ERROR", "WARN", "INFO")
        },
        "findings": [asdict(f) for f in sort_findings(findings)],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_gallery.py",
        description=(
            "Read-only pre-deploy audit of the Paintings sheet behind "
            "sveta-colours.com. Default mode fetches the published CSV and "
            "needs no auth and no venv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 = clean at the --fail-on threshold, "
            "1 = findings at or above it, 2 = could not load the data."
        ),
    )
    p.add_argument(
        "--check-drive", action="store_true",
        help="also cross-check Drive (dead image pointers + orphan files). "
             "Needs OAuth and may open a browser — never runs by default.",
    )
    p.add_argument(
        "--csv", type=Path, metavar="PATH",
        help="audit a local CSV file instead of fetching the published sheet.",
    )
    p.add_argument("--json", action="store_true", help="emit JSON findings.")
    p.add_argument(
        "--quiet", action="store_true",
        help="suppress the report; communicate only via the exit code.",
    )
    p.add_argument(
        "--fail-on", choices=["error", "warn", "info"], default="error",
        help="exit non-zero when a finding at or above this severity exists "
             "(default: error).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.csv:
        source = str(args.csv)
        text = read_local_csv(args.csv)
    else:
        source = "published sheet CSV"
        text = fetch_csv()

    header, rows = parse_csv(text)

    findings: list[Finding] = []
    findings.extend(check_header(header))
    findings.extend(check_rows(header, rows))

    if args.check_drive:
        records = [
            (i + 2, (row_to_dict(header, r).get("title") or f"<row {i + 2}>"),
             row_to_dict(header, r))
            for i, r in enumerate(rows)
        ]
        findings.extend(check_drive(records))

    if not args.quiet:
        if args.json:
            print(render_json(findings, source, len(rows)))
        else:
            print(render_text(findings, source, len(rows)))

    threshold = SEVERITY_ORDER[args.fail_on.upper()]
    tripped = any(SEVERITY_ORDER.get(f.severity, 9) <= threshold for f in findings)
    return 1 if tripped else 0


if __name__ == "__main__":
    sys.exit(main())
