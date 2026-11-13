#!/usr/bin/env python3
"""Push a processed painting to Drive + upsert its row in the Paintings sheet.

Usage:
  push_painting.py --site sveta|olga <image> --title "..." [--description "..."] [--medium "..."]
                          [--size "..."] [--price "..."] [--collection "..."]
                          [--orientation portrait|landscape]
                          [--hero] [--sold] [--recent N] [--slug ...]
                          [--crop-from top|center|bottom|left|right]
                          [--dry-run]

Mirrors the Drive folder + Paintings sheet conventions. Replaces an existing
Drive file with the same name in-place (file ID stable), upserts the row by
case-insensitive title.

Schema (11 columns, as of 2026-06-09, plus `slug` in column L since 2026-11-11):
  title, medium, price, sold, size, image, description, collection,
  orientation, hero, recent, slug

The whole tab is read and written by the sheet's own header, so extra columns
(slug, cover, order) are preserved. `slug` is the painting's permanent link key
(#painting/<slug>): a new row gets slugify(title) or --slug; an existing slug is
never changed. A slug already used by another row stops the push before upload.

For an existing row, columns not explicitly passed on the CLI are preserved
from the sheet — so manual edits to hero/recent/orientation survive a re-push.
For a new row, orientation auto-detects from the image; everything else
defaults to empty (except sold='no').

--site is required (no default): the target sheet, tab and Drive folder come
from sites/<site>/site.json ("tools"), and the script prints them before it
reads or writes anything. Olga's sheet and folder are shared with Sveta's
account as Editor, so one OAuth token serves both sites.

Auth: OAuth as the human user. First run opens a browser; refresh token cached
at push-painting-token.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "scripts"))
import site_config  # noqa: E402  (scripts/site_config.py)
OAUTH_CLIENT_PATH = SCRIPT_DIR / "oauth-client.json"
TOKEN_PATH = SCRIPT_DIR / "push-painting-token.json"

# Set from sites/<site>/site.json in main() — no site is the default.
DRIVE_FOLDER = ""
SHEET_ID = ""
SHEET_TAB = ""
COLS = [
    "title", "medium", "price", "sold", "size", "image",
    "description", "collection", "orientation", "hero", "recent",
]
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]

MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
}


def get_user_creds() -> Credentials:
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        refreshed = False
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                refreshed = True
            except RefreshError as exc:
                # An unpublished ("Testing") OAuth consent screen expires refresh
                # tokens after 7 days, so a cached token goes stale between
                # publishing sessions and Google answers 'invalid_grant'. That's
                # recoverable — drop the dead token and re-consent in the browser
                # instead of dying with a traceback.
                print(f"cached token rejected ({exc.args[0]}) — re-authenticating")
                TOKEN_PATH.unlink(missing_ok=True)
                creds = None
        if not refreshed:
            if not OAUTH_CLIENT_PATH.exists():
                sys.exit(f"OAuth client file not found: {OAUTH_CLIENT_PATH}")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(OAUTH_CLIENT_PATH), SCOPES,
            )
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(json.dumps({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or []),
        }))
    return creds


def get_services():
    creds = get_user_creds()
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets


def normalize_title(s: str) -> str:
    return (s or "").strip().lower()


def sanitize_cell(v) -> str:
    if v is None:
        return ""
    return str(v).replace("\r\n", "\n").strip()


def drive_filename(title: str, ext: str) -> str:
    return f"{title.strip() or 'untitled'}.{ext}"


def detect_orientation(image_path: Path) -> str:
    img = cv2.imread(str(image_path))
    if img is None:
        return "portrait"
    h, w = img.shape[:2]
    return "landscape" if w > h else "portrait"


def check_canvas_aspect(image_path: Path, size_str: str,
                        crop_from: str = "top") -> Path:
    """Enforce photo aspect ratio matches the painting's physical canvas.

    Why: gallery cards on /works lock to canvas aspect (so paintings of
    the same physical size hang at the same visible scale on the page).
    A photo whose aspect drifts from canvas creates either letterbox
    bars or, worse, off-grid card heights. Catch the drift at ingest.

    Returns the path to use for upload — either the original (within 2%
    tolerance) or a sibling `*.canvas.<ext>` (snap-cropped when drift is
    2-5% per-side). Exits with a reshoot message above 5%.

    `crop_from` picks the edge the snap-crop eats. Default "top": a height
    trim comes entirely off the top, because the signature lives on the
    bottom edge. Width trims are symmetric unless "left"/"right" is given.
    """
    if not size_str:
        return image_path
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", size_str)
    if not m:
        return image_path
    a, b = float(m.group(1)), float(m.group(2))
    if a <= 0 or b <= 0:
        return image_path

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path
    h_px, w_px = img.shape[:2]
    photo_aspect = w_px / h_px

    # Size strings like "11x14 in" don't carry orientation — a 16×20
    # canvas is either portrait (16w × 20h) or landscape (20w × 16h),
    # and the sheet's `size` field stores the numbers in whatever order
    # Sveta typed. Pick the canvas aspect that matches the photo's own
    # orientation, so landscape photos compare against landscape canvas.
    smaller, larger = min(a, b), max(a, b)
    canvas_aspect = (larger / smaller) if w_px > h_px else (smaller / larger)
    delta_pct = abs(photo_aspect - canvas_aspect) / canvas_aspect

    if delta_pct < 0.02:
        return image_path

    # Compute crop direction + per-side trim ratio
    if photo_aspect > canvas_aspect:
        new_w = round(h_px * canvas_aspect)
        per_side = (w_px - new_w) // 2
        trim_pct = per_side / w_px
    else:
        new_h = round(w_px / canvas_aspect)
        per_side = (h_px - new_h) // 2
        trim_pct = per_side / h_px

    if trim_pct >= 0.05:
        sys.exit(
            f"Photo aspect {photo_aspect:.3f} differs from canvas "
            f"{canvas_aspect:.3f} by {delta_pct*100:.1f}% — would need "
            f"{trim_pct*100:.1f}% trim per side, exceeds 5% safety "
            f"threshold (would likely eat into painted area). Re-crop "
            f"tighter in Grok, or reshoot."
        )

    # WHERE the pixels come off matters as much as how many.
    #
    # Height trims default to taking the whole bite off the TOP. Sveta signs
    # the bottom edge — "SM 2025" — and a symmetric crop puts half the bite
    # straight through it: on Waiting For Rain (2026-08-23) a 48px bottom trim
    # all but erased the signature, while the 95px it needed off the top was
    # uniform sky. Bottom edge = signature, top edge = usually sky. Bias up.
    #
    # Width trims stay symmetric: she signs bottom-LEFT on some paintings and
    # bottom-RIGHT on others, so neither side is reliably safe and centring is
    # the least-bad default. Override with --crop-from when it isn't.
    #
    # trim_pct keeps its symmetric-equivalent meaning above, so the 5%
    # accept/reject boundary is exactly what it always was — only the crop
    # position changed.
    if photo_aspect > canvas_aspect:
        removed = w_px - new_w
        if crop_from == "left":
            x0 = removed
        elif crop_from == "right":
            x0 = 0
        else:
            x0 = removed // 2
        cropped = img[:, x0:x0 + new_w]
        edge = (f"{removed}px off the left" if crop_from == "left"
                else f"{removed}px off the right" if crop_from == "right"
                else f"{removed}px split L/R")
    else:
        removed = h_px - new_h
        if crop_from == "bottom":
            y0 = 0
        elif crop_from == "center":
            y0 = removed // 2
        else:                      # "top" — the default, protects the signature
            y0 = removed
        cropped = img[y0:y0 + new_h, :]
        edge = (f"{removed}px off the bottom" if crop_from == "bottom"
                else f"{removed}px split T/B" if crop_from == "center"
                else f"{removed}px off the top")

    out_path = image_path.parent / f"{image_path.stem}.canvas{image_path.suffix}"
    if not cv2.imwrite(str(out_path), cropped):
        sys.exit(f"Failed to write snap-cropped image: {out_path}")
    print(
        f"aspect snap:    photo {photo_aspect:.3f} → canvas "
        f"{canvas_aspect:.3f} ({delta_pct*100:.1f}% off; {edge} "
        f"→ {out_path.name})"
    )
    print(
        "                check the cropped edge before publishing — "
        "verify the signature survived"
    )
    return out_path


def find_existing_drive_file(drive, name: str) -> str | None:
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    q = (
        f"name = '{escaped}' and "
        f"'{DRIVE_FOLDER}' in parents and trashed = false"
    )
    resp = drive.files().list(
        q=q, fields="files(id,name)", pageSize=10,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def upload_or_replace(drive, image: Path, name: str, dry_run: bool) -> str:
    existing = find_existing_drive_file(drive, name)
    if dry_run:
        return existing or "DRYRUN_NEW_ID"
    mime = MIME_BY_EXT.get(image.suffix.lower(), "application/octet-stream")
    media = MediaFileUpload(str(image), mimetype=mime, resumable=False)
    if existing:
        drive.files().update(
            fileId=existing, media_body=media, fields="id,name",
            supportsAllDrives=True,
        ).execute()
        return existing
    file = drive.files().create(
        body={"name": name, "parents": [DRIVE_FOLDER]},
        media_body=media, fields="id,name",
        supportsAllDrives=True,
    ).execute()
    return file["id"]


def col_letter(i: int) -> str:
    """0-based column index -> A1 letter(s): 0 -> A, 11 -> L, 26 -> AA."""
    s, i = "", i + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def slugify(name: str) -> str:
    """Same rule as index.html's slugify(), so site and sheet agree."""
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", name.lower()))


SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def resolve_slug(args, header: list, rows: list, found_idx: int,
                 existing: dict) -> str:
    """The slug to write, '' when the sheet has no `slug` column.

    Written once, never changed: an existing row keeps its slug even if the
    title or --slug changed. Exits (before any upload) on a bad or taken slug.
    """
    if "slug" not in header:
        return ""
    kept = existing.get("slug", "").strip()
    if kept:
        if args.slug and args.slug != kept:
            print(f"slug:           keeping {kept!r} (slugs never change; --slug ignored)")
        return kept
    slug = (args.slug or slugify(args.title)).strip()
    if not SLUG_RE.match(slug):
        sys.exit(f"slug {slug!r} must be lower-case letters, digits and single dashes. Pass --slug.")
    si = header.index("slug")
    for i in range(1, len(rows)):
        if i == found_idx:
            continue
        other = rows[i][si].strip() if si < len(rows[i]) else ""
        if other == slug:
            title = rows[i][0] if rows[i] else "?"
            sys.exit(f"slug {slug!r} is already used by {title!r} (row {i + 1}). "
                     f"Pass a different --slug.")
    return slug


def read_sheet(sheets):
    """Read the whole tab. Returns (header_list, all_rows_including_header)."""
    resp = sheets.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, range=SHEET_TAB,
    ).execute()
    rows = resp.get("values", [])
    header = rows[0] if rows else list(COLS)
    return header, rows


def find_match(rows: list, title: str):
    """Return (found_idx_0_based_in_rows, existing_dict_by_header_name).
    found_idx = -1 if no match. existing_dict empty if no match."""
    if len(rows) < 2:
        return -1, {}
    header = rows[0]
    nt = normalize_title(title)
    for i in range(1, len(rows)):
        cell = rows[i][0] if rows[i] else ""
        if normalize_title(cell) == nt:
            row = rows[i]
            d = {h: (row[j] if j < len(row) else "")
                 for j, h in enumerate(header)}
            return i, d
    return -1, {}


def build_row(args, header: list, file_id: str, existing: dict,
              orientation_auto: str, slug: str = "") -> list:
    """Build a row list aligned to the sheet's actual header.

    Rule per column:
      - explicit CLI arg → use it
      - else existing row had a value → preserve it
      - else sensible default
    """
    has_existing = bool(existing)

    def merge(cli_val, col_name, default=""):
        if cli_val is not None:
            return cli_val
        if has_existing:
            return existing.get(col_name, "")
        return default

    out = {}
    out["title"] = args.title
    out["medium"] = merge(args.medium, "medium")
    out["price"] = merge(args.price, "price")
    out["sold"] = "yes" if args.sold else merge(None, "sold", "no")
    out["size"] = merge(args.size, "size")
    out["image"] = file_id
    out["description"] = merge(args.description, "description")
    out["collection"] = merge(args.collection, "collection")
    out["orientation"] = merge(args.orientation, "orientation", orientation_auto)
    out["hero"] = "TRUE" if args.hero else merge(None, "hero", "")
    out["recent"] = merge(args.recent, "recent")
    if slug:
        out["slug"] = slug

    # Align to the actual sheet header (in case columns are reordered or extra
    # columns exist beyond COLS — those get preserved-or-empty per merge rule).
    row_values = []
    for col_name in header:
        if col_name in out:
            row_values.append(sanitize_cell(out[col_name]))
        else:
            # Unknown column — preserve existing if any, else empty
            row_values.append(sanitize_cell(merge(None, col_name, "")))
    return row_values


def write_row(sheets, row_values: list, header: list, found_idx: int,
              rows_count: int, dry_run: bool) -> str:
    last_col = col_letter(len(header) - 1)
    if found_idx >= 0:
        row_num = found_idx + 1
        target = f"{SHEET_TAB}!A{row_num}:{last_col}{row_num}"
        if not dry_run:
            sheets.spreadsheets().values().update(
                spreadsheetId=SHEET_ID, range=target,
                valueInputOption="RAW", body={"values": [row_values]},
            ).execute()
        return f"update row {row_num}"
    if not dry_run:
        sheets.spreadsheets().values().append(
            spreadsheetId=SHEET_ID, range=f"{SHEET_TAB}!A1",
            valueInputOption="RAW", body={"values": [row_values]},
        ).execute()
    return f"append row {rows_count + 1}"


def ensure_header(sheets, header: list, dry_run: bool) -> list:
    """If the sheet is empty, write the default header. Returns the header
    actually in use."""
    if header and header != []:
        return header
    if not dry_run:
        sheets.spreadsheets().values().update(
            spreadsheetId=SHEET_ID, range=f"{SHEET_TAB}!A1",
            valueInputOption="RAW", body={"values": [COLS]},
        ).execute()
    return list(COLS)


def trash_drive_file(drive, file_id: str) -> bool:
    """Move a Drive file to trash. 404 treated as success."""
    try:
        drive.files().update(
            fileId=file_id, body={"trashed": True}, fields="id,trashed",
            supportsAllDrives=True,
        ).execute()
        return True
    except HttpError as e:
        if e.resp.status == 404:
            return True
        raise


def main() -> int:
    p = argparse.ArgumentParser(description="Push painting to Drive + Paintings sheet.")
    site_config.add_site_arg(p)
    p.add_argument("image", type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--description", default=None,
                   help="If omitted on an existing row, preserves current value.")
    p.add_argument("--medium", default=None)
    p.add_argument("--size", default=None)
    p.add_argument("--price", default=None)
    p.add_argument("--collection", default=None,
                   help="Comma-separated list, e.g. 'Seasons,Flowers'.")
    p.add_argument("--orientation", choices=["portrait", "landscape"], default=None,
                   help="If omitted: preserved for existing row, auto-detected for new.")
    p.add_argument("--sold", action="store_true",
                   help="Set sold=yes. Without this flag: 'no' for new, preserved for existing.")
    p.add_argument("--hero", action="store_true",
                   help="Set hero=TRUE. Without this flag: '' for new, preserved for existing.")
    p.add_argument("--recent", default=None,
                   help="Set recent position '1'..'6'. Use empty string to clear.")
    p.add_argument("--slug", default=None,
                   help="Link key for a NEW painting (default: from the title). "
                        "Ignored when the row already has a slug: slugs never change.")
    p.add_argument("--crop-from",
                   choices=["top", "center", "bottom", "left", "right"],
                   default="top",
                   help="Which edge the aspect snap-crop takes pixels from. "
                        "Default 'top': height trims come off the top, "
                        "protecting the signature on the bottom edge. "
                        "Width trims are symmetric unless left/right given.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    global DRIVE_FOLDER, SHEET_ID, SHEET_TAB
    t = site_config.require(args.site, "sheetId", "sheetTab", "driveFolder")
    DRIVE_FOLDER, SHEET_ID, SHEET_TAB = t["driveFolder"], t["sheetId"], t["sheetTab"]
    print(site_config.describe(args.site))

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")
    if not args.title.strip():
        sys.exit("Title is required")

    # Match the folder's extension convention (all existing portfolio files
    # are `.jpeg`). Normalize jpg → jpeg so search-by-name finds prior uploads.
    ext = args.image.suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    name = drive_filename(args.title, ext)
    orientation_auto = detect_orientation(args.image)

    drive, sheets = get_services()
    print(f"image:          {args.image}")
    print(f"title:          {args.title}")
    print(f"drive filename: {name}")
    print(f"orientation:    auto={orientation_auto}"
          + (f" (explicit={args.orientation})" if args.orientation else ""))
    print(f"dry-run:        {args.dry_run}")

    try:
        # Read the sheet BEFORE uploading: on a re-push the canvas size lives
        # in the existing row, not on the command line. Checking against
        # `args.size` alone silently skipped the aspect gate for every update
        # — exactly the case it was added for (a replacement photo shot at a
        # different aspect than the canvas). Fall back to the row's `size`.
        header, rows = read_sheet(sheets)
        header = ensure_header(sheets, header, args.dry_run)
        found_idx, existing = find_match(rows, args.title)
        old_image_id = existing.get("image", "").strip()
        slug = resolve_slug(args, header, rows, found_idx, existing)
        if slug:
            print(f"slug:           {slug}")

        size_for_check = args.size or existing.get("size", "")
        if size_for_check and not args.size:
            print(f"canvas size:    {size_for_check} (from sheet row)")
        image_for_upload = check_canvas_aspect(args.image, size_for_check,
                                              args.crop_from)
        file_id = upload_or_replace(drive, image_for_upload, name, args.dry_run)
        print(f"drive file id:  {file_id}")

        row_values = build_row(args, header, file_id, existing, orientation_auto, slug)
        action = write_row(sheets, row_values, header, found_idx,
                           len(rows), args.dry_run)
        print(f"sheet:          {action}")

        if old_image_id and old_image_id != file_id and not args.dry_run:
            trash_drive_file(drive, old_image_id)
            print(f"orphan trashed: {old_image_id}")
    except HttpError as e:
        sys.exit(f"\nGoogle API error: {e}\n")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
