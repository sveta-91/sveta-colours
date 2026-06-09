#!/usr/bin/env python3
"""Push a processed painting to Drive + upsert its row in the Paintings sheet.

Usage:
  push_painting.py <image> --title "..." [--description "..."] [--medium "..."]
                          [--size "..."] [--price "..."] [--collection "..."]
                          [--orientation portrait|landscape]
                          [--hero] [--sold] [--recent N]
                          [--dry-run]

Mirrors the Drive folder + Paintings sheet conventions. Replaces an existing
Drive file with the same name in-place (file ID stable), upserts the row by
case-insensitive title.

Schema (11 columns, as of 2026-06-09):
  title, medium, price, sold, size, image, description, collection,
  orientation, hero, recent

For an existing row, columns not explicitly passed on the CLI are preserved
from the sheet — so manual edits to hero/recent/orientation survive a re-push.
For a new row, orientation auto-detects from the image; everything else
defaults to empty (except sold='no').

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
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCRIPT_DIR = Path(__file__).resolve().parent
OAUTH_CLIENT_PATH = SCRIPT_DIR / "oauth-client.json"
TOKEN_PATH = SCRIPT_DIR / "push-painting-token.json"

DRIVE_FOLDER = "1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX"
SHEET_ID = "170BuYN45Wk7H2g31gfRo35quok76JwrhFB8YV4-ckfI"
SHEET_TAB = "Paintings"
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
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
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


def check_canvas_aspect(image_path: Path, size_str: str) -> Path:
    """Enforce photo aspect ratio matches the painting's physical canvas.

    Why: gallery cards on /works lock to canvas aspect (so paintings of
    the same physical size hang at the same visible scale on the page).
    A photo whose aspect drifts from canvas creates either letterbox
    bars or, worse, off-grid card heights. Catch the drift at ingest.

    Returns the path to use for upload — either the original (within 2%
    tolerance) or a sibling `*.canvas.<ext>` (snap-cropped centered when
    drift is 2-5% per-side). Exits with a reshoot message above 5%.
    """
    if not size_str:
        return image_path
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", size_str)
    if not m:
        return image_path
    w_in, h_in = float(m.group(1)), float(m.group(2))
    if w_in <= 0 or h_in <= 0:
        return image_path
    canvas_aspect = w_in / h_in

    img = cv2.imread(str(image_path))
    if img is None:
        return image_path
    h_px, w_px = img.shape[:2]
    photo_aspect = w_px / h_px
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

    if photo_aspect > canvas_aspect:
        x0 = (w_px - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        y0 = (h_px - new_h) // 2
        cropped = img[y0:y0 + new_h, :]

    out_path = image_path.parent / f"{image_path.stem}.canvas{image_path.suffix}"
    if not cv2.imwrite(str(out_path), cropped):
        sys.exit(f"Failed to write snap-cropped image: {out_path}")
    print(
        f"aspect snap:    photo {photo_aspect:.3f} → canvas "
        f"{canvas_aspect:.3f} ({delta_pct*100:.1f}% off; cropped "
        f"{trim_pct*100:.1f}% per side → {out_path.name})"
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


def read_sheet(sheets):
    """Read all rows. Returns (header_list, all_rows_including_header)."""
    last_col = chr(64 + len(COLS))
    resp = sheets.spreadsheets().values().get(
        spreadsheetId=SHEET_ID, range=f"{SHEET_TAB}!A:{last_col}",
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
              orientation_auto: str) -> list:
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
    last_col = chr(64 + len(header))
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
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

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
        image_for_upload = check_canvas_aspect(args.image, args.size or "")
        file_id = upload_or_replace(drive, image_for_upload, name, args.dry_run)
        print(f"drive file id:  {file_id}")

        header, rows = read_sheet(sheets)
        header = ensure_header(sheets, header, args.dry_run)
        found_idx, existing = find_match(rows, args.title)
        old_image_id = existing.get("image", "").strip()

        row_values = build_row(args, header, file_id, existing, orientation_auto)
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
