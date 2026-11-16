#!/usr/bin/env python3
"""Small Drive / Sheets helper on the publishing account (push_painting.py's OAuth).

Replaces the one-off scripts written while setting up olgazai.com: list a
shared folder and what the account may edit, upload photos, map photo names
to Drive IDs, fill a sheet column for rows whose cell is empty, and check
that images load for a signed-out visitor.

Run from tools/image_pipeline with the venv:

  .venv/bin/python drive_tool.py ls FOLDER_ID [--tree]
  .venv/bin/python drive_tool.py info FILE_ID
  .venv/bin/python drive_tool.py upload FOLDER_ID photo1.jpg photo2.jpg [--apply]
  .venv/bin/python drive_tool.py ids FOLDER_ID > ids.json        # {"<name without ext>": "<id>"}
  .venv/bin/python drive_tool.py fill --site olga --column image ids.json [--apply]
  .venv/bin/python drive_tool.py public FILE_ID ... | --folder FOLDER_ID

Writes (upload, fill) are dry runs unless --apply is given. upload skips a file
whose name is already in the folder; fill writes only empty cells, matching
rows by --key (default: title), and never overwrites a filled cell.
A Drive folder or sheet owned by someone else works once it is shared with
the publishing account as Editor.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

GOOGLE_PREFIX = "application/vnd.google-apps."
OFFICE_PREFIX = "application/vnd.openxmlformats-officedocument."


def services():
    import push_painting as pp  # noqa: E402  (same folder; OAuth token cache)
    return pp.get_services()


def short_mime(m: str) -> str:
    return m.replace(GOOGLE_PREFIX, "g:").replace(OFFICE_PREFIX, "ox:")


def list_folder(drive, folder: str) -> list[dict]:
    out, tok = [], None
    while True:
        r = drive.files().list(
            q=f"'{folder}' in parents and trashed=false",
            fields="nextPageToken,files(id,name,mimeType,capabilities(canEdit))",
            pageSize=200, pageToken=tok,
            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        out += r["files"]
        tok = r.get("nextPageToken")
        if not tok:
            return sorted(out, key=lambda f: f["name"].lower())


def cmd_ls(a) -> int:
    drive, _ = services()
    root = drive.files().get(fileId=a.folder, fields="name,capabilities(canEdit)",
                             supportsAllDrives=True).execute()
    print(f"{root['name']}/  {'edit' if root['capabilities'].get('canEdit') else 'VIEW ONLY'}")

    def walk(fid: str, ind: str) -> None:
        for f in list_folder(drive, fid):
            edit = "edit" if f["capabilities"].get("canEdit") else "VIEW ONLY"
            if f["mimeType"] == GOOGLE_PREFIX + "folder":
                print(f"{ind}{f['name']}/  {f['id']}  {edit}")
                if a.tree:
                    walk(f["id"], ind + "  ")
            else:
                print(f"{ind}{f['name']}  {short_mime(f['mimeType'])}  {f['id']}  {edit}")

    walk(a.folder, "  ")
    return 0


def cmd_info(a) -> int:
    drive, _ = services()
    m = drive.files().get(
        fileId=a.file, supportsAllDrives=True,
        fields="id,name,mimeType,owners(emailAddress),parents,webViewLink,"
               "capabilities(canEdit,canShare)").execute()
    try:
        perms = drive.permissions().list(fileId=a.file, supportsAllDrives=True,
                                         fields="permissions(type,role)").execute()
        anyone = [p["role"] for p in perms["permissions"] if p["type"] == "anyone"]
        m["anyoneWithLink"] = anyone[0] if anyone else "no"     # reader / commenter / writer
        if "writer" in anyone:
            print("!! anyone with the link can EDIT this file", file=sys.stderr)
    except Exception:  # noqa: BLE001 — listing permissions needs share rights
        m["anyoneWithLink"] = "unknown (no share rights)"
    print(json.dumps(m, indent=2))
    return 0


def cmd_upload(a) -> int:
    from googleapiclient.http import MediaFileUpload
    import push_painting as pp

    drive, _ = services()
    have = {f["name"] for f in list_folder(drive, a.folder)}
    for p in map(Path, a.files):
        if not p.is_file():
            print(f"missing   {p}")
            continue
        if p.name in have:
            print(f"skip      {p.name} (already in the folder)")
            continue
        if not a.apply:
            print(f"would upload {p.name}")
            continue
        mime = pp.MIME_BY_EXT.get(p.suffix.lower(), "application/octet-stream")
        f = drive.files().create(body={"name": p.name, "parents": [a.folder]},
                                 media_body=MediaFileUpload(str(p), mimetype=mime),
                                 fields="id", supportsAllDrives=True).execute()
        print(f"uploaded  {p.name}  {f['id']}")
    if not a.apply:
        print("dry run: add --apply to upload")
    return 0


def cmd_ids(a) -> int:
    drive, _ = services()
    ids = {Path(f["name"]).stem: f["id"] for f in list_folder(drive, a.folder)
           if not f["mimeType"].startswith(GOOGLE_PREFIX + "folder")}
    json.dump(ids, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def cmd_fill(a) -> int:
    if a.site:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
        import site_config
        t = site_config.require(a.site, "sheetId")
        sheet_id, tab = t["sheetId"], a.tab or t.get("sheetTab") or "Paintings"
    elif a.sheet:
        sheet_id, tab = a.sheet, a.tab or "Paintings"
    else:
        sys.exit("fill: give --site or --sheet")
    values = json.loads(Path(a.mapping).read_text(encoding="utf-8"))
    _, sheets = services()
    rows = sheets.spreadsheets().values().get(spreadsheetId=sheet_id, range=tab).execute()["values"]
    head = [h.strip().lower() for h in rows[0]]
    for need in (a.key, a.column):
        if need.lower() not in head:
            sys.exit(f"fill: no '{need}' column in {tab} (have: {', '.join(head)})")
    ki, ci = head.index(a.key.lower()), head.index(a.column.lower())
    col, data, filled, unknown = col_letter(ci), [], 0, []
    for n, r in enumerate(rows[1:], 2):
        key = r[ki].strip() if len(r) > ki else ""
        cur = r[ci].strip() if len(r) > ci else ""
        if not key:
            continue
        if cur:
            filled += 1
            continue
        if key in values:
            data.append({"range": f"{tab}!{col}{n}", "values": [[values[key]]]})
            print(f"{'write' if a.apply else 'would write'} {col}{n}  {key} = {values[key]}")
        else:
            unknown.append(key)
    print(f"{len(data)} empty cells to fill, {filled} already filled (left alone)"
          + (f", no value for: {', '.join(unknown)}" if unknown else ""))
    if a.apply and data:
        sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=sheet_id,
            body={"valueInputOption": "RAW", "data": data}).execute()
        print("written")
    elif not a.apply:
        print("dry run: add --apply to write")
    return 0


def cmd_public(a) -> int:
    ids = list(a.ids)
    if a.folder:
        drive, _ = services()
        ids += [f["id"] for f in list_folder(drive, a.folder)
                if not f["mimeType"].startswith(GOOGLE_PREFIX)]
    bad = 0
    for i in ids:
        url = f"https://lh3.googleusercontent.com/d/{i}=w200"
        try:   # no cookies, no token: what a signed-out visitor gets
            with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as r:
                ok = r.status == 200 and r.headers.get("Content-Type", "").startswith("image/")
                print(f"{'ok  ' if ok else 'FAIL'} {i}  {r.status} {r.headers.get('Content-Type')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL {i}  {e}")
        bad += not ok
    print(f"{len(ids) - bad}/{len(ids)} load signed out"
          + ("; share the folder as 'Anyone with the link: Viewer'" if bad else ""))
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ls", help="list a folder: name, type, id, edit / VIEW ONLY")
    s.add_argument("folder"); s.add_argument("--tree", action="store_true", help="recurse into subfolders")
    s = sub.add_parser("info", help="one file: owner, parents, edit/share rights, link sharing")
    s.add_argument("file")
    s = sub.add_parser("upload", help="upload files into a folder, skipping names already there")
    s.add_argument("folder"); s.add_argument("files", nargs="+"); s.add_argument("--apply", action="store_true")
    s = sub.add_parser("ids", help="print {name without extension: id} for a folder, as JSON")
    s.add_argument("folder")
    s = sub.add_parser("fill", help="fill empty cells of one sheet column from a JSON {key: value} map")
    s.add_argument("mapping", help="JSON file, e.g. the output of `ids`")
    s.add_argument("--site", help="sites/<name>/site.json gives the sheet (tools.sheetId)")
    s.add_argument("--sheet", help="spreadsheet ID, instead of --site")
    s.add_argument("--tab", help="sheet tab (default: the site's sheetTab, or Paintings)")
    s.add_argument("--column", default="image", help="column to fill (default image)")
    s.add_argument("--key", default="title", help="column matched against the map's keys (default title)")
    s.add_argument("--apply", action="store_true")
    s = sub.add_parser("public", help="check images load for a signed-out visitor (lh3 URL)")
    s.add_argument("ids", nargs="*"); s.add_argument("--folder", help="check every file in this folder")
    a = p.parse_args()
    return {"ls": cmd_ls, "info": cmd_info, "upload": cmd_upload, "ids": cmd_ids,
            "fill": cmd_fill, "public": cmd_public}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
