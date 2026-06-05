# Photo Manager — Acceptance Evidence

Drop screenshots (and any supporting notes) here as the photo-manager rewrite acceptance
checklist gets run in a real browser. One file per checklist item is fine; pick names that
make the item obvious at a glance.

The checklist lives in [`../../../photo-manager-setup.md`](../../../photo-manager-setup.md)
under **Acceptance checklist**. The eight scenarios to cover:

1. 30-MP painting upload — Drive file is full resolution, not a thumbnail
2. `=IMPORTDATA(...)` Title saves as a literal string in Sheets (no formula execution)
3. Reload mid-flow — no stuck spinners, no orphan Drive file with no Sheets row
4. OAuth popup closed by user — status shows "OAuth cancelled," no infinite spinner
5. iPhone JPEG with EXIF orientation 6 — preview and saved Drive file are both upright
6. AI offline (no Gemini key, quota exhausted, network down) — manual-paste fallback works
7. Every form field reachable by keyboard from the file input through Save
8. Auto-correct clicked twice on the same upload — second click is a no-op with a hint

Suggested filenames: `01-full-res-upload.png`, `02-importdata-literal.png`, etc.

This directory is intentionally checked in (`.gitkeep`) so the path exists before Sveta
starts the acceptance run.
