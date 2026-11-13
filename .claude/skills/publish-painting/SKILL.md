---
name: publish-painting
description: Use when publishing or updating one painting on sveta-colours.com (or Olga Zai's olgazai.com) — triggered by "publish this", "publish", "publish to portfolio", "new painting, go", "update <painting title>", a re-push of an existing painting, or a raw painting photo dropped in with few or no words. Runs the Grok crop → trim → title/description → push_painting.py → verify workflow end to end.
---

# Publish Painting

The single most-repeated workflow in this repo: get one painting from a raw
phone photo onto the portfolio site (Google Drive image + a row in the
`Paintings` sheet). This file is canonical. If it disagrees with
`tools/image_pipeline/README.md` or with memory, fix this file.

**All commands run from `tools/image_pipeline/`.** Use the venv interpreter by
absolute path — `/Users/sveta/IdeaProjects/sveta-colours/tools/image_pipeline/.venv/bin/python` —
not `source .venv/bin/activate && python`. One command per Bash call, no `&&`
chaining (keeps auto mode free of permission popups).

---

## 0. Trigger — start, don't negotiate

"publish this" / "publish" / "new painting, go" / "publish to portfolio" /
"update <title>" / a bare painting photo — all mean *run this whole workflow*.
Sveta agreed to this trigger on 2026-06-08 after running it end to end on
*Holding On In Winter*. It exists so she doesn't retype the recipe.

**Do not:**

- Re-explain the five steps back to her before starting.
- Ask "would you like me to…" — she already said yes by using the phrase.
- Ask permission between steps. Stop only for real decisions: the title/
  description confirmation (step 3), a genuinely empty required field, a hard
  failure, or the pre-real-run confirmation in step 4.

**Do:** acknowledge the image, hand her the Grok prompt, and go.

### Which site — `--site` on every command

Two sites share these tools: `sveta` (sveta-colours.com) and `olga`
(olgazai.com, Olga Zai's watercolors and acrylics). Every command takes
`--site`; there is no default.

- While `tools.sheetId` in `sites/olga/site.json` is empty, Olga's site can't
  take paintings, so the target is `sveta` — no question.
- Once it is filled in, a message that doesn't say whose painting it is gets
  one short question before step 4: **"Sveta or Olga?"** ("my painting",
  "моя" → sveta; "Olga's", "Олина" → olga). Never guess from the image.
- The first line of every push prints the target (site, URL, sheet, Drive
  folder). Show it to her with the dry-run output, before the real run.

Read every prompt she sends in both English and Russian — see the
[`sveta-writing`](../sveta-writing/SKILL.md) skill, section 1. Russian
portions carry the truer meaning.

---

## 1. Grok perspective-correct + crop (manual, web UI)

She pastes this into her already-open Grok tab, along with the raw photo.
Give it verbatim:

```
Perspective-correct the uploaded photo and crop to the painted canvas only.
Remove all wall/background and all frame/border areas. Do not normalize, redraw,
or restyle the painting; preserve the original artwork exactly, including the
signature if it is on the painted surface. Compare result and original painting —
the original artwork must be preserved exactly, failure if it isn't.
```

Then she downloads Grok's output to `~/Downloads/`, or drops it into chat — in
which case save it yourself to `~/Downloads/<title>.jpeg`.

Why manual: Grok is the only tool that perspective-corrects + crops Sveta's
paintings acceptably. She has a **paid Grok account but no xAI API key**, so
this step is web-UI-only and cannot be automated. Don't try.

---

## 2. Trim Grok's white padding — *conditional, usually skip*

Grok **sometimes** pads its output to 1024×1024 with white bars on the short
axis. Most of the time it returns a clean rectangle. Check the dimensions
first; only run this when you see the padding.

```bash
/Users/sveta/IdeaProjects/sveta-colours/tools/image_pipeline/.venv/bin/python \
  trim_borders.py \
  ~/Downloads/<grok-output>.jpg \
  ~/Downloads/<title>-trimmed.jpeg
```

Positional args only: `input`, `output`. It reports the trim per edge, refuses
to crop to an empty image, and does pure pixel slicing — no resampling, just a
JPEG q95 re-save.

If borders survive (off-white, light grey), retune:

| flag | default | move it when |
| --- | --- | --- |
| `--min-bright N` | `240` | border is off-white/grey → **lower** it |
| `--max-std F` | `8.0` | border has faint texture/noise → **raise** it |

---

## 3. Title + description

**Voice is not specified here.** For Sveta's paintings every word of user-facing
copy is governed by the [`sveta-writing`](../sveta-writing/SKILL.md) skill —
read it and follow it. **For Olga's paintings sveta-writing does not apply**
(`tools.voiceSkill` is null): use Olga's own words when Sveta has them; otherwise
write plainly — what is in the painting and one specific thing, 30–55 words,
no invented facts, no art-statement or mood vocabulary — the same neutral rules
the photo manager uses (`aiRulesNeutral()` in `tools/photo_manager.html`).
Do not restate, summarize, or "simplify" its rules; it owns length, sentence
shapes, forbidden moves, and the three pre-publish tests.

Propose a title + description, discuss, and only proceed once she confirms.

**Existing-painting case (a re-push / "update <title>"):** pull the current row
*before* asking anything.

```bash
curl -sSL "https://docs.google.com/spreadsheets/d/e/2PACX-1vQrJdmLlZeUwFysVsa6rmHiEnby2moRMSYcBEixMNb3AoLxdpnvrSaWlxC23TF68VMTV3UDuQbuWOXV/pub?output=csv" -o /tmp/paintings.csv
```

Match on title, case-insensitively. Pre-fill `medium`, `price`, `size`,
`collection`, `orientation`, `hero`, `recent`, `sold` from that row and **ask
only about fields that are genuinely empty.** Never re-ask for a value the
sheet already holds.

`size` matters mechanically, not just cosmetically — see the aspect gate in
step 4.

---

## 4. Push to Drive + sheet

```bash
/Users/sveta/IdeaProjects/sveta-colours/tools/image_pipeline/.venv/bin/python \
  push_painting.py --site sveta \
  ~/Downloads/<title>.jpeg \
  --title "..." \
  --description "..." \
  --medium "acrylic" \
  --size "16x20 in" \
  --price "400" \
  --collection "Flowers,Seasons" \
  --orientation portrait \
  --dry-run
```

**Always `--dry-run` first**, show her the output, then re-run the identical
command with `--dry-run` removed.

Complete flag set (verified against `push_painting.py` argparse — do not invent
others):

| flag | notes |
| --- | --- |
| `--site sveta\|olga` | **required**, no default. Sheet, tab and Drive folder come from `sites/<site>/site.json` → `tools`; exits if they are empty |
| `<image>` | positional, required, must exist |
| `--title "..."` | **required** |
| `--description "..."` | omitted on an existing row → preserved |
| `--medium "..."` | |
| `--size "..."` | e.g. `16x20 in`; feeds the canvas-aspect gate below |
| `--price "..."` | |
| `--collection "A,B"` | comma-separated, no spaces. Known names per site: `tools.collections` in `sites/<site>/site.json` (Sveta: `Big City`, `Flowers`, `Forest`, `Seasons`) |
| `--orientation portrait\|landscape` | choices enforced; omit → preserved (existing) or auto-detected (new) |
| `--sold` | store_true → `sold=yes`. No flag: `no` for new, preserved for existing. Reverting to `no` is a manual sheet edit |
| `--hero` | store_true → `hero=TRUE`. No flag: `''` for new, preserved for existing |
| `--recent N` | string, positions `'1'`–`'6'`. `--recent ''` clears it |
| `--slug ...` | link key for a **new** painting; default `slugify(title)` (`Before the Thaw` → `before-the-thaw`). An existing slug is never changed (flag ignored, says so). A slug taken by another row exits **before upload**: pick another with `--slug` |
| `--dry-run` | store_true; prints, mutates nothing |

### The three behaviors that matter

1. **In-place Drive byte replacement.** It searches the portfolio folder for
   `<title>.jpeg`. Found → `files().update()` swaps the bytes and the **file ID
   stays stable**, so the sheet's `image` pointer and any cached site URL keep
   working. Not found → creates a new file.
2. **Preserve existing values on update.** For an existing row, any column not
   explicitly passed on the CLI keeps its current sheet value. This is why
   manual `hero` / `recent` / `orientation` edits survive a re-push — and why
   you must never pass a blank flag "just to be complete."
3. **Orphan cleanup to Trash.** If the sheet row pointed at a *different* image
   ID before the push, that old Drive file is moved to Trash (recoverable 30
   days). Real runs only.

### Canvas-aspect gate (will hard-exit)

When `--size` is given, `check_canvas_aspect()` compares the photo's aspect to
the canvas aspect (gallery cards lock to canvas aspect on `/works`):

- drift < 2% → uploads as-is.
- 2–5% per-side trim → writes a sibling `<name>.canvas.<ext>`, uploads *that*,
  prints an `aspect snap:` line.
- ≥ 5% per-side trim → **`sys.exit`** with a "re-crop tighter in Grok, or
  reshoot" message. Not a bug. Go back to step 1 with a tighter crop, or
  confirm the `size` string is right.

**Auth:** OAuth as Sveta (Drive + Sheets). First run opens a browser; the
refresh token is cached at `push-painting-token.json`, no browser afterwards.

---

## 5. Verify

- Confirm the printed `drive file id` and the `sheet: update row N` /
  `append row N` line.
- **Show her the actual resulting image** — Read/display it, don't just paste
  the log. She judges visual quality by looking.
- The site reads a **published CSV snapshot**, not the sheet directly. Google
  caches it: the site can lag the sheet by several minutes. Set that
  expectation and stop. Do not "debug" a missing row that is only cache lag —
  re-fetch the CSV a few minutes later if she wants proof.
- Site image URLs are derived from the Drive file ID as
  `https://lh3.googleusercontent.com/d/<fileId>`.

---

## Traps — do not repeat these

Each of these has cost real time.

- **Never propose `geometry.py` for this path.** It looks like the perspective-
  correction tool and it is not — for the per-painting track. It belongs to the
  separate batch `run_pipeline.py` track (Drive inbox → local `output/`).
  Sveta's words: *"of all my attempts only grok makes perspective correction +
  crop properly, no python script does."* The two tracks do not merge.
- **Never propose an xAI / Grok API integration.** There is no key. Suggesting
  one re-litigates a settled decision.
- **Never infer orientation from how the photo looks.** Raw photos include
  landscape context (wall, table) around portrait paintings. Guessing from
  thumbnails got **5 of 17 wrong** — Autumn Light, Downtown, Holding On In
  Winter, Morning Coffee, Summer Hut all read landscape and are all portrait.
  Get it from: the `orientation` column, the *cropped* image's dimensions, or
  by asking. Auto-detect in the CLI is safe only because it runs on the
  post-Grok crop.
- **Everything is `.jpeg`.** All portfolio files use that extension; the CLI
  normalizes `.jpg` → `.jpeg` so the by-name lookup finds prior uploads. Don't
  hand-write `.jpg` into the Drive name.
- **Only one painting is `hero` at a time.** `--hero` sets the new one; it does
  **not** unset the old one. Unsetting is a manual sheet edit — tell her.
- **`recent` is a position, not a boolean.** `1`–`6`, the order of the homepage
  recent carousel. `--recent 3` means slot 3. Setting a slot another painting
  already holds creates a duplicate; check the CSV first.
- **Show the image, not the log line.** After any push, display the actual
  result.
- **Don't reset `--dry-run` state carelessly.** After the real run, the next
  command you write should again start with `--dry-run`.

---

## Constants

Sveta's values below; every site's own are in `sites/<site>/site.json`
(`tools.*`, `site.sheetCsvUrl`), which the scripts read.

| | Value |
| --- | --- |
| Portfolio Drive folder | `1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX` |
| Paintings sheet | `170BuYN45Wk7H2g31gfRo35quok76JwrhFB8YV4-ckfI` |
| Sheet tab | `Paintings` |
| Columns (in order) | `title, medium, price, sold, size, image, description, collection, orientation, hero, recent, slug`. The script reads/writes the whole tab by the sheet's own header, so extra columns survive |
| Row match | case-insensitive on `title`, column A |
| File extension | `.jpeg` everywhere (`.jpg` normalized by the CLI) |
| `sold` | `yes` / `no` |
| `hero` | `TRUE` or empty — exactly one painting at a time |
| `recent` | `1`–`6` or empty — carousel position |
| `orientation` | `portrait` / `landscape`. Drives card sizing: landscape cards are 350px wide with transposed image dims (canvas rotated 90°), NOT a 2-column span — see the playbook's "Landscape cards have transposed dimensions" |
| `collection` | comma-separated, no spaces: `Seasons,Flowers` |
| Published CSV the site reads | `https://docs.google.com/spreadsheets/d/e/2PACX-1vQrJdmLlZeUwFysVsa6rmHiEnby2moRMSYcBEixMNb3AoLxdpnvrSaWlxC23TF68VMTV3UDuQbuWOXV/pub?output=csv` |
| Site image URL shape | `https://lh3.googleusercontent.com/d/<fileId>` |
| Token cache | `tools/image_pipeline/push-painting-token.json` (git-ignored) |
