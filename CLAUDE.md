# Sveta Colours

Live single-artist portfolio at **www.sveta-colours.com** — a single-file `index.html`
(all CSS/JS inlined) served from GitHub Pages off `main` (custom domain via `CNAME`,
DNS fronted by Cloudflare). Gallery data comes from a Google Sheet published as CSV;
painting images live in one public Google Drive folder. No server, no build step.
The repo also holds local Python tooling for getting a painting photo onto the site.

Most accurate document in the repo: **`docs/portfolio-playbook.md`**. Read it before
any design- or layout-bearing change. Don't re-derive what it already answers.

---

## Two tracks — one is active

### 1. Per-painting publishing — ACTIVE, this is the default

Raw photo → **Grok web UI** (perspective correct + crop) → `trim_borders.py`
*(only if Grok padded to 1024×1024)* → discuss title + description in chat →
`push_painting.py` (upload to Drive + upsert the sheet row) → verify.

- **Canonical spec: `.claude/skills/publish-painting/SKILL.md`.** It owns the flags,
  the constants, and the traps. If it disagrees with anything else, it wins — fix
  the other file.
- Runbook prose + file inventory: `tools/image_pipeline/README.md`.
- Grok is web-UI only (paid account, **no xAI API key**). Never propose `geometry.py`
  for this path, and never propose a Grok/xAI API integration — both are settled.
- Trigger phrase: **"publish this"** (also "publish", "new painting, go",
  "update <title>", or a bare painting photo) means run the whole workflow. Start;
  don't re-explain it back.

### 2. Batch inbox pipeline — PAUSED, reference only

`run_pipeline.py`: Drive inbox → straighten → crop → local `output/processed/{portrait,landscape}/`.
Service-account auth. Not the publishing path; don't merge the two.

Real status: straightening, orientation detection, crop *detection* (3 signals, 19/20)
and crop *implementation* (3 strategies + `_trim_frame_border()`) are all **implemented**
— 18/20 clean on the 2026-04 test set. The open problem is the **golden wooden frame**,
where the outer-frame trim leaves a thin remnant on 6 images. Approach it as finding the
*inner* painting rectangle, not trimming the outer frame. Thresholds, the 20-image
reference set, and every failed approach are in the `project_cropping_experience` memory —
read it before touching this code. Drive upload from this track is still unimplemented.

---

## Site architecture

`index.html` fetches the published-CSV snapshot of the `Paintings` sheet, parses it
client-side with a case-insensitive `get('column')` helper, and renders. Every field
degrades gracefully; JS constants are the safety net (`HERO_PIN_TITLE`,
`FEATURED_TITLES`) so an emptied column never blanks the home page.

**Sheet schema — 11 columns, in order:**
`title, medium, price, sold, size, image, description, collection, orientation, hero, recent`

| Column | Semantics |
| --- | --- |
| `image` | Drive file ID only. Rendered as `https://lh3.googleusercontent.com/d/<id>` |
| `sold` | `yes`/`no`. Shown, never hidden — the portfolio is a record. Price → "Sold" |
| `orientation` | `portrait`/`landscape`. Drives card sizing (transposed dims, flex layout) |
| `hero` | `TRUE` or empty. **Exactly one at a time**; `--hero` does not unset the old one |
| `recent` | `1`–`6`, a *position* in the home carousel, not a boolean |
| `collection` | Comma-separated, no spaces. Values: `Big City`, `Flowers`, `Forest`, `Seasons` |

Gallery cards lock rendered aspect to the **physical canvas** (`size`), never to the
photo's pixel aspect; `push_painting.py`'s `check_canvas_aspect()` enforces the match at
publish time (<2% pass, 2–5% snap-crop, ≥5% hard exit). Layout patterns, the hero-sizing
trap, masonry timing, cursor scoping, and the CSV cache lag: all in the playbook.

---

## Skills in this repo

| Skill | Fires when |
| --- | --- |
| `sveta-writing` (`.claude/skills/sveta-writing/SKILL.md`) | Any site-visible text (descriptions, About, captions, buttons). Also carries the **always-on bilingual prompt-reading rule** |
| `publish-painting` (`.claude/skills/publish-painting/SKILL.md`) | The publishing workflow above |
| `brainstorm-panel` | Design-bearing batches with >3 simultaneous changes. Confirmed patterns: `.claude/brainstorm-panel/roster.md`. Per-panel narrative: `.claude/brainstorm-panel/log.md` |
| `watermark-removal` (global) | Removing logos/text/overlays from artwork photos. Playbook: `docs/watermark-removal-playbook.md` |

---

## Tooling

| Path | Purpose |
| --- | --- |
| `tools/image_pipeline/` | All Python tooling. Venv at `.venv/`; run commands from this dir |
| `tools/photo_manager.html` | Browser alternative to `push_painting.py`. Serve via `scripts/dev.sh` |
| `scripts/audit_gallery.py` | Read-only pre-deploy data audit of the gallery. CSV-only by default; `--check-drive` opts into Drive checks |
| `scripts/verify_site.py` | Parameterized Selenium UI verification against the deployed site |
| `scripts/templates/selenium_verify.py` | Generic verify scaffold — copy to `/tmp/<intent>_verify.py` and edit the assertions |

---

## Working rules

### Never

- Commit `config.json`, `service-account-key.json`, `oauth-client.json`, any `*-token.json`, or credentials.
- Run `dry_run: false` (or drop `--dry-run`) without confirming with Sveta first.
- Guess a painting's orientation from a photo thumbnail. Photos include landscape
  context around portrait paintings — guessing got **5 of 17 wrong**. Read the
  `orientation` column, or the *cropped* image's dimensions, or ask.
- Straighten or re-encode images that don't need it — pass through clean JPEGs.
- Push visual QA onto Sveta as the default. Verify it yourself first.

### Always

- **Read every prompt in both English and Russian.** Sveta mixes languages mid-thought;
  when the halves diverge, **prefer the Russian** — it's the honest version. She switches
  to Russian when she means it most. Cyrillic-keyboard-typed English
  (*"аре тчере нехт степс"*) parses as English. Reply in English unless her message is
  entirely Russian.
- **"go" means execute immediately** — no confirmation, no plan, no discussion.
- **"auto mode" means zero approvals AND zero harness permission popups.** A popup is a
  bug, not a speed bump: pre-wire `.claude/settings.local.json` before dispatching, use
  venv binaries by absolute path, one command per Bash call (no `&&`), and avoid inline
  `python -c` (it's gated separately).
- **Preserve flavor over minimalism.** Confirmed across three brainstorm panels: the
  gallery grid kept meta + price, About kept "learning to really see", descriptions
  promoted "the small thing". When in doubt keep the richer option and lean on typography.
- **Show the actual output images**, not just log lines. She judges by looking.
- **Verify UI changes with Selenium before declaring done** — Selenium 4 is in
  `tools/image_pipeline/.venv`, Chrome is in `/Applications`. Screenshots + computed-CSS
  diffs; hand her evidence, not a "go look".
- Reset `dry_run: true` in `config.json` after any real batch run.
- Activate the venv, or call `.venv/bin/python` by absolute path, from `tools/image_pipeline/`.
- **Commit and push as one action.** Stage specific files (never `git add -A`); the
  message says what was implemented, the approach, and what was tested.
- **Keep the README layout map in sync** whenever a file or folder is added, removed,
  renamed, or moved.
- Work incrementally on pipeline code — one stage, tested, before the next.
