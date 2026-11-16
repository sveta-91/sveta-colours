# New artist site: onboarding runbook

The order that worked for olgazai.com (2026-11-12 … 11-15), for the next artist.
Design and reasons: `docs/olgazai-plan.md`. Site folder format and the Cloudflare
details: `sites/README.md`.

**Who does it.** The agent does every step it can in Chrome (Claude in Chrome) or
with the tools below. The person does only: typing passwords and 2FA codes,
and anything in an account that isn't signed in to the extension's Chrome
profile. Before starting, ask the artist (or Sveta) to add the artist's Google
account to that profile (Google avatar → Add account); it then appears as a
`/u/1`, `/u/2` … slot in Google URLs. OAuth "Allow" screens: the agent clicks
them after a yes in chat. If a step does fall back to the person, give one step
per message and wait for a screenshot or "done".

| # | Step | Who | How |
| --- | --- | --- | --- |
| 1 | Site folder | agent | `sites/<name>/`: `site.json`, `theme.css`, `texts/` + `texts/ru/` (backup copies of home/about/faq). `python3 scripts/build_site.py <name>` warns about the empty links |
| 2 | Drive kit | agent | `python3 scripts/make_drive_kit.py --site <name> --out <dir> [--csv rows.csv] [--photos <dir>]` → `Paintings.csv`, `Paintings photos/`, `Site texts/*.txt` |
| 3 | Upload the kit to the artist's Drive | agent (artist's account in the profile) / artist | drag the kit folder into drive.google.com of the **artist's** account, so she owns it |
| 4 | Convert to Google files | agent / artist | right-click `Paintings.csv` → Open with → Google Sheets; each `.txt` → Open with → Google Docs. Uploaded .csv/.txt/.xlsx/.docx can't be published until converted. Rename the sheet tab to `Paintings` |
| 5 | Share with the publishing account | agent / artist | share the kit folder with sveta.colours as **Editor** (no notification needed). Check: `tools/image_pipeline/.venv/bin/python drive_tool.py ls <folder> --tree` shows `edit` on everything |
| 6 | Photos public | agent / artist | photo folder → Share → General access: Anyone with the link, Viewer. Check: `drive_tool.py public --folder <photo folder>` → all `ok` |
| 6b | Lock the sheet | agent | sheet → General access: **Restricted** (the published CSV keeps working). Only the photo folder is public, and only as **Viewer**, never Editor. First make sure sveta.colours is listed by name as Editor (`drive_tool.py info <id>`), or restricting locks the tools out. Olga's sheet and both photo folders were "Anyone with the link: Editor" until 2026-11-16 |
| 7 | Photo IDs into the sheet | agent | `drive_tool.py ids <photo folder> > ids.json`, then `drive_tool.py fill --site <name> ids.json` (dry run) and `--apply` after Sveta's yes |
| 8 | Publish the sheet | agent | File → Share → Publish to web → `Paintings` tab, CSV → copy the link into `site.sheetCsvUrl`; spreadsheet ID into `tools.sheetId`, folder into `tools.driveFolder` |
| 9 | Publish the six text Docs | agent | each Doc: File → Share → Publish to web → Publish. The confirm pop-up freezes the Chrome extension: run `window.confirm = () => true` in the page first. Links (`…/pub`) into `site.textDocs` / `textDocsRu`. Share the Docs with the artist as Editor if sveta.colours made them |
| 10 | Inquiry form | agent / artist | Apps Script in the **artist's** account (the form's mail goes from there): new project, paste `inquiry/Code.gs`, Deploy → Web app, execute as me, anyone → authorize (Allow screen: agent after a yes) → `/exec` URL into `site.formEndpoint`. Set `contactEmail`. Send one test inquiry |
| 11 | Build check | agent | `python3 scripts/build_site.py <name> --strict` passes; commit + push |
| 12 | Cloudflare Pages | agent (Sveta is a member of the artist's account) | `sites/README.md` → "olgazai.com on Cloudflare Pages": Workers & Pages → Create → **"Continue to Pages"** (the default screen is the Workers flow) → Connect to Git → build `python3 scripts/build_site.py <name> --strict`, output `dist` |
| 13 | Domain | agent | Pages project → Custom domains → apex + www. A read-only DNS record means another Cloudflare app (an old Worker) owns the name: remove the domain from that app first. Leave Email Routing MX/TXT records alone |
| 14 | Verify | agent | `tools/image_pipeline/.venv/bin/python scripts/verify_site.py --site <name> --smoke` (and `--url https://<project>.pages.dev/` before the domain is attached) → every page × language × 1440/390 `ok` |
| 15 | Tell the artist | Sveta | the letter: only the steps she must do herself (writing-style LEARNINGS) |

## Traps hit the first time

- The Chrome extension sees only its own Chrome profile. An artist's account in
  another profile can't be driven: add the account to the extension's profile instead.
- Google URLs pick the account by slot (`/u/0`, `/u/1`): check the slot before
  sharing or publishing, or the file lands in the wrong account.
- The Google Drive connector in Claude can read and search, but can't share
  files: share through Drive in Chrome, or `drive_tool.py` for what it covers.
- Local tests: Chrome blocks some Drive image responses to a page served from
  127.0.0.1 (`ERR_BLOCKED_BY_ORB`). Judge images on pages.dev or the live site.
- The published CSV lags a few minutes behind sheet edits, and Chrome may cache
  it; the site fetches it with `cache: 'no-cache'`.
