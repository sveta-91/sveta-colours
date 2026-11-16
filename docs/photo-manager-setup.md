# Photo Manager — Setup

`admin/index.html` publishes one prepared painting image: watermark,
AI title/description dialog, Save to Drive + the `Paintings` sheet, live check.

## One-time Google setup (OAuth for Drive + Sheets)

1. https://console.cloud.google.com → project **sveta-colours**, which issued the
   OAuth client `754552250731-hdl67ujna2lk1lataah35mnl69emi8qt`.
2. **OAuth consent screen** (Google Auth Platform → Audience) → External, Testing.
   **Test user: the sveta.mordkovich Gmail account**, the one that owns the
   `Paintings` sheet and the portfolio Drive folder. Sign in with that account
   (it is not the old work address that used to be listed here; any other
   account gets "access blocked"). For `olgazai.com/admin/`, also add the
   Google account that owns Olga's sheet and Drive folder. Scopes:
   `https://www.googleapis.com/auth/drive.file`, `https://www.googleapis.com/auth/spreadsheets`.
3. **Credentials** → the OAuth client → Authorized JavaScript origins:
   `http://localhost:8000`, `http://127.0.0.1:8000`, and for the live admin
   pages `https://www.sveta-colours.com`, `https://sveta-colours.com`,
   `https://olgazai.com`, `https://www.olgazai.com`. Without its origin a live
   page's sign-in fails with `origin_mismatch`.
4. **APIs & Services** → enable **Google Drive API** and **Google Sheets API**.

`drive.file` only lets the page delete files it created itself. When it replaces
an image first uploaded by `push_painting.py`, the old Drive file stays (harmless
orphan; delete it by hand if you care).

## OpenAI key (the AI dialog)

Create a key at https://platform.openai.com/api-keys and set a monthly budget
there. On the page: **OpenAI settings** → paste → Save. The key is never in the
repo. On localhost it stays in this browser's localStorage. On the live admin
page it is kept in sessionStorage and is gone when the tab closes, because any
script on the public site's origin could read localStorage. The model is set
in the same place (default `gpt-5.4-mini`).

## Running the page

```bash
./scripts/dev.sh
```

Open http://localhost:8000/admin/?site=sveta (or `?site=olga`; without it the page asks) (not `file://`: the page
reads the voice rules from `.claude/skills/sveta-writing/` through the server).

## Acceptance checklist

- [ ] Prepared JPEG, watermark off → the Drive file is byte-identical to the original
- [ ] Watermark on → Drive file is a JPEG with the text bottom-right
- [ ] Existing title → form fills from its row, status says Save will update it; Save rewrites that row in place, keeps extra columns
- [ ] New title → row appended; **Live on the site** link appears once the published CSV catches up
- [ ] Hero ticked → the previous hero row is cleared
- [ ] Recent position already used → warning names the other painting
- [ ] Sheet has a `slug` column → new rows get one; an existing slug is read-only; a clash blocks Save before upload
- [ ] Photo proportions ≥5% off the canvas size → red warning (Save still allowed)
- [ ] AI: Start sends image + fields + notes; replies keep context; **Use this** fills title, description, slug
- [ ] OAuth popup closed → status shows the error, no stuck spinner
- [ ] Title `=IMPORTDATA("https://evil.example.com")` saves as literal text (RAW)
