# Inquiry form backend (Google Apps Script)

The "Write to studio →" button in a painting's window opens a small form
(name, email, message). On Send, the site posts it to this Apps Script web app
(`Code.gs`), which:

1. appends a row to a **private** sheet "Sveta Colours — Inquiries" (created on
   the first run, in the Drive of the account that deploys the script), and
2. emails `sveta.colours@gmail.com` with **Reply-To = the visitor**, so
   answering is one click on Reply.

Until `FORM_ENDPOINT` in `index.html` is set, the site shows the mailto link
and the copyable address instead. The address stays visible under the form
either way.

## Contract (site ↔ script)

- `POST`, `Content-Type: text/plain;charset=utf-8`, body = JSON. Plain text
  keeps it a "simple" request: no CORS preflight, which Apps Script can't answer.
- Fields: `name`, `email`, `message` (required), `painting`, `intent`
  (`question` | `commission` | `buy` | `contact`), `page`, `elapsedMs`,
  `_gotcha` (honeypot), `_subject` (ignored).
- Reply: always HTTP 200 with `{"ok": true}` or `{"ok": false, "error":
  "invalid" | "busy" | "server"}`. The site checks `ok`, not the status code.
- Bots: a filled `_gotcha` or `elapsedMs < 2500` gets `{"ok": true}` and is
  dropped silently. More than 30 inquiries an hour → `busy`.

## Deploy (one time, ~10 minutes)

1. Sign in to Google with the account that should own the Inquiries sheet
   (the studio account, sveta.colours, keeps it next to the studio mail).
2. https://script.google.com → **New project**. Rename it `sveta-colours inquiry`.
3. Replace everything in the editor's `Code.gs` with this folder's `Code.gs`. Save.
4. In the function menu pick **setup** → **Run**. Google asks for permission
   (Sheets + send email as you). "Google hasn't verified this app" is expected
   for your own script: **Advanced → Go to sveta-colours inquiry (unsafe) → Allow**.
   The log prints the new Inquiries sheet's URL.
5. **Deploy → New deployment** → gear icon → **Web app**.
   - Execute as: **Me**
   - Who has access: **Anyone**
   **Deploy**, then copy the **Web app URL** (ends in `/exec`).
6. Put that URL into `formEndpoint` in `sites/<site>/site.json`; for Sveta run
   `python3 scripts/build_site.py sveta --root`. Commit, push.

Check: opening the `/exec` URL in a browser shows
`{"ok":true,"service":"sveta-colours inquiry"}`.

## Changing the script later

Edit in the Apps Script editor (and mirror the change into this folder), then
**Deploy → Manage deployments → ✏️ → Version: New version → Deploy**. This keeps
the same URL. "New deployment" would create a new URL and break the site.

## Status column

The sheet's last column, `status`, is empty on arrival: yours to fill
(`replied`, `sold`, …).
