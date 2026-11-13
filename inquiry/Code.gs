/**
 * Sveta Colours — inquiry form backend (Google Apps Script web app).
 *
 * The site's form POSTs JSON here (Content-Type text/plain, so the browser
 * sends no CORS preflight). Each inquiry is:
 *   1. appended as a row to a PRIVATE spreadsheet "Sveta Colours — Inquiries"
 *      (created on the first run; never the published Paintings sheet,
 *      because a published sheet exposes every tab), and
 *   2. emailed to STUDIO_EMAIL with Reply-To set to the buyer, so answering
 *      is one click on Reply.
 *
 * Deploy as the studio Google account: see inquiry/README.md.
 * Another artist's site (olgazai.com) deploys its own copy under its own
 * account: change only the four values below.
 */

const STUDIO_EMAIL = 'sveta.colours@gmail.com';   // where inquiries arrive
const SITE_DOMAIN = 'sveta-colours.com';           // shown in the email
const SERVICE = 'sveta-colours inquiry';           // what the /exec URL answers
const INBOX_TITLE = 'Sveta Colours — Inquiries';   // the private spreadsheet it creates
const SHEET_NAME = 'Inquiries';
const MAX_PER_HOUR = 30;       // all visitors together; MailApp's own cap is 100/day
const MAX_FIELD = 4000;

function doPost(e) {
  try {
    const d = JSON.parse((e && e.postData && e.postData.contents) || '{}');

    // Bots: the hidden field is filled, or the form was sent faster than a
    // person can type. Answer "ok" so they don't retry; store nothing.
    if (d._gotcha) return json_({ ok: true });
    if (Number(d.elapsedMs) > 0 && Number(d.elapsedMs) < 2500) return json_({ ok: true });

    const name = clean_(d.name, 200);
    const email = clean_(d.email, 320);
    const message = clean_(d.message, MAX_FIELD);
    if (!name || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) || !message) {
      return json_({ ok: false, error: 'invalid' });
    }
    if (!underRateLimit_()) return json_({ ok: false, error: 'busy' });

    const row = {
      when: new Date(),
      intent: clean_(d.intent, 40),
      painting: clean_(d.painting, 200),
      name: name,
      email: email,
      shipTo: clean_(d.shipTo, 200),
      message: message,
      page: clean_(d.page, 300),
    };
    sheet_().appendRow([row.when, row.intent, row.painting, row.name, row.email, row.shipTo, row.message, row.page, '']);

    const label = { buy: 'Wants to buy', question: 'Question', commission: 'Commission', contact: 'Message' }[row.intent] || 'Inquiry';
    const subject = row.painting ? `${label}: "${row.painting}" (${row.name})` : `${label} from ${row.name}`;
    const body = [
      `${label}${row.painting ? ' about "' + row.painting + '"' : ''}`,
      '',
      `From: ${row.name} <${row.email}>`,
      row.shipTo ? `Ship to: ${row.shipTo}` : '',
      '',
      row.message,
      '',
      '—',
      `Sent from ${row.page || SITE_DOMAIN}. Reply to this email to answer ${row.name} directly.`,
    ].filter((l, i, a) => l !== '' || a[i - 1] !== '').join('\n');
    MailApp.sendEmail({ to: STUDIO_EMAIL, replyTo: row.email, name: SITE_DOMAIN, subject: subject, body: body });

    return json_({ ok: true });
  } catch (err) {
    console.error(err);
    return json_({ ok: false, error: 'server' });
  }
}

// GET: a health check you can open in the browser after deploying.
function doGet() {
  return json_({ ok: true, service: SERVICE });
}

function sheet_() {
  const props = PropertiesService.getScriptProperties();
  let id = props.getProperty('SHEET_ID');
  let ss;
  if (id) {
    try { ss = SpreadsheetApp.openById(id); } catch (e) { ss = null; }
  }
  if (!ss) {
    ss = SpreadsheetApp.create(INBOX_TITLE);
    props.setProperty('SHEET_ID', ss.getId());
  }
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) {
    sh = ss.getSheets()[0];
    sh.setName(SHEET_NAME);
    sh.appendRow(['when', 'intent', 'painting', 'name', 'email', 'ship to', 'message', 'page', 'status']);
    sh.setFrozenRows(1);
  }
  return sh;
}

function underRateLimit_() {
  const lock = LockService.getScriptLock();
  lock.waitLock(5000);
  try {
    const cache = CacheService.getScriptCache();
    const key = 'n' + Utilities.formatDate(new Date(), 'UTC', 'yyyyMMddHH');
    const n = Number(cache.get(key) || 0);
    if (n >= MAX_PER_HOUR) return false;
    cache.put(key, String(n + 1), 3700);
    return true;
  } finally {
    lock.releaseLock();
  }
}

function clean_(v, max) {
  return String(v == null ? '' : v).replace(/\u0000/g, '').trim().slice(0, max);
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

// Run once from the editor (▶ setup) to grant permissions and create the sheet.
function setup() {
  const sh = sheet_();
  console.log('Inquiries sheet: ' + sh.getParent().getUrl());
}
