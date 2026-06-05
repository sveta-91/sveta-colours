# Photo Manager Audit — Decision Summary

**Date:** 2026-06-05
**Subject:** `tools/photo_manager.html` (560 lines)
**Process:** 5 agents × Round 1 parallel audit + Round 2 critique pass
**Reports:** `audit-r1-*.md`, `audit-r2-*.md` in this folder

---

## Verdict tally

| Agent | Round 1 | Round 2 | Movement |
|---|---|---|---|
| AI integration manager | improve | **rewrite** | flipped |
| Google integration expert | improve | improve | held |
| Frontend developer | rewrite | rewrite (narrower) | held |
| Image processing expert | rewrite | rewrite (narrower) | held |
| QA tester | rewrite | rewrite (re-ranked) | held |

**Consensus: rewrite, 4–1.** Google is the lone dissenter; they argue their own lane is fixable in place, but accept the rest of the file isn't.

---

## Blockers everyone agreed on

After Round 2, four findings show up in every report — they're the load-bearing reasons not to ship the current file as-is.

1. **Drive uploads are silently downscaled to ~460 px.** The Drive save (line 517) and download (line 479) read from the display canvas, which `setupCanvas` caps at 460 px tall (line 250). Sveta uploads a 30 MP painting and gets a thumbnail in her Drive folder, while the Sheets row reports the original dimensions. *Data destruction on the happy path.* (Source: Image Processing R1 #1; confirmed by Frontend, QA, Google in R2.)

2. **The page renders nearly unstyled at `file://`.** The whole stylesheet relies on `--color-text-primary`, `--font-sans`, `--border-radius-md`, etc. — CSS custom properties that are never defined in the file. They look like they were inherited from a Claude.ai chat-artifact host. Opened locally without a token-injecting host, the page is essentially raw HTML. *The page has likely never actually been rendered.* (Source: Frontend R1 #1.)

3. **`USER_ENTERED` on every Sheets write is a formula-injection vector.** Any title like `=IMPORTDATA(...)` or `=HYPERLINK(...)` runs as a live formula on every sheet open. Trivially triggered by AI-generated descriptions too. (Source: Google R1 #2 + QA R1 #1.)

4. **The AI feature has never worked.** `analyzeWithClaude` (line 329) calls `api.anthropic.com` with no `x-api-key` header, and the browser would block it on CORS anyway. Every analysis call has been silently landing in the catch block. (Source: AI manager R1 + spec.)

There are ~20 more high-severity bugs (broken WebGL homography that's actually affine; AI auto-fill clobbering user input; non-atomic Drive+Sheets writes leaving orphan files; OAuth popup cancellation hanging forever; no `img.onerror` anywhere; no keyboard navigation). They're in the per-agent reports.

---

## What "rewrite" should mean concretely

After Round 2, the rewriters narrowed scope. Nobody wants to throw out everything. The convergent view:

**Keep verbatim:**
- AI response JSON schema and the `aiData` cache shape (AI manager)
- `cachedToken` structure and the GIS token-client integration (AI manager + Google)
- Information architecture: drop zone → canvas → bottom row of AI box + portfolio fields (Frontend)
- Auto-correct ordering: persp → crop → adjust (Image)
- Manual-crop's source-resolution preservation (Image — line 471 is the only correct pixel op in the file)

**Rewrite:**
- HTML structure: `<title>`, `<lang>`, `<main>`, `<form>`, `<label for>`, keyboard-accessible drop zone
- CSS: define `:root` token block, align with `index.html` palette (bone/ink), drop the SaaS-admin chrome
- Pixel-quality layer: maintain a **full-resolution off-screen canvas** as source of truth; the visible canvas becomes a preview only; export, Drive upload, and watermark composition all read from full-res
- Perspective: replace the broken affine-per-triangle WebGL with a real CPU homography (one-time per apply, no shaders needed)
- AI integration: swap broken Claude call for a working provider (see decision below); store coordinates as normalized fractions, never re-apply to already-cropped images
- Sheets writes: `RAW` not `USER_ENTERED`; atomic Drive+Sheets with in-flight lock; honor `expires_in` instead of hardcoded 58-min token cache
- Deployment: serve via `python3 -m http.server`, register `http://localhost:8000` as authorized JS origin; document this in a README

---

## What stays out of scope

- Matching the Python pipeline's pixel quality. The bar is **Instagram + portfolio (3000 px long edge, sRGB)** — not a port of `_find_frame_corners_from_hough`. (Image processing, R2.)
- ICC/P3 color preservation. Canvas is sRGB; iPhone P3 will shift; document and move on.
- Mobile-first responsive layout. Sveta uses desktop; mobile is "doesn't break" not "delightful."

---

## Conflicts that need Sveta's decision

The team converged on most things. These are the few they did not:

### 1. AI provider

| Option | Pro | Con |
|---|---|---|
| **Gemini 2.0 Flash free tier** (AI manager primary rec) | $0, structured JSON output, browser-friendly, generous quota | Needs a free Google AI Studio API key. Cloud call (~1–3 s latency). New scenarios for QA: 429, malformed JSON, slow network. |
| **Manual paste into ChatGPT.com** (AI manager fallback) | Uses existing ChatGPT Plus. Zero new dependencies. | Extra clicks per painting. JSON discipline on Sveta. |
| **Skip AI entirely for v2** | Removes a class of bugs. Manual field entry is fast anyway. | Loses the auto-fill feature the original brief asks for. |

**Team's lean:** Gemini Flash primary, manual-paste affordance as fallback even when Gemini is wired up (for offline / quota-exhausted days). One agent (AI manager R2) recommends *deferring the provider swap until the rewrite is in*, so it's not done on a broken base.

### 2. Frontend rewrite scope

- **Image-processing R2:** "Frontend's rewrite is largely aesthetic re-skin; should not gate the pixel-quality rewrite."
- **Frontend R2:** "Structural rewrite, not aesthetic. Keyboard accessibility, semantic HTML, defined tokens — that's structural; can't be patched."

These aren't really in conflict in practice — they're independent layers and can be done in parallel. The question is whether to do them in one commit or two.

---

## Recommended next move

Based on the team's combined verdict:

1. **Decide AI provider** (Gemini Flash / manual paste / skip). One choice now, can be changed later.
2. **Decide rewrite shape:** (a) one big rewrite, or (b) layered — pixel-quality first, then structure/style, then AI.
3. **Then** I'd dispatch the implementation, using the agents' specific findings as the spec. The blockers above are the acceptance criteria.

Two open questions for you. I'll ask them one at a time.
