# AI Integration Manager — Round 1 Audit

## Verdict
`improve` — the surrounding canvas/Drive/Sheets scaffolding is reusable, but the entire AI path is non-functional and the provider choice needs to be redone before any other work is worth doing.

## Top 3 strengths

- **AI data shape is well-designed for downstream consumers** (`photo_manager.html:339-354`). The JSON schema asked of the model bundles everything the UI needs — description, medium, style, tags, three adjustment ints, perspective corners as normalised fractions, crop rect as fractions, and a `quality_notes` field — in a single call. That structure is provider-agnostic and worth preserving when the backend is swapped.
- **Graceful degradation if AI is silent** (`photo_manager.html:294-309`). Each auto-correct stage is gated on `aiData?.needs_*` flags, so when the API fails the user still gets the manual crop/perspective/sliders. The file does not hard-block on AI success.
- **Lazy `analyzeWithClaude` trigger + silent re-use** (`photo_manager.html:241, 290, 426`). Analysis fires once on upload and is cached in `aiData`; subsequent `Auto-correct` and `Auto-detect` calls reuse the cached result instead of re-billing. This is exactly the right pattern for any paid or rate-limited provider.

## Top 5 issues

1. **[blocker] Claude API call cannot work from the browser** (`photo_manager.html:335-356`). No `x-api-key`, no `anthropic-version`, no `anthropic-dangerous-direct-browser-access`, and Anthropic's API does not send CORS headers to `file://` or arbitrary origins. Every call lands in the `catch` at line 371. **Fix:** remove the direct `fetch` to `api.anthropic.com` and replace with one of the providers below (primary recommendation: Gemini 2.0 Flash via REST with a key embedded in the local-only HTML).
2. **[high] Hard-coded to a single, paid provider Sveta does not have credentials for** (`photo_manager.html:337`). `claude-sonnet-4-20250514` requires an Anthropic key Sveta does not own and cannot get under the constraints. **Fix:** introduce a `const AI_PROVIDER = 'gemini' | 'manual' | 'none'` switch and a thin `analyze(file)` wrapper so the call site at line 241 doesn't care which backend answers.
3. **[high] No API-key plumbing of any kind** (entire file). There is no input, no `localStorage` read, no constant — the request would be unauthenticated even if CORS were solved. **Fix:** add a one-time prompt that stores the key in `localStorage` (`gemini_api_key`), with a small "Reset key" link in the AI card. Acceptable for a local-only file; do **not** commit the key.
4. **[medium] JSON-from-LLM parsing is brittle** (`photo_manager.html:359`). `JSON.parse(text.replace(/```json|```/g,'').trim())` will throw on any prose preamble, trailing comma, or truncated response, and the catch surfaces "API error: Unexpected token" — confusing for the artist. **Fix:** extract the first `{ ... }` block via regex, validate required keys, and on parse failure fall back to a "describe-only" mode that still fills `description` and `tags` from raw text.
5. **[medium] Perspective corners and crop rect are trusted blindly from the model** (`photo_manager.html:294, 299, 380, 393`). LLMs are notoriously bad at pixel-accurate geometry; an off-by-10% corner produces a visibly skewed warp on a real painting. **Fix:** clamp corners to `[0,1]`, reject crops with area < 25% or > 99%, and require the four perspective points to form a convex quad (cross-product sign check) before `runAutoPersp()` fires.

## AI provider matrix

| Provider | Setup effort | Cost/month | Latency / image | Quality (artwork analysis) | Browser-friendly? | Rank |
|---|---|---|---|---|---|---|
| **Google Gemini 2.0 Flash (free tier)** | One-time: create key in AI Studio, paste into localStorage prompt | $0 (15 RPM, 1500 req/day free) | 2–4 s | Strong — handles description, medium/style inference, tags, and rough crop/perspective hints in one call | **Yes** — REST endpoint serves CORS, key-in-URL works from `file://` | **1** |
| **Manual paste into chat.openai.com (ChatGPT Plus)** | None (already paid) | $0 incremental | 10–30 s of human time per image | Best quality, but Sveta has to copy/paste prompt + image + paste JSON back | N/A — out of band | **2 (fallback)** |
| **Ollama + LLaVA / Qwen2-VL / MiniCPM-V** | One-time install (~4 GB model download) + `ollama serve` running | $0 | 5–20 s on M-series Mac | Decent description, weak on style attribution and unreliable on pixel coordinates | Yes via `http://localhost:11434` (Ollama sends `Access-Control-Allow-Origin: *` since 0.1.34) | 3 |
| **transformers.js / WebGPU (e.g. Moondream, SmolVLM)** | One-time: ~1–2 GB model download into IndexedDB on first run | $0 | 15–60 s first load, 5–15 s after | Weak — small VLMs hallucinate medium/style; coordinates unusable | Yes, fully in-browser | 4 |
| **Groq / Together / Cerebras free credits** | Sign up, key in localStorage | $0 until credits exhaust (weeks, not months) | 1–2 s | Strong on Llama-3.2-Vision-90B, similar to Gemini | Yes, CORS-enabled REST | 5 (rolling expiry makes it unfit as primary) |
| Anthropic direct (current code) | — | Needs paid key | — | — | **No** (CORS blocks browser) | excluded |

## Recommended path

**Primary: Google Gemini 2.0 Flash via REST, key stored in `localStorage`.** It is the only option that (a) costs nothing within Sveta's realistic volume — a painter producing a handful of artworks per week is two orders of magnitude under the 1500/day free quota, (b) works from `file://` without any local server, daemon, or proxy, (c) returns vision-grounded structured JSON of comparable quality to Claude/GPT-4o for this task, and (d) requires only a one-time key paste, not an ongoing process. Replace the `fetch` at line 335 with `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=…`, send the image as inline base64 in `parts`, and keep the existing JSON schema almost verbatim (Gemini supports `responseMimeType: "application/json"` + `responseSchema` for guaranteed parseable output, which also fixes issue #4).

**Fallback: manual paste workflow.** Add a "Copy prompt + image" button that copies the same JSON-schema prompt to the clipboard and downloads the image; Sveta pastes both into ChatGPT.com, copies the JSON reply back into a textarea, clicks "Apply". Zero external dependencies, uses the subscription she already pays for, and survives any API outage.

## Out-of-scope concerns

- The OAuth scope `drive.file` (`photo_manager.html:488`) only grants access to files this app created — re-running the page in a different browser profile cannot update files saved earlier. Google integration expert's call.
- The Sheets dedup loop (`photo_manager.html:531`) is a case-insensitive title match with no normalisation; "Autumn Light " (trailing space) and "autumn light" would create dupes. Also Google integration expert.
- The `canvasToBase64()` downscale to 800 px (`photo_manager.html:480`) is fine for cost but actively harms perspective-corner accuracy — the model sees an image where a 1-pixel error becomes a ~10-pixel error at full resolution. Image processing expert should weigh in on whether to send full-res for the analysis call.
- No HEIC support (`accept="image/*"` at line 75 will surface .heic on macOS but `<img>.src` won't decode it). Tester's lane.
- The watermark stroke at line 267 uses `lineWidth=2` regardless of canvas size — fine for the preview, but `downloadImg()` exports the preview canvas, not a full-res render, so the watermark scales with the preview not the artwork. Image processing expert.
