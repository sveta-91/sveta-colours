# Photo Manager — Team Audit Spec

**Date:** 2026-06-05
**Target file:** `tools/photo_manager.html` (560 lines, staged)
**Phase 1 goal:** parallel audit + one round of critique → user decides next step (improve existing vs rewrite).

## Background

Sveta wants an HTML page where she can upload an artwork photo, have AI process it (crop, perspective correction, normalize, watermark), name it, save to a specific Google Drive folder, and append/update a row in a specific Google Sheet. AI also pre-fills metadata fields.

An existing implementation at `tools/photo_manager.html` covers most of this surface (upload, crop, perspective, watermark, brightness/contrast/saturation, resize, Drive save, Sheets update, AI analysis). Before deciding whether to improve or rewrite, a team of specialised agents audits the file and produces a verdict.

## Constraints

- **No additional paid services.** Sveta has a ChatGPT Plus subscription only — no separate OpenAI API key, no Anthropic API key, no Ollama installed.
- **Local-only deployment is acceptable.** The page runs in Sveta's browser on her Mac, no server hosting.
- **Drive folder:** `1M8wTOfPAYeJZZL6d7RYo2DLtIapuxDOX`
- **Sheet:** `170BuYN45Wk7H2g31gfRo35quok76JwrhFB8YV4-ckfI`, gid `1031748871`, tab `Sheet1`
- **Sheet schema (existing):** Date, Title, Description, Medium, Size, Year, Tags, Status, Price, Notes, Dimensions, Format, Drive link
- **Watermark default:** `© sveta-colours`
- **Known broken-as-shipped:** The Claude API call at line ~335 has no `x-api-key` header and would hit CORS — i.e. the AI-analysis feature is a placeholder.

## Team (5 agents)

| Role | Focus |
|---|---|
| AI integration manager | Research AI-provider options under "no extra payment, local OK". Produce price/performance matrix (Ollama local, Gemini free tier, manual paste via ChatGPT.com, free credits). Recommend an approach. |
| Google integration expert | OAuth scopes, CORS, Drive multipart upload correctness, Sheets append-vs-update logic, dedup-by-title race conditions, header-drift handling, error surfaces. |
| Frontend developer | UX, accessibility (a11y), markup quality, progressive disclosure, button states, error states, mobile/responsive, dark-mode tokens used in the file. |
| Tester (QA) | Concrete scenarios with repro steps: drag&drop variants, HEIC/PNG/WEBP/huge files, broken images, offline, OAuth failure, duplicate titles, double-clicks, slow networks, edge cases. |
| Image processing expert | Quality of crop / perspective / normalize / watermark logic. Compare to what the existing Python pipeline at `tools/image_pipeline/` already does. Where AI helps, where it hurts. |

## Process

### Round 1 — parallel independent audit

Each agent reads `tools/photo_manager.html` end-to-end and produces a markdown report:

- **Verdict** (1 of: `keep`, `improve`, `rewrite`)
- **Top 3 strengths** (specific, with line numbers)
- **Top 5 issues** (severity: blocker / high / medium / low; with line numbers and proposed fix)
- **Out-of-scope concerns** (what they noticed but isn't their lane)

Agents are **read-only** in this phase — no edits to the file.

### Round 2 — one critique pass

Each agent reads the other four reports and produces a short reaction:

- Where they **agree** with another agent (and why it strengthens the case)
- Where they **disagree** (and why)
- Anything they want to **add** based on what others raised

### Round 3 — consolidation

Claude (orchestrator) compiles a single decision-ready summary:

- Consensus verdict (or split, if no consensus)
- Blocker issues all agreed on
- Conflicts requiring Sveta's call
- Recommended next move

Sveta then decides: improve existing / rewrite / mixed approach.

## Out of scope (Phase 1)

- Writing or editing code in `photo_manager.html` (Phase 2 only, after Sveta's decision).
- Choosing the actual AI provider — the manager recommends, Sveta picks.
- Deploying anywhere beyond local file://.
