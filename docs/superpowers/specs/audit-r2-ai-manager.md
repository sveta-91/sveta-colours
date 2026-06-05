# AI Integration Manager — Round 2 Critique

## Updated verdict
`rewrite` — I moved. My lane's fixes are still correct, but Image-Processing's export-at-preview-resolution bug and Frontend's undefined-tokens blocker mean the file ships thumbnails through an unstyled UI, which makes "improve the AI call" beside the point.

## Strongest points I agree with from the others

- **Image-Processing's export-at-display-resolution finding** (their Issue #1, `photo_manager.html:248–254, 479, 517`). This is the single most damaging bug in the file and it makes my Issue #5 (clamp AI corners, validate crop area) look small — the AI could return perfect pixel-accurate corners and the export would still be a ~600×460 thumbnail. Any AI work I recommend is downstream of this fix.
- **Image-Processing on affine-per-triangle WebGL warp** (their Issue #2, line 449). I noted in my Round 1 that AI-supplied corners are trusted blindly; I missed that even *correct* corners produce a bent warp because there is no projective divide. The AI's geometry output isn't the only suspect.
- **Frontend's "undefined design tokens" blocker** (their Issue #1, lines 7, 12–62). I missed this entirely. The page renders nearly blank on `file://`, which is the exact deployment mode I was optimizing for. My "primary: Gemini-via-REST from `file://`" recommendation assumed a usable UI underneath.
- **QA's "AI auto-fill clobbers user-typed fields"** (their Issue #4, line 320–325). I called the lazy-cache pattern a strength in Round 1 but missed that `fillFields` overwrites already-typed values. Any provider swap I recommend must also gate `fillFields` on "is this field empty?", not just "did the AI return something?".
- **Google's formula-injection vector via AI-generated description/tags** (their Issue #1). I treated the AI's output as a UI concern; Google's right that prompt-injected EXIF/visual content could turn the description into a `=IMPORTDATA(...)` payload that hits Sheets. The AI output sanitation isn't only about parse robustness — it's a security boundary.

## Points I push back on

- **Frontend calls for a rewrite primarily over styling and a11y.** Those are real, but they don't invalidate the JS architecture — the lazy `analyzeWithClaude` cache, the gated auto-correct triggers, and the JSON-schema shape are sound and survive any visual reskin. Frontend's lane and mine disagree on what "rewrite" means: I read it as "structure is wrong"; they read it as "presentation layer is unsalvageable." Both can be true while my AI-path scaffolding still gets ported as-is.
- **QA's Issue #5 (Drive+Sheets non-atomic, orphan files) is real but not in my lane and doesn't change the AI recommendation.** Mentioning for completeness — it's solvable independently of provider choice.
- **Image-Processing claims sending an 800px JPEG to the AI means coords are accurate to ±15 source px** (their out-of-scope note). True for *crop rects*, false for *quality assessment* — Gemini at 800px is plenty for "is this tilted / is there background visible / what medium." We should still send 800px for description+tags+needs_* flags, and *not* trust AI coordinates as transform inputs at all. That's a sharper rule than my Round 1 "clamp and validate."
- **QA implies the AI feature is broken-as-shipped end-to-end** (their out-of-scope). Correct that the call fails, but the *data contract* — the requested JSON schema — is the most reusable thing in the file. A rewrite should keep that schema verbatim.

## What I'd add now, having seen the others

- **Resolution mismatch between AI input and transform output is a cross-lane bug.** Image-Processing flagged the export-resolution bug; nobody connected it to the AI call. The AI sees 800px, returns fractional coordinates, the transform applies them to whatever resolution `workingImg` currently is — which after one crop is no longer the resolution the AI analyzed. QA's "re-running Auto-correct after Apply Crop re-applies the same fractional crop" (their auto-correct bug) is the same root cause. The fix is: every AI result must be tagged with the resolution+crop-state it was computed against, and invalidated when either changes.
- **AI's description/tags are an XSS vector AND a Sheets-injection vector AND a screen-reader-attack vector.** Frontend noted `innerHTML` of AI text in `#aiBox` (their a11y bullet). Google noted formula injection. Combined: AI output needs a single sanitization step that escapes HTML, strips leading `=+-@`, and trims control chars — applied once on receipt, before it touches DOM or Sheets.
- **`drive.file` scope interacts with AI-aided dedup.** Google flagged that `drive.file` can't list pre-existing folder contents. If we later want the AI to consult Drive for "have we seen a similar painting?", the scope choice forecloses it. Worth deciding now.

## Cross-lane implications for my recommendation

The Gemini-Flash-primary / manual-paste-fallback recommendation **still holds**, but with three modifications in light of the other findings:

1. **Don't trust AI coordinates as transform inputs.** Use AI output for `description`, `medium`, `style`, `tags`, `quality_notes`, and the boolean `needs_*` flags only. For perspective corners and crop rect, the Python pipeline's Hough-based geometry (via a small server-side helper if we keep this browser-first, or as a porting target) is the source of truth. This moots Image-Processing's correct objection that AI-at-800px can't drive a pixel-accurate transform.
2. **Sanitize AI output once, centrally, before it reaches DOM or Sheets.** One function: strip leading formula chars, escape HTML, trim. Both Google's and Frontend's findings collapse to this.
3. **Defer all AI work until the export-resolution and design-token blockers are fixed.** Until then, AI suggestions paint a thumbnail prettily — wrong order of operations. The provider swap should be done in the rewrite, not before it.

Manual-paste fallback becomes more attractive in the rewrite world: it sidesteps the sanitization-on-receive complexity (Sveta reviews the JSON before pasting), and it survives the inevitable Gemini free-tier policy change.
