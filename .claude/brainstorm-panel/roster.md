# Brainstorm-panel roster — sveta-colours

What four panels **confirmed**. Start panel #5 here, not from scratch.
`log.md` stays the per-panel narrative; this file holds only what recurred.

---

## Standing seats

Propose these by default on every panel in this repo. Each earned its seat on evidence.

| Seat | Earned it | Panel |
| --- | --- | --- |
| **Anti-AI / adversarial skeptic** | Caught the catalog-№ over-reach and the 8-of-17 "preview-not-curation" framing. Then flagged generic prose ("Many of", "simple memory", "the viewer") with specific cuts attached. Then drove the ~30-word ban list and the em-dash-cap rationale (the About's density is the *upper* bound, not the target), which survived R2 intact. | 2026-06-08, 2026-06-10 (About), 2026-06-10 (descriptions) |
| **UX/UI design auditor** | User-added on the gallery panel; surfaced WCAG 2.4.10 chapter headings, focus-visible during stagger, and the empty-postcards Nielsen violation. Held its seat next run by catching the CLS jump from `onLoad` overriding `--ratio` — nobody else saw it. | 2026-06-08, 2026-06-09 |
| **Painter who began in her 50s, now in her 80s** | User-added to the About panel; produced the truest R2 moves (the door-had-closed framing, "one forgives me, the other doesn't"). Carried forward to the descriptions panel, where her "the small thing" rule turned out to be the load-bearing constraint. | 2026-06-10 (both) |

Note: the Skeptic ↔ domain-lead disagreement (Skeptic vs. Curator on the gallery panel)
was logged as the single most productive axis. Don't staff a panel without it.

---

## Default roster by panel type

**(a) Design / layout** — target is `index.html` or a visual artifact.
Gallery curator *(director)*, visual designer, frontend dev, collector/buyer advocate,
UX/UI auditor, adversarial skeptic. Add a photographer when the target is image
rendering (aspect, crop, hero sizing).

**(b) Prose / voice** — target is site-visible copy.
Literary editor *(director)*, memoirist editor, buyer, anti-AI skeptic,
painter-in-her-80s. The artifact itself is the data — no separate inventory needed.

**(c) Greenfield spec / voice-derivation** — target is a rule set that doesn't exist yet.
Literary editor *(director)*, voice anthropologist, visual storyteller,
painter-in-her-80s, anti-AI skeptic, buyer/collector. Lead with the locked voice source
(the About statement); the deliverables are scaffolding, not the data.

---

## Coordination style

**Swarm → director-led, 2-round cap.** Ran clean on all four panels (the gallery panel
was capped at 3 and finished in 2). Diverge wide in R1, director synthesizes in R2.
Six R1 agents return in parallel in ~50s each; R2 synthesis ~110s.

Director is chosen as the seat that owns the target's domain: **curator** for design/layout,
**literary editor** for prose and for greenfield voice specs.

---

## Standing rules

1. **Pull a real inventory/data snapshot before convening.** Paid off twice. The gallery
   panel assumed ~50 paintings against an actual 17, all identically priced, all acrylic,
   `/postcards` empty — half of R1 had to be re-derived. The aspect panel pulled live photo
   aspects via Selenium and found the `onLoad` `--ratio` override, which reframed the whole
   discussion. Without a snapshot the panel argues in the abstract.
2. **A user-added seat outranks the proposed roster on its axis.** Weight it above a normal
   5th voice. Editor + Memoirist + Skeptic would have agreed to *cut* the medium sentence;
   the Painter argued to *replace* it with a lived contrast, and hers was the right move.
3. **When the Skeptic is right about SCOPE, fold their framing into HOW, not WHETHER.**
   "Visible defect is 2–3 photos, not 7" was empirically correct; the director conceded the
   framing and adopted the publish-time guard in `push_painting.py` instead of dropping the work.
4. **If the user's prompt is an open question, say explicitly that "do nothing" is off the
   table.** Otherwise a Skeptic picking option E gets treated as a procedural exit instead of
   being engaged on substance.
5. **Confirmed domain preference: Sveta wants flavor preserved over minimalism.** Three panels
   — gallery grid kept meta + price, About kept "learning to really see", descriptions promoted
   "the small thing" over efficiency. Don't read a strip-down consensus as inevitable.
6. **Sveta switches into Russian when she means it most.** Build bilingual reading into the
   prompt-reading layer (codified in `feedback_bilingual_prompts.md` and the `sveta-writing` skill).
7. **Don't infer painting orientation from photo thumbnails** — photos carry landscape context
   around portrait paintings. Use the `orientation` column or ask.
8. **Prefer skill-as-deliverable.** The descriptions panel left a callable spec at
   `.claude/skills/sveta-writing/SKILL.md` that later tasks pick up automatically, instead of a
   lesson buried in the log. Use this output shape for future voice / style / UX panels.

---

## How to update this file

`log.md` is append-only per-panel narrative and stays that way. A pattern gets promoted here
only **after it recurs across panels** — a single run's observation lives in the log. When a
promotion happens, cite the panel dates that earned it, as above.
