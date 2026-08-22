# Brainstorm-panel log — sveta-colours

Append-only record of what each panel run produced and what got tuned at the user's review gate. Used by `brainstorm-panel` skill (Step 6) to bias future proposals on this repo.

---

## 2026-06-08 — gallery/catalog grid in `index.html` (improving existing artifact)

- Proposed: gallery curator (director), visual designer, frontend dev, collector/buyer advocate, adversarial UX skeptic. User added: **UX/UI design auditor**. User removed: none.
- Style: swarm → director-led, 3-round cap. Ran 2 rounds (R1 cold + R1.5 with inventory facts). Fit well — diverge produced 41 distinct critiques, director synthesis kept it coherent.
- **Big lesson: pull real inventory data before convening.** R1 assumed ~50 paintings; actual was 17, all priced identically, all acrylic, /postcards empty. Half the R1 moves had to be re-derived in R1.5. *Next panel run on this site: snapshot live data before drafting the bar and convening roles.*
- User gate edits trended toward "keep more on the card, lean on typography" — Designer + Curator's keep-meta-and-price call edged out Buyer + Skeptic + Auditor's strip-down consensus. Note for future panels here: don't read R1.5-style consensus as inevitable; user weights chromatic/typographic distinctness over information minimalism.
- Auditor (the user-added role) earned its seat — surfaced WCAG 2.4.10 (chapter headings on /works), focus-visible during stagger, and the postcards-empty-state Nielsen violation that Curator + Buyer had also flagged from different angles. **Keep proposing the UX/UI auditor on visual / cross-context UI targets in this repo.**
- Skeptic earned its seat too — caught the catalog-№ over-reach and the 8-of-17 "preview-not-curation" framing. Disagreement between Skeptic and Curator was the most productive axis.
- Domain quirk for next time: **don't infer painting orientation from photo thumbnails** — photos include landscape context around portrait paintings. Use the `orientation` column or ask. (Also captured as a feedback memory.)

---

## 2026-06-09 — image aspect ratios on /works (choice between approaches)

- Proposed: Curator (director), Photographer, Frontend, Buyer, UX/UI Auditor, Skeptic. User added: none. User removed: none. **Auditor's standing seat held up** — caught the CLS jump from `onLoad` overriding `--ratio` that nobody else flagged.
- Style: swarm → director-led (Curator). 2-round cap. Both rounds ran clean.
- Convergence was strong: 4 of 6 picked A (force canvas ratio + letterbox); Photographer wanted C-only; Skeptic dissented toward E. Director paired A (now) + C (publish-time guard in `push_painting.py`).
- **Inventory snapshot before convening paid off again.** Pulled live photo aspects via Selenium and discovered the actual mechanism: `onLoad` handler overrides `--ratio` to photo-natural when off >5%. That detail reframed the entire discussion — without it the panel would have argued in the abstract.
- Skeptic's strongest counter — "visible defect is 2-3 photos, not 7" — was empirically correct (4 of the 7 "overridden" cards happen to render identically). Director conceded the framing and adopted C as ingest-time enforcement rather than manual discipline. Note: when Skeptic is right about scope, fold their framing into HOW we implement, not whether.
- Recurring across two panels: **buyer trust signals + visual consistency** trump information minimalism. Pattern holds; promote to CLAUDE.md if it survives one more run.
- Note for next time: when the user's prompt is an open question ("should X be adjusted and how?"), option E (do nothing) is implicitly off the table by the question's framing. State this explicitly so the Skeptic's E-pick gets engaged on the substance (visible defect or not?) rather than treated as a procedural exit.

---

## 2026-06-10 — About-page statement refinement (improving existing artifact)

- Proposed: Literary editor (director), Memoirist editor, Buyer, Anti-AI skeptic. **User added: Painter who began in her 50s, now in her 80s** — a peer-experience seat. Earned every minute of it: the truest moves in R2 (the door-had-closed framing, "one forgives me, the other doesn't") came from her notes, not from the craft roles.
- Style: swarm → director-led (literary editor). 2-round cap, ran clean.
- **Convergence was extraordinary** across 5 R1 outputs: 4 of 5 roles flagged the witchcraft/magic echo, 4 of 5 flagged the "inviting the viewer" closer. Director just had to pick *which* substitute, not whether to substitute.
- **Lesson on user-added roles:** the Painter's seat replaced what would have been an editorial pile-on. Editor + Memoirist + Skeptic could have agreed to cut the medium sentence; the Painter argued to *replace* it with a lived contrast ("one forgives me, the other doesn't"). The right move was hers, not theirs. **When the user adds a seat, it usually outranks the proposed roster on whatever axis it represents — give it more weight than a 5th voice would normally get.**
- Anti-AI skeptic continued to earn its standing seat: flagged generic patterns (e.g., "Many of," "simple memory," "the viewer") with specific cuts attached.
- Recurring across three panels now: **Sveta wants flavor preserved when in doubt.** Director sided with Painter+Buyer on protecting "learning to really see" against Skeptic's cliché flag. Same instinct as "keep more on the card, lean on typography" from the gallery-grid panel. Promoting this to CLAUDE.md when the next panel confirms it.
- Process note: the artifact-as-context worked well. Sveta's draft text was the data; no separate inventory needed. For prose-editing targets here, the artifact IS the data.

---

## 2026-06-10 — painting descriptions spec (greenfield, voice-derivation)

- Proposed: Literary editor (director), Voice anthropologist, Visual storyteller, Painter who began in her 50s, Anti-AI skeptic, Buyer/collector. User added: none — but Sveta added a meta-directive mid-run that became the most durable outcome: **`sveta-writing` skill** at `.claude/skills/sveta-writing/SKILL.md`, with a bilingual prompt-reading rule that applies always (also saved as feedback memory).
- Style: swarm → director-led (Editor). 2-round cap, ran clean. All 6 R1 agents returned in ~50s each in parallel; R2 director synthesized in ~110s with no rewrites needed on the 5 sample drafts.
- **Convergence was strong on the bans, divergent on the requirements.** Editor wanted "all three" required ingredients; Buyer wanted "at least one of three" rotating; Painter wanted "the small thing" elevated. Director folded all three by splitting *content* requirements (small thing + hedge, both mandatory) from *voice-pattern* requirement (one of three, rotating). The compromise resolved the templated-feel risk Buyer flagged without losing Editor's structural floor.
- **Anti-AI skeptic continues to earn its standing seat** — the ~30-word ban list and the em-dash cap rationale (the About's density is the *upper* bound, not the target) were skeptic-driven and survived intact.
- **Painter's "small thing" rule turned out to be the load-bearing constraint.** Without it, all the voice-marker rules still allow generic landscape copy. With it, each description must point to something a non-painter would scroll past. This is the rule the next greenfield-voice panel should propose by default.
- **Format note: the voice anchor IS the data, the painting subjects are scaffolding.** For voice-derivation panels here, lead with the locked voice source (About statement), not with the deliverables.
- **Promoting the "preserve flavor over minimalism" pattern to CLAUDE.md.** Three panels now: gallery grid kept meta + price, About kept "learning to really see," descriptions promoted "the small thing" over efficiency. Confirmed enough.
- Domain quirk for next time: **Sveta switches into Russian when she means it most.** Build that into the prompt-reading layer (now codified in `feedback_bilingual_prompts.md` memory + the sveta-writing skill).
- Process note: skill-as-deliverable is durable. Earlier panels left lessons in the log; this one left a callable spec at `.claude/skills/sveta-writing/SKILL.md` that the next writing task picks up automatically. Worth doing this for future voice/style/UX panels here.
