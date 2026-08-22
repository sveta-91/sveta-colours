---
name: sveta-writing
description: Use when writing or editing any user-facing text for sveta-colours.com (painting descriptions, About statement, gallery copy, FAQs) and when interpreting Sveta's prompts in any context. Reads every prompt in both English and Russian; writes in Sveta's voice.
---

# Sveta Writing

This skill governs two related jobs:

1. **Reading Sveta's prompts** — always bilingual (English + Russian).
2. **Writing user-facing text for her site** — in her voice.

Job 1 applies to every interaction. Job 2 applies whenever you produce text that will be visible to her buyers.

---

## 1. Reading prompts (bilingual rule — applies always)

**Always analyze Sveta's prompts in both English and Russian.** She mixes languages mid-thought, and the Russian portions almost always carry the truer meaning — she switches to Russian when she wants to be most direct about what she actually wants. Never treat Russian as a redundant paraphrase of nearby English.

Observed patterns:

| Sveta wrote | Don't read it as | Read it as |
| --- | --- | --- |
| *"убери упоминания watercolor"* | a soft suggestion | a direct order — cut the watercolor line, no clarification needed |
| *"'feel calm and warm' - not really, I want to express my temperament and my feelings of the moment, may be not calm and not warm"* | her contradicting herself | the actual brief — the Russian-shaped sentence is what she means; rewrite accordingly |
| *"разные формы, еще до цветов: стопка книг, рукописи в свитках, человеческие профили..."* | a side-note | substantive content, written more freely than she would write it in English |
| *"аре тчере нехт степс планнед?"* | Russian | English typed on the Cyrillic keyboard layout — parse as English |

Rules:

- **Translate Russian internally before responding.** Don't ask her to translate — she switched languages on purpose.
- **If the English and Russian halves diverge, prefer the Russian.** It's the honest version.
- **Preserve connotation, not literal words.** If a Russian phrase carries something English can't quite render, keep the connotation in your interpretation.
- **Reply in English** unless her message is entirely in Russian. Mixed prompts → English response.
- **Cyrillic-keyboard typos** (e.g. *"аре тчере"*) parse as English. Don't get confused.

This rule is also stored in user-memory so it fires regardless of whether this skill is invoked.

---

## 2. Writing in Sveta's voice (for site-visible text)

### Voice anchor — the About statement (locked)

This is the source of truth for Sveta's voice. Every rule below traces back to a pattern in these three paragraphs:

> I've always doodled — on whatever paper was at hand — but I'd assumed learning to paint properly was something other people did.
>
> Eventually I tried, starting from the very beginning: discovering that a tree shadows itself from within, lit only on the outside, lit from below at sunset — and that mixing colors is a sort of witchcraft.
>
> My paintings start from a memory — a walk through the woods, warm evening light, wandering through the city. Each one carries the feeling I had then — whatever it was.

If a draft doesn't sound like the same person wrote it, the draft is wrong. Not the voice.

### Voice spec — painting descriptions

*Locked from the 2026-06-10 brainstorm panel (R2 director synthesis). Adjust by appending to the Evolution log below.*

#### Length

- **32–55 words per description. Hard ceiling 60.**
- **2–4 sentences; three is the sweet spot.** Two when the painting is small and spare. Four only when the fourth is half a clause.
- Below 32 words you can't fit a concrete-plus-lyrical pair AND a hedge without reading captioned. Past 60, the buyer feels sold to.

#### Sentence shapes

- **Em-dash count: 1–2 per description, never 3.** Aim for 1 most of the time, 2 when the second pivot earns it. The About's density (~1 per 17 words) is the *upper* bound, not the target — same density 17 times in a row IS the AI fingerprint.
- **Em-dash function** (either / or, not both): (a) appositive adding a concrete detail, or (b) trailing hedge / ownership. Never as a dramatic pause. Never as a colon substitute.
- **Opening: flat-footed, concrete-first.** Mirrors the About's *"I've always doodled," "Eventually I tried," "My paintings start from a memory."* No participial openers, no inversions.
- **Closing: trail, don't button.** Hedge, name, or hand off. Never end on an abstract noun. The About ends on *"whatever it was"* — concrete shrug.
- **One comma-list of parallel noun phrases when it fits.** Three items, no "and." Reach for it; don't force it.

#### POV / tense

- **First person, used sparingly.** Across all 17 descriptions, expect roughly half to use "I"; the rest let the painting be the subject.
- **Past tense for the memory, present tense for what's in the painting.**
- "I" only when noticing or remembering. Never *"I love how…"* or *"I wanted to capture…"*.
- **No second person. No "you."** No *"the artist," "this piece," "the painting"* — gallery-card distance.

#### Required ingredients

Every description must have **all of (A) and (B), plus at least one of (C):**

- **(A) The small thing.** One noticed physical detail visible in the painting that a casual viewer would scroll past. The single most important constraint. Never the obvious subject — if it's a coffee-mug painting, the small thing is not the mug. The clock with no hands. The wet path catching light. The one red coat.
- **(B) One hedged lyrical move.** *"A sort of," "something like,"* a trailing fragment, *"— whatever it was."* Without a hedge, the lyrical claim tips into mystical-artist cliché. Hedge the *meaning*, never the *subject*.
- **(C) One of these three About-patterns carries the description, rotating across paintings:**
  - Em-dash rhythm pivoting setup-to-claim, OR
  - A concrete comma-list of parallel noun phrases, OR
  - Explicit ownership of uncertainty.

A **memory anchor** is encouraged, not required. If it shows up naturally as past-tense framing, take it. Forcing it on every painting reads as my-process gallery copy.

#### "From the easel" voice

Allowed in roughly **1 in 5 descriptions** — so 3–4 of the 17 total, not more. When used, must be a **fact, not a feeling.**

- *"I painted the mountains first and the jar last"* — fact, in.
- *"I struggled with this one"* — feeling, out.

Two shapes are OK: a choice she made (*kept the wood grain showing, moved the jar closer than it was*), or a noticing while painting (*the green I needed was somewhere between two greens I had*). If it wouldn't pass "would Sveta say this out loud at the easel," cut it.

### Forbidden moves (cut on sight)

- **Invitation verbs:** *invites, draws you in, beckons, asks.*
- **Art-statement vocabulary:** *explores, captures, evokes, conveys, expresses, renders, juxtaposes, interplay, narrative, journey, essence, themes of, meditation on, reflection on, a study in.*
- **AI-tell abstract nouns:** *interplay of, play of light, dance of, symphony of, tapestry of, weaving, a sense of, a feeling of, an atmosphere of, a moment of grace, a quiet hope.*
- **Mood prescription:** *peaceful, serene, calm, cozy, contemplative, nostalgic, wistful, bittersweet, joyful, vibrant, luminous, ethereal, evocative.*
- **Buyer projection / second person:** *you'll feel, imagine yourself, brings to mind, reminds one of, for the collector who, perfect for.*
- **Gallery-card distance:** *this piece, this work, the painting, the artist.*
- **Color filler:** *rich with, imbued with, suffused with, awash in, drenched in, bathed in.*
- **Domain clichés:** *golden hour, kissed by sunlight, whispers of autumn, snow-kissed, a riot of color, solitary figure, bustling city.*
- **Symbolism announced:** *"the red coat representing…"*
- **Title-restating:** *"In Morning Coffee, a morning coffee…"*
- **Sentence fragments for poetic effect:** *"Stillness. Warmth. A red coat."*
- **Three-act arcs ending in feeling:** *"You see X, notice Y, finally feel Z."*
- **Exclamation points. Rhetorical questions. Colons introducing mood lists.**
- **The word *beauty*.** And *beautiful, stunning.*
- Price hints, biographical reach-backs, room-projection (*"would feel at home above a reading chair"*).

### Voice markers to preserve

- Em-dash as pivot tool (setup → specific, or specific → hedged claim).
- *"Sort of X"* hedge or one of its variants.
- Ownership of variation (*"whatever it was"*).
- Concrete-then-lyrical inside a single sentence when possible.
- Light named precisely (*"warm evening light," "low winter sun," "light filtered through yellow leaves"*). No fancier synonyms.
- One adjective per noun, max. No adjective stacks.
- Plain SVO sentences with em-dash extensions. No inversions, no participial openers, no appositive stacks.
- Kitchen-table register: *doodled, tried, walk, woods, evening.* Negative space of this register = the Skeptic's avoidance list.

### Manifesto (read before drafting any line)

> Name what's actually there. Hedge the meaning, never the thing. Let the em-dash do the feeling.

If a draft says more than this, cut.

### The pre-publish test

Before any description ships, all three must pass:

1. **Anchor test.** Hide the image, read the line. Can a buyer sketch one specific detail of the painting from text alone?
2. **Voice test.** Does the line sound like the same person who wrote the About statement?
3. **Search test.** Could the line, googled, appear in another artist's portfolio? If yes, kill it.

If any test fails, rewrite — don't loosen the test.

---

## 3. Scope — when this skill applies

**In scope (site-visible text):**

- Painting descriptions for the sheet's `description` column.
- About-page edits.
- FAQ, contact, or any other site copy.
- Captions, labels, headlines, button text.
- Commission/portfolio framing copy.

**Out of scope (use ordinary terse-and-technical voice):**

- Code comments.
- Internal docs, READMEs, playbooks.
- Commit messages.
- Shell output, error messages.
- Conversation with Sveta about the work (be a normal teammate, not the site voice).

The bilingual reading rule still applies in the out-of-scope contexts; the voice rules don't.

---

## 4. The bar

> Any buyer reads any description and recognizes the same voice as the About statement. Each description is anchored in *this specific painting*, not generic. Sveta says "yes, this is me" with no edits she'd un-do within a week.

---

## 5. Evolution log

Sveta asked for this skill to be **iterative — adjust by the way.** Append a dated note when something material changes.

- **2026-06-10 — v1.** Created. Bilingual reading rule + voice anchor.
- **2026-06-10 — v1.1.** Voice spec locked from R2 director synthesis of the paintings-descriptions panel. Key rulings: 32–55 word range with 60 ceiling; em-dash cap 2 (typical 1); required-ingredients restructured into (A) small thing + (B) hedged lyrical + (C) one of three About-patterns rotating; "from the easel" voice allowed ~1 in 5, fact-not-feeling; *beauty* added to hard bans; memory anchor demoted from required to encouraged. 5 sample descriptions drafted; awaiting Sveta's review of Morning Coffee outdoor-vs-window question.
