# Corrections log — sveta-writing

Append-only record of every time Sveta corrected a text. This is the learning
surface of the skill: `SKILL.md` holds the *rules*, this file holds the
*evidence*, and rules get promoted from here once a pattern recurs.

**Read this before drafting.** The rules in `SKILL.md` tell you what good looks
like; this file tells you how drafts actually go wrong with her.

---

## How to use this file

**After every correction from Sveta on any user-facing text — append an entry.
Same turn, before moving on.** Not at session end; sessions get truncated and
the correction is lost. An entry costs 6 lines.

Entry format:

```
### YYYY-MM-DD — <artifact> — `CATEGORY`
- **I wrote:** <the draft, or the part that was wrong>
- **She said:** <verbatim, in her language — do not translate away the Russian>
- **She meant:** <the reading, once resolved>
- **Fix applied:** <what changed>
```

**Promotion protocol.** Count entries per `CATEGORY`:

| count | status | action |
| --- | --- | --- |
| 1 | observation | lives here only |
| 2 | candidate | note it in the category index below, flag it while drafting |
| 3+ | **rule** | write it into `SKILL.md` body, cite the dates, keep the entries here |

Only *tighten* automatically. A correction that would *loosen* a rule (let a
draft do more) needs Sveta to say so explicitly — as she did on 2026-08-22 with
the word floor. Never infer permission from a single approval; she approves
things she'd still rather not have.

---

## Category index

| category | count | status | one-line rule |
| --- | --- | --- | --- |
| `WRONG-SUBJECT` | 4 | **rule** | The draft described the wrong thing. Ask what she wanted to show before polishing anything. |
| `FACTUAL-WRONG` | 3 | **rule** | I named an object that isn't there. Verify at full resolution; never describe from a thumbnail. |
| `CUT-EXPLANATION` | 2 | candidate | Her version is shorter because it deletes the part that explains. Explanation is the default failure. |
| `MOOD-PRESCRIBED` | 2 | candidate | I assigned a feeling. She wants the thing named, not the mood assigned. |
| `READING-HER-WORDS` | 2 | candidate | Her word was idiomatic, not literal. Resolve intent before acting. |
| `BUYER-PROJECTION` | 1 | observation | Addressing or imagining the viewer. |
| `FLAVOR-OVER-MINIMALISM` | 1 | observation | When in doubt she keeps the richer option. |

---

## Entries

### 2026-11-08 — IMG_0108 (Vancouver lagoon, skyline) — `WRONG-SUBJECT`
- **I wrote:** (Codex draft) "The towers rise behind the trees. A little orange mark on the far bank has a thinner one below it, and the blue edges of the buildings break into strips in the water…"
- **She said:** *"Description should be about touch of sun on the rocks without naming colors."* Kept: *"Across the water, the towers stand under all that cloud. The city looks a little less permanent from here — or perhaps it's just the stones so close."*
- **She meant:** The subject is the light landing on the foreground rocks, not the skyline details. Colours are for the eye, not the text. Location is Vancouver (supplied by her; neither model could tell).
- **Fix applied:** New opening sentence about sun on the rocks, no colour words; her chosen two sentences kept verbatim, including "or perhaps".

### 2026-08-22 — Vancouver — `WRONG-SUBJECT` + `FACTUAL-WRONG`
- **I wrote:** "Everything here is grey-blue except the two red funnels and a line of gold pinpricks along the hull — portholes, lit in the middle of the afternoon."
- **She said:** *"главное, что я хотела показать, что вот этот корабль, он совершенно огромный и, в сущности, из какого-то другого мира… Мрачное лето, которое для ванкувера вообще очень характерно… А какого-то там цветотрубы. И желтое это не иллюминаторы, а спасательные лодки… Но это даже неважно."*
- **She meant:** The subject is the ship's impossible scale and its not-belonging, against a gloomy summer that is characteristic of Vancouver. The funnels and the yellow marks are not the point — and the yellow marks are lifeboats, not portholes.
- **Fix applied:** Rebuilt around scale + gloomy summer. Then she supplied her own text and that replaced mine entirely.

### 2026-08-22 — Vancouver (her rewrite) — `CUT-EXPLANATION`
- **I wrote:** 45 words: "A grey summer, which is most summers here. Then this ship, out of some entirely other world — the little yellow marks along its side are lifeboats, each the size of a bus. The bridge hangs so pale…"
- **She said:** *"сине-серые сумерки. И этот корабль — из какого-то совсем другого измерения. Мост висит на фоне гор такой бледный, что его легко принять за дымку."*
- **She meant:** 26 words, three beats, nothing explained. Note the specific choices: **сумерки** (twilight) not "grey summer" — it's dusk, not weather; **измерения** (dimension) not мира (world) — stronger and stranger; the scale-proof detail deleted entirely. She kept only my bridge line.
- **Fix applied:** Published her text verbatim. Word floor later waived for her own writing.

### 2026-08-22 — Warkworth Lilac Festival — `WRONG-SUBJECT`
- **I wrote:** "A woman in a yellow jacket has stopped to look — the only warm color that isn't the path. Further along, someone else, already past them, not looking."
- **She said:** *"женщина тянется понюхать. Ну, или просто нюхает. Сирень, потому что сирень это не только красиво, но еще и аромат… Ее спутник терпеливо ждет, когда она насладится сиренью."*
- **She meant:** She is *smelling*, not looking — lilac is a scent, and that's why it's lilac. And the second figure is not an indifferent passer-by: he is her companion, waiting patiently for her to finish. I had inverted the warmest fact in the painting into indifference.
- **Fix applied:** Rewrote around scent-that-paint-can't-reach and the waiting companion.

### 2026-08-22 — Warkworth (instruction wording) — `READING-HER-WORDS`
- **I wrote:** —
- **She said:** *"Сделай из этого красивую фразу."*
- **She meant:** "Make this well-made." NOT "use the word beautiful" — *beauty / beautiful / stunning* are on her own ban list. Идиома, не инструкция к слову.
- **Fix applied:** Wrote it well; did not use the word. Flagged the reading to her, and she didn't object.

### 2026-08-22 — description length rule — `RULE-CHANGE` (loosening, explicit)
- **I wrote:** Flagged her 26-word Vancouver as below the panel's 32-word floor.
- **She said:** *"нижняя граница снимается, когда текст пишу я"*
- **She meant:** The floor governs drafts, not her. The spec is downstream of her voice, not above it.
- **Fix applied:** `SKILL.md` Length rule + Evolution log v1.2; `scripts/audit_gallery.py` grew `SVETA_AUTHORED`. Ceiling still applies to everyone.

### 2026-06-10 — About statement — `BUYER-PROJECTION`
- **I wrote:** A closer that invited the viewer in.
- **She said:** (via panel) 4 of 5 roles flagged the "inviting the viewer" closer; she took the substitute.
- **She meant:** The statement ends on her, not on the reader. No hand extended to the buyer.
- **Fix applied:** Final About ends *"Each one carries the feeling I had then — whatever it was."* A shrug, not an invitation.
- Source: `.claude/brainstorm-panel/log.md`, About panel.

### 2026-06-10 — About statement — `FLAVOR-OVER-MINIMALISM`
- **I wrote:** —
- **She said:** (via panel) Sided with Painter + Buyer to protect *"learning to really see"* against the Skeptic's cliché flag.
- **She meant:** A phrase being slightly worn is not reason enough to cut it if it's true. Confirmed across three panels (gallery grid kept meta + price; descriptions promoted "the small thing").
- **Fix applied:** Standing preference; also in `CLAUDE.md`.
- Source: `.claude/brainstorm-panel/log.md`.

### 2026-06-?? — painting copy — `FACTUAL-WRONG`
- **I wrote:** Copy mentioning watercolor.
- **She said:** *"убери упоминания watercolor"*
- **She meant:** A flat order, not a suggestion. She works in acrylic; the medium was simply wrong.
- **Fix applied:** Removed. Also the origin of the bilingual rule — the Russian sentence was the operative one.
- Source: memory `feedback_bilingual_prompts.md`.

### 2026-06-?? — painting copy — `MOOD-PRESCRIBED`
- **I wrote:** Copy built on "feel calm and warm."
- **She said:** *"'feel calm and warm' - not really, I want to express my temperament and my feelings of the moment, may be not calm and not warm"*
- **She meant:** Don't decide the mood on her behalf, and don't default to pleasant. The feeling is whatever it was at the time, including unpleasant. This is the seed of the whole mood-prescription ban list.
- **Fix applied:** Ban list in `SKILL.md`; the About closes on *"whatever it was"* for the same reason.
- Source: memory `feedback_bilingual_prompts.md`.

### 2026-06-?? — About content — `CUT-EXPLANATION` (inverse: she added)
- **I wrote:** —
- **She said:** *"разные формы, еще до цветов: стопка книг, рукописи в свитках, человеческие профили..."*
- **She meant:** Substantive content she wanted in, written more freely in Russian than she'd have written it in English. Not a side-note to skim.
- **Fix applied:** Treated as About material. Standing lesson: Russian asides are content, not commentary.
- Source: memory `feedback_bilingual_prompts.md`.
