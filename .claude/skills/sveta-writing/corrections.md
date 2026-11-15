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
| `NATURAL-RUSSIAN` | 1 | observation | Rewrite Russian idiomatically; preserve meaning and the author's original Russian rather than English syntax. |
| `AUTHOR-REVISION` | 5 | **rule** | Preserve the author's factual updates, wording corrections, and explicit deletions verbatim. |

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

### 2026-11-14 — Russian site copy — `NATURAL-RUSSIAN`
- **I wrote:** Existing published translations included «Удержанное лето», «бледная вода и небо за ними почти ничего не делают», and «спилы плавника» (reviewed here; authorship of the earlier translations is not established).
- **She said:** "review texts, titles, descriptions translations to russian for sveta-colours.com, I need it sound russian, not strignt translation"
- **She meant:** Russian should read as original Russian prose, preserving the subject and voice without copying English constructions. Good existing Russian titles need not be renamed.
- **Fix applied:** Prepared `docs/russian-copy-review.md` with proposed page copy, all 22 painting descriptions, selective title changes, FAQ and interface wording. Restored her original Russian Vancouver wording in the proposal. No live content changed.

### 2026-11-14 — Russian home introduction — `AUTHOR-REVISION`
- **I wrote:** «Пишу акрилом уже восемь лет. Моя мастерская — в Торнхилле, в канадской провинции Онтарио. Каждая картина — единственная.»
- **She said:** «Пишу акрилом уже восемь лет -> Пишу акрилом уже лет десять; Каждая картина — единственная - убери вообще»
- **She meant:** Use her approximate duration, «уже лет десять», and delete the uniqueness sentence from this paragraph without replacing it.
- **Fix applied:** Updated the paragraph in `docs/russian-copy-review.md` and recorded her exact revision. Live content remains unchanged during the sequential review.

### 2026-11-14 — Russian on-painting paragraph — `AUTHOR-REVISION`
- **I wrote:** «Я работаю над одной картиной за раз. Она может неделями стоять в мастерской, пока я ищу, каким будет в ней свет, каким — воздух.»
- **She said:** «не заметила сразу: Я работаю над одной картиной за раз. Она может неделями стоять в мастерской, пока я ищу, каким будет в ней свет, каким — воздух. -> ... каким будет в ней свет, а каким — воздух»
- **She meant:** Add «а» before «каким — воздух» in the previously approved paragraph.
- **Fix applied:** Updated the approved draft verbatim. The following paragraph remains pending; no live content changed.

### 2026-11-14 — Russian materials and starting points — `AUTHOR-REVISION`
- **I wrote:** «Пишу акрилом на хлопковом или льняном холсте, грунтую сама. Отталкиваюсь от быстрых набросков и того, что замечаю на прогулках.»
- **She said:** «Пишу акрилом на хлопковом или льняном холсте, грунтую сама -> Пишу акрилом на холсте;  В основе обычно быстрые наброски и то, что я замечаю на прогулках»
- **She meant:** Remove the canvas-fibre and self-priming claims; use her wording for sketches and observations. This deletion does not establish which materials or preparation she actually uses.
- **Fix applied:** Replaced the first two sentences in the review draft; kept the last sentence pending approval. Promoted AUTHOR-REVISION after three entries on 2026-11-14. No live content changed.

### 2026-11-14 — Russian About opening — `AUTHOR-REVISION`
- **I wrote:** «Я всегда что-нибудь рисовала на любой бумажке»; «Но долго думала, что научиться рисовать».
- **She said:** «Я всегда что-нибудь рисовала на любой бумажке -> Я всегда чего-нибудь калякала на любой бумажке; Но долго думала, что научиться рисовать -> Но мне почему-то всегда казалось, что учиться рисовать»
- **She meant:** Keep her colloquial «чего-нибудь калякала» and the distinction between an unexplained impression about learning and a considered belief about attaining ability.
- **Fix applied:** Applied both replacements verbatim in the review draft, retaining the rest of the paragraph. Awaiting approval of the combined paragraph; no live content changed.

### 2026-11-14 — Russian About light and shadow — `AUTHOR-REVISION`
- **I wrote:** «как внутри кроны прячется тень, как свет цепляется за листья по краям, как заходящее солнце подсвечивает дерево снизу».
- **She said:**
  > как внутри кроны прячется тень, как свет цепляется за листья
  >   по краям, -> например, что внутри дерева всегда его собственная тень, листья освещены только снаружи, а заходящее солнце подсвечивает дерево снизу
- **She meant:** Replace the figurative phrasing with her direct observations about the tree's own shadow, outside illumination of leaves, and sunset lighting from below.
- **Fix applied:** Used her replacement once, including its final sunset clause, in the review draft. The combined paragraph awaits approval; no live content changed.

### 2026-11-14 — Held Summer description — `AUTHOR-REVISION`
- **I wrote:** «Жёлтые рудбекии, два вида лиловых астр и белая ромашка, для которой лето ещё не кончилось. За ними уже осенние краски стены, а букет всё ещё летний. Наверное, я так и держусь за последние тёплые дни.»
- **She said:** «Теплое разноцвеетье букета, для которого лето еще не кончилось. За ним уже осенние краски стены, а цветы все еще летние. Наверное, я так держусь за последние тёплые дни.»
- **She meant:** Center the description on the bouquet's warm variety; refer to the bouquet and then the flowers; retain her simpler final wording.
- **Fix applied:** Replaced the description in `docs/russian-copy-review.md` with her wording, normalizing «разноцвеетье» to «разноцветье» and ё. Awaiting approval; no live content changed.

### 2026-11-14 — Holding On In Winter description — `AUTHOR-REVISION`
- **I wrote:** «розовые детские куртки»; «Где-то позади — Си-Эн Тауэр»; «а они всё идут, вместе».
- **She said:** «только розовые детские куртки -> только розовые детские курточки; Где-то позади — Си-Эн Тауэр -> Где-то впереди — Си-Эн Тауэр; , а они всё идут, вместе -> а они идут, все вместе, сквозь этот зимний город.»
- **She meant:** Use «курточки», place the tower ahead, and describe the family walking together through the winter city.
- **Fix applied:** Applied all three corrections verbatim in the review draft. Awaiting approval; no live content changed.

### 2026-11-14 — Lost In The Fog description — `AUTHOR-REVISION`
- **I wrote:** «Туман стоит неподвижно. Свет двух фонарей едва сквозь него пробивается — два ярких пятна среди серого. Двое гуляют с собаками и почти исчезают в тумане. Не знаю, почему я снова и снова возвращаюсь к нему в своих картинах.»
- **She said:** «Туман не движется. Фонари едва пробиваются сквозь него - два ярких пятна среди серого. Двое гуляют с собаками и почти исчезают в тумане. Не знаю, почему я снова и снова возвращаюсь к туману в своих картинах.»
- **She meant:** Keep the direct phrasing, make the returning reference explicitly «к туману», and preserve the observed scene.
- **Fix applied:** Replaced the proposed description with her wording, normalizing the dash. Awaiting approval; no live content changed.

### 2026-11-14 — Noise And Silence neutral strip — `AUTHOR-REVISION`
- **I wrote:** «А деревья стоят посередине, никого не выбирая.»
- **She said:** «предложи три варианта улучшеного перевода. Деревья вроде нейтральной полосы между шумом и тишиной»
- **She meant:** The trees form a neutral strip between the noisy road and the quiet path. Make that spatial relationship central to three alternative Russian descriptions.
- **Fix applied:** Added three alternatives to the review draft for selection; none is approved yet.

### 2026-11-14 — Noise And Silence author wording — `AUTHOR-REVISION`
- **I wrote:** «Слева — дорога с машинами»; «всего десять метров»; «А между ними — ряд деревьев, словно нейтральная полоса».
- **She said:** «Слева — магистраль с машинами и свет в окнах. Справа двое гуляют с собаками по заснеженной тропинке под фонарём. Одна улица, один и тот же час, всего несколько метров между шумом и тишиной. И ряд деревьев между ними, словно нейтральная полоса.»
- **She meant:** Use «магистраль», avoid an exact distance, and retain her final sentence about the neutral strip.
- **Fix applied:** Replaced the draft and alternatives with her full wording. Awaiting approval.

### 2026-11-14 — Snowfall description — `AUTHOR-REVISION`
- **I wrote:** «В таком снегопаде будто можно простить почти всё.»
- **She said:** «простить -> забыть»
- **She meant:** Use «забыть» instead of «простить».
- **Fix applied:** Updated the proposed Snowfall description accordingly. Awaiting approval.

### 2026-11-14 — Summer Hut flowers — `AUTHOR-REVISION`
- **I wrote:** «У двери теснятся жёлтые лилейники, лиловые ирисы, красные цветы.»
- **She said:** «У двери теснятся яркие цветы»
- **She meant:** Replace the list of flower types and colours with her simpler sentence.
- **Fix applied:** Updated the Summer Hut review draft. Awaiting approval of the full description.

### 2026-11-14 — Waiting For Rain cup — `AUTHOR-REVISION`
- **I wrote:** «и коричневая чашка».
- **She said:** «и коричневая чашка -> и чашка кофе»
- **She meant:** Name the coffee rather than the cup's colour.
- **Fix applied:** Updated the Waiting For Rain review draft. Awaiting approval of the full description.

### 2026-11-14 — Walking Home figures and snowfall — `AUTHOR-REVISION`
- **I wrote:** «Отец ведёт ребёнка»; «они идут от меня».
- **She said:** «Взрослый ведет ребенка за руку по заснеженному тротуару. Красная куртка сразу притягивает взгляд: ни башня с часами, ни провода, ни серое небо с ней не спорят. Больше красного здесь нет. Я смотрю им вслед — они уходят в метель.»
- **She meant:** Avoid identifying the adult as the father; use her ending about disappearing into the snowstorm.
- **Fix applied:** Replaced the review description with her wording, normalizing ё and removing pasted quotation formatting. Awaiting approval.

### 2026-11-14 — Inner Spring setting — `AUTHOR-REVISION`
- **I wrote:** «На деревянном столе»; «На стене за ними угадываются другие ветви»; «по эту сторону стены».
- **She said:** «На подоконнике стеклянная банка с голыми вишнёвыми ветками и белыми цветами. За окном за ними дождь, темнеют ветки деревьев и весна ещё не пришла. но здесь, по эту сторону окна, кажется, уже началась.»
- **She meant:** The jar is on a windowsill; rain and dark branches are outside the window. The contrast is between spring indoors and its absence outdoors.
- **Fix applied:** Used her wording in the review draft, adding a dash after «подоконнике» and capitalizing «Но». Awaiting approval.

### 2026-11-14 — Inner Spring full author rewrite — `AUTHOR-REVISION`
- **I wrote:** «На подоконнике — стеклянная банка с голыми вишнёвыми ветками и белыми цветами. За окном за ними дождь, темнеют ветки деревьев и весна ещё не пришла. Но здесь, по эту сторону окна, кажется, уже началась.»
- **She said:** «На подоконнике стеклянная банка, в ней зацвели вишнёвые веточки. За окном дождь, темнеют ветки деревьев и до весны еще далеко. Но здесь, по эту сторону окна, уже весна и белые цветы.»
- **She meant:** Use her complete rewrite: flowering cherry twigs indoors, spring still far away outdoors, and the direct ending about spring and white flowers.
- **Fix applied:** Replaced the description verbatim, normalizing «еще» to «ещё». Awaiting approval.

### 2026-11-14 — Under Vancouver Sky clouds — `AUTHOR-REVISION`
- **I wrote:** «За водой — городские башни под затянутым небом.»
- **She said:** «За водой — городские башни под затянутым небом -> На другом берегу над городскими башнями нависли тучи»
- **She meant:** Use her more direct description of clouds hanging over towers on the opposite shore.
- **Fix applied:** Replaced the sentence verbatim in the review draft. Awaiting approval.

### 2026-11-14 — Under Vancouver Sky island rocks — `AUTHOR-REVISION`
- **I wrote:** «пробилось немного солнца и осветило камни».
- **She said:** «пробилось немного солнца и осветило камни островка»
- **She meant:** Identify the rocks as belonging to the small island.
- **Fix applied:** Added «островка» to the first sentence in the review draft. Awaiting approval.

### 2026-11-14 — Studio visit note — `AUTHOR-REVISION`
- **I wrote:** «О встрече в мастерской договоримся заранее.»
- **She said:** «Визит в мастерскую по договорённости.»
- **She meant:** Retain the existing studio visit note.
- **Fix applied:** Restored her selected wording in the review document.

### 2026-11-14 — Recent navigation label — `AUTHOR-REVISION`
- **I wrote:** «Новые работы» for the Recent menu item.
- **She said:** «нет, оставь Новые»
- **She meant:** Retain «Новые» in the menu, overriding the preceding approval of «Новые работы».
- **Fix applied:** Updated the review decision to retain «Новые». The About menu label remains pending.

### 2026-11-14 — Remaining interface copy — `AUTHOR-REVISION`
- **I wrote:** Proposed further revisions starting with «Главная картина» → «Картина на главной странице».
- **She said:** «остальное оставь как есть»
- **She meant:** Finish the sequential review and preserve all remaining unapproved interface text. Previously approved changes still stand.
- **Fix applied:** Removed remaining proposed interface replacements, retained the existing replyWhen text, and marked the review complete. Approved painting headings now match the agreed titles. Publication is pending.

### 2026-11-14 — FAQ retain existing wording — `AUTHOR-REVISION`
- **I wrote:** Proposed rewritten Russian FAQ questions and answers.
- **She said:** «вопросы и ответы оставь как есть»
- **She meant:** Keep the existing Russian FAQ in full, without the proposed revisions.
- **Fix applied:** Removed the proposed FAQ rewrite from the review document and recorded that existing published wording and `texts/ru/faq.txt` must be retained.
