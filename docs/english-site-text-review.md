# English site texts: changes for approval

16 November 2026. Compared the approved Russian page copy in `texts/ru/` and `sites/sveta/site.json` with the current connected English [home document](https://docs.google.com/document/d/1OEtkUDGZngCjAB2Fj6IxBbSSEFQTcQlJGDCrwew1Ck8/edit), [About document](https://docs.google.com/document/d/1Mx-lCcGQBsVbGsw0c5S-V5pA3z6mKg46TzPtxsNcRA0/edit), and English short strings in `sites/sveta/site.json`.

Review complete: eight English site-text changes approved, and the existing homepage headline explicitly retained. Publication authorized by Sveta: “go publish”. The five approved long-text replacements were applied to the live English home Google Doc; the remaining three short strings and the years-practising metadata are included in the site publication. Tests were not run, as requested. The seven approved painting-description edits were already written to the Sheet; those are recorded in `english-description-review.md`. The decisions to retain Vancouver and Under Vancouver Sky in English still stand.

## Meaning and fact changes

### 1. Home headline

Russian: «Будто снова там.» English conveys presence but omits returning.

Current:

> The feeling of being there.

Proposed:

> The feeling of being there again.

Decision: retain “The feeling of being there.” Sveta: “leave as it is”. The proposed addition of “again” is declined.

### 2. Text below the headline

Russian now describes familiar places, changing light and what stays in memory. English still mentions ordinary beauty and promises a quiet moment in the buyer's home. That is a different message, and quietness was deliberately removed in Russian.

Current:

> Paintings of familiar places, fleeting light, and ordinary beauty.
> Acrylic paintings to bring a quiet moment into your home.

Proposed:

> Paintings of familiar places, fleeting light, and ordinary beauty.
> Acrylic paintings of what stays with me.

Revised by Sveta: “Paintings of familiar places, fleeting light, and ordinary beauty. Acrylic paintings of what stays with me.” Approved: “approve. next”. Publication authorized: “go publish”.

### 3. Studio introduction: experience and location

Russian: about ten years painting in acrylic, rather than eight; the sentence about every piece being one-of-a-kind was removed.

Current:

> I paint in acrylic. I work from Thornhill, Ontario, Canada. I've been painting for 8 years. Each piece is one-of-a-kind.

Proposed:

> I've been painting in acrylic for about ten years. My studio is in Thornhill, Ontario, Canada.

`site.artist.yearsPracticing` is updated from 8 to 10; the visible text retains “about ten years” and no exact start date is inferred.

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

### 4. On painting: first paragraph

Russian explicitly describes Sveta working out the light and air, rather than the canvas finding its own weather. This is a change in the explanation of the process, not just sentence structure.

Current:

> I paint one piece at a time. Each canvas sits in the studio for weeks while it finds its weather.

Proposed:

> I work on one painting at a time. A canvas can stay in the studio for weeks while I work out the light and atmosphere.

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

### 5. On painting: materials and starting points

Russian deliberately removes cotton/linen and self-priming claims, and describes how a painting can take an unexpected direction. The English still contains the material and preparation claims. Their deletion does not establish that they are false; it establishes that they were removed from the approved copy.

Current:

> Acrylic on cotton canvas or linen, primed in the studio. I work from references — quick sketches, light I notice on walks — but the painting itself decides what it wants to be.

Proposed:

> I paint in acrylic on canvas, primed in the studio. I work from references — quick sketches, light I notice on walks — but the painting itself decides what it wants to be.

Revised by Sveta to retain “primed in the studio” and the existing account of references and the painting deciding its direction. Approved: “approve. next”. Publication authorized: “go publish”. This explicit English choice takes precedence over the earlier proposal to align every detail with Russian.

### 6. Footer tagline

The central message changed. Russian is about noticing and remembering; English is about slow production and original paintings.

Current:

> Original paintings —
> made slowly, one at a time.

Proposed:

> About the things I notice and don't want to forget.

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

## Deliberately removed claims: optional alignment

These do not establish factual contradictions, but the Russian review intentionally removed the repeated originality/reproduction wording. To mirror that editorial choice in English, make the following deletions. Each requires approval.

### 7. On painting: final paragraph

Keep the first sentence and remove the second:

> Each finished piece is signed on the verso, photographed in daylight, and shipped rolled or stretched depending on size.

Delete: “Nothing is reproduced — every painting is the only one of its kind.”

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

### 8. Painting-list introduction

The availability, private collections and commission offer already match. For consistency with the deletion of the originality sentence:

> Available works and pieces already in private collections — commission requests welcome.

Delete the current opening: “Every painting one-of-a-kind.”

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

### 9. Footer bottom line

Shipping worldwide is already consistent. To reflect the Russian removal of the originality claim:

> I ship paintings worldwide.

Replaces: “All works one-of-a-kind. Shipped worldwide.”

Approved by Sveta: “approve. next”. Publication authorized: “go publish”.

## Differences that do not require an English rewrite

- The memory paragraph in the studio introduction has the same meaning.
- About, paragraph 1: “doodled” fits «калякала»; the sense that learning was for other people is preserved. «Почему-то» adds a small shade of uncertainty, not a different account of events.
- About, paragraph 2: the tree's own shadow, outer leaves catching light, sunset illumination and mixing colours as witchcraft are all present in English. Russian makes “always” and “only” explicit, but an English rewrite is not necessary to preserve the observation.
- About, paragraph 3: the walks, evening light, city wandering and wish to preserve what was felt remain the same. Russian's shorter ending does not restrict the emotion to something pleasant.
- Shipping note, reply time and contact invitation describe the same service. First-person Russian wording does not require English changes.
- Contact heading still offers contact and studio visits. Appointment arrangements remain consistent.
- Navigation and contact buttons have the same destination and purpose; “About” and “Write to the studio” remain natural English even though Russian uses first person.
- FAQ was explicitly retained unchanged; do not reopen its translation review.
- Collection names, sizes and remaining interface labels were explicitly retained.

When publishing approved page changes, update the live English Google Docs and matching `texts/` backups. Edit artist-specific English short strings in `sites/sveta/site.json` and regenerate the root `index.html`. Leave shared defaults for other artists untouched.
