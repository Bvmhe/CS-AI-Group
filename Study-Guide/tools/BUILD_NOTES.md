# How the Canva study guide is built

**Title:** Hot Wok, Smart Chips — We're Coooking! (Team Fengwei Fried Rice, Stage 2 V1.0)

## Canva designs
- **Template library** (AI-generated layouts, do not edit): `DAHZ9JICGMo` — 12 pages. Layout pages used:
  1 cover · 2 team · 4 five-step · 5 big+side card · 6 score grid · 7 four tall cards · 8 red recipe opener ·
  9 three cards + band · 10 six-box activity · 11 two big cards · 12 four cards
- **Study guide** (the real one): `DAHZ9Mj3rIs` — edit link https://www.canva.com/d/DgMiAqa6VqSV09l
  Pages are copies of the template pages (merge-designs), then text and pictures are swapped in.

## Pipeline
1. `content/*.md` — page script (the words), written from `Main/Chapter-0X/*.md`.
2. `tools/build_assets.py` — picks/resizes pictures from `Assets_temp/` into `assets/hosts` and builds
   Recipe Book collages in `assets/book` (288 dishes on 12 pages). `assets/asset_sources.json` lists every source file.
3. Pictures are uploaded to Canva (upload URL + POST); media ids are logged in the scratch `uploads.jsonl`.
4. `tools/specs_partN.py` — per-page spec (template slots → text, layout, pictures).
5. `tools/canva_fill.py` — reads the design JSON (read-design with an open transaction) and writes one list of
   edit operations per page. Shape resizes must come last in a page's operation list (Canva quirk).

## Page map (planned)
| Pages | Part |
|---|---|
| 1–7 | Cover, team, menu, how to play, goals, score card, cast |
| 8–25 | Recipe 01 · The Kitchen Crew (AI hardware) |
| 26–43 | Recipe 02 · Order Up! (networks) |
| 44–61 | Recipe 03 · Training the Kitchen Pup (behaviourist AI / RL) |
| 62–77 | Recipe 04 · Chef's Senses (sensors) |
| 78–93 | Recipe 05 · Running the Restaurant Chain (integration & optimisation) |
| 94–103 | Mama's Pantry (glossary), Word Soup, videos, references, credits |
| 104+ | Recipe Book (unlockable dish collages) |
