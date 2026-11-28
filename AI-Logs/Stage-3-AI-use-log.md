# How We Used AI in Stage 3 (Final Markdown & Repository)

**Module:** Hardware, Networks & Behaviourist AI
**Team:** Fengwei Fried Rice
**Date:** 28 November 2026
**Tool:** Claude Code (Claude Opus 5.5), run by 林元甲 (Bvmhe, CIO) on the `stage3-final` branch

This log records the AI assistance used to finalise the shared Markdown content and organise the repository for Stage 3. AI help on the Study Guide, Test Bank and games is recorded in the Improvement Summary in `Stage3-Final/`.

## 1. What the AI did, and how we checked it

| Task | What AI did | How it was checked | Evidence |
|---|---|---|---|
| Gap review | Read every chapter, the README, the Stage 3 draft ZIP and the slides, and listed gaps against the Stage 3 instructions (placeholders, version labels, duplicates, broken links, missing PDF/report). | 林元甲 read the gap list and approved which items to fix before any file was changed. | Gap list in the Final Submission Report |
| Real-world photos | Took the 15 Wikimedia Commons photos already used in the Final Study Guide out of the guide PDF and placed them in the chapters with captions and credit lines. | Every photo was viewed on a contact sheet to confirm it is the right object; licences and authors were copied from `Stage3-Final/Photo_Credits.txt`, which the team had checked on each Commons page. | Commit `da16bac` |
| Alt text and captions | Replaced four "Alt text describing the image" placeholders and added captions to Figures 4.1 and 5.1. | Each description was compared with the actual image. | Commit `da16bac` |
| Module Overview | Drafted `Main/00-Module-Overview.md` using the learning goals and prerequisites from the Final Study Guide. | Wording compared with guide page 4–5 and the chapter headers; the old outline's different outcomes were not kept, to avoid a contradiction. | Commit `32ea010` |
| PDF copy | Wrote `tools/build_pdf.py` and built the 219-page PDF. | Searched the PDF text for un-rendered diagram code, error messages and alert markers (none); checked sample pages by eye. | `Main/FengweiFriedRice_Stage3_FinalMarkdown.pdf` |
| Consistency check | Compared 14 worked answers in the slide speaker notes with the Markdown; mapped Test Bank tags, slides and videos to sections. | All 14 answers agree. Unclear video ticks were replaced with the actual video names after reading the guide's video list. | `docs/resource-mapping.md` |
| Repository tidy-up | Proposed and made the moves/removals listed in commit `918bcd0`. | Each removed file is either empty, a build cache, or an older duplicate of a file that remains; all stay in Git history. | Commit `918bcd0` |

## 2. What AI got wrong and how it was fixed

- **Photo placement:** two photos were first inserted between a sentence ending in ":" and the formula it introduced (LiDAR and ultrasonic, Module 04). They were moved below the formulas before committing.
- **Video mapping:** the first draft of the resource mapping marked some sections with ✔ for videos without checking which video covers which section. These were replaced with the specific video titles taken from the guide's video list.
- **Module_V2.md:** the AI first described `Main/Module_V2.md` as a reformatted copy of the chapters. A paragraph-by-paragraph comparison then showed about 40 paragraphs only in V2. The Git history explains them: V2 was assembled on 4 Nov 23:55, and on 5 Nov 00:47 曾富良 (LimunZ) replaced the Python check programs and text diagrams in the chapter files with Mermaid flowcharts. The chapters are therefore the newer version; the same numbers and references are present there as diagrams. V2 was still removed from the main path, and this reason is recorded here and in the Final Submission Report.
- **Google Images:** the AI did not use photos from Google Images because most are copyrighted; it reused the team's licence-checked Wikimedia photos instead.

## 3. Responsible use statement

AI was used as an assistant for checking, formatting and organising. It did not invent feedback, contributions or test results. Content changes were limited to images, captions, labels and the overview page; the explanations, worked examples and answers in the chapters were written and checked by the team in Stages 1–2 and were not rewritten by AI in Stage 3. All changes are visible as separate commits on the `stage3-final` branch and in the pull request.
