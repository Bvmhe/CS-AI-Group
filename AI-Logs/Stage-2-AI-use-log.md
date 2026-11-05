# How We Used AI in Stage 2

**Module:** Hardware, Networks & Behaviourist AI
**Team:** Fengwei Fried Rice

This document records how our team used AI tools in Stage 2, while building the study guide *Hot Wok, Smart Chips* (`Study-Guide/`) and the separate Test Bank (`Test-Bank/`). The main tool was **Claude Code** (an AI coding assistant), plus the Canva connector it used. Team members chose the topics, theme and structure, reviewed the results and made the final decisions.

> **To complete before submission:** replace every `[ ... ]` with your own details (who reviewed what, when, and a screenshot or commit link as evidence).

## Table of Contents

1. [Drafting and Checking Content](#1-drafting-and-checking-content)
2. [Page Design and Canva Production](#2-page-design-and-canva-production)
3. [Kitchen Labs and Expected Results](#3-kitchen-labs-and-expected-results)
4. [Test Bank](#4-test-bank)
5. [Videos and Sources](#5-videos-and-sources)
6. [Exports, Files and GitHub](#6-exports-files-and-github)
7. [What AI Got Wrong and How We Fixed It](#7-what-ai-got-wrong-and-how-we-fixed-it)
8. [Responsible Use Statement](#8-responsible-use-statement)

---

## 1. Drafting and Checking Content

**What we did**
We turned our Stage 1 Markdown notes on the five topics (AI hardware, networks, reinforcement learning, sensors, system integration) into 319 guide pages written in a Cooking Mama "recipe" style.

**How AI helped**
- Rewrote technical notes in simpler language and suggested worked examples (e.g. Amdahl's law, ridge point, all-reduce, stopping distance)
- Recalculated the numbers in the worked examples with Python so the answers match
- Suggested diagrams for each chapter (the "See it" pages)

**Our role**
[Name(s)] checked every chapter for correctness and level: [what was changed or removed].

## 2. Page Design and Canva Production

**What we did**
We designed the Cooking Mama theme (colours, food icons, "Mama's Feedback" boxes) and put the guide into four Canva designs.

**How AI helped**
- Wrote Python scripts (`Study-Guide/tools/`) that render page backgrounds and work out where text boxes go
- Uploaded the backgrounds and pictures and created **editable** text and photo boxes in Canva, so we can still change every page by hand
- Fixed text that overflowed its box after Canva's font turned out wider

**Our role**
We chose the theme and layout and approved each Part before export. Cooking Mama images © Office Create, credited in the guide and used for class only.

## 3. Kitchen Labs and Expected Results

**What we did**
Five hands-on labs (Colab CPU vs GPU, ping/tracert, GridWorld, phyphox, JAX jit), each with an offline version.

**How AI helped**
- Drafted the lab steps and the "⭐ Mama's Feedback (expected result)" windows, with typical numbers (e.g. T4 GPU 50-100× faster than CPU at size 4,096)

**Our role**
[Name(s)] ran the labs on [date] and confirmed the expected results: [evidence / screenshot].

## 4. Test Bank

**What we did**
A separate editable Word Test Bank (with PDF copy) following the section F rules.

**How AI helped**
- Generated the Word file with a script (`Study-Guide/tools/make_testbank.py`): 25 True/False, 25 multiple-choice (21 choose-one, 4 select-all) and 10 open questions, 5 + 5 + 2 per recipe
- Wrote lab-linked questions (interpret an experiment, explain a failed plan, predict a program, justify a design) with all data inside the question
- Checked that every key term appears in our guide, and recalculated every numeric answer in Python. This check found one wrong answer letter, which was fixed
- Spread the correct answers over A-D

**Our role**
[Name] reviewed all 60 questions and the answer key on [date]: [changes made].

## 5. Videos and Sources

**How AI helped**
- Checked that all 17 linked YouTube videos are still online (automatic check of each link)
- Wrote the "why watch / timestamps / task / limitations" notes for each video

**Our role**
[Name(s)] watched the videos and confirmed the timestamps and notes.

## 6. Exports, Files and GitHub

**How AI helped**
- Exported the Canva designs to one 319-page PDF, made a smaller phone (WeChat) PDF, and built `FengweiFriedRice_Stage2_StudyGuide_TestBank_V1.zip`
- Organised the repository: `Study-Guide/` and `Test-Bank/` as separate folders, committed and pushed with our approval

## 7. What AI Got Wrong and How We Fixed It

| Problem | Fix |
|---------|-----|
| The first phone PDF showed only page backgrounds (text and pictures missing) | Re-made it by rendering each full page; checked pages visually |
| Text in Canva overflowed some boxes | Box sizes recalculated for Canva's wider font |
| One multiple-choice answer key pointed to the wrong letter | Found by recalculating all answers; corrected |
| The ZIP was first named without `_V1`, against the section J rule | Renamed |

## 8. Responsible Use Statement

AI helped us draft, calculate, check and produce files, but the topics, the theme, the final wording and every decision are ours. We checked AI output against our notes and sources and corrected the errors listed above. Third-party images are credited and used for class only. [All members: confirm by adding your names.]
