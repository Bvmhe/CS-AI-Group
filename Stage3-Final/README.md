# 🏁 Stage 3 · Final Study Guide and Test Bank (finalized)

**Fengwei Fried Rice** · Hardware, Networks & Behaviourist AI · Introduction to Artificial Intelligence · due 28 November 2026

This folder holds the **finalized Stage 3 package**: the Final Study Guide, the Final Test Bank, the three interactive games and the Improvement Summary. It is the same content as the submitted ZIP `FengweiFriedRice_Stage3_FinalStudyGuide_TestBank.zip`, unzipped so every file can be opened on GitHub.

## 📦 What is here

| File | What it is |
|------|------------|
| [`FengweiFriedRice_Stage3_FinalStudyGuide.pdf`](FengweiFriedRice_Stage3_FinalStudyGuide.pdf) | **Final Study Guide**, 168 pages, Recipes 01–05. Menu, page links and web links are clickable. |
| [`FengweiFriedRice_Stage3_FinalTestBank.docx`](FengweiFriedRice_Stage3_FinalTestBank.docx) / [`.pdf`](FengweiFriedRice_Stage3_FinalTestBank.pdf) | **Final Test Bank**, paper edition: 60 questions (25 true/false, 25 multiple choice, 10 open) with answer key. |
| [`Improvement_Summary.docx`](Improvement_Summary.docx) / [`.pdf`](Improvement_Summary.pdf) | Every change since Stage 2: the feedback or problem, what we changed and why, and how we checked it, with before/after pictures. |
| [`Interactive/`](Interactive/) | The games: `Mamas_Quiz_Kitchen.html` (Test Bank game), `Pup_Pantry_Maze_offline.html` (Recipe 03 lab), `Fengwei_24H_Diner_Study_Guide.html` (whole guide as a 3D review game). Download and open in any browser; no install or internet needed. |
| [`Photo_Credits.txt`](Photo_Credits.txt) | Source, author and licence of every real photo in the guide. |
| [`PACKAGE_README.md`](PACKAGE_README.md) | The README inside the submitted ZIP (setup steps, built vs borrowed). |
| [`tools/`](tools/) | Python scripts that build the Final files from the Stage 2 versions. |

## 🍳 What changed from Stage 2 (teacher feedback)

| Teacher feedback | What we did |
|------------------|-------------|
| “Add images when appropriate, for instance a CPU image is expected when talking about it.” | **15 real photos** (Wikimedia Commons, CC BY / CC BY-SA / CC0) on the hardware, network, sensor and robot pages: CPU (p12), the four chips (p11), RAM, iPhone A11 board, data centre, 5G mast, ANYmal robot dog, car camera, LiDAR, parking sensor and radar, Tesla, TPU pod. Each has a caption that links it to the idea on the page, and all credits are on p167. |
| “I don't get the kitchen game's relevancy to the learning.” | New **“Why this game?”** box on guide p145, and a **“Why this game?” start screen** in Pup's Pantry Maze and the 24-Hour Diner. Each shows the learning goal, what each game part stands for (e.g. the Learner Pup = the Q-learning agent of 3.3, treats = rewards of 3.2, Cookie Jar = reward hacking), what to notice, and 3 debrief questions. |
| “The interactive test bank is well designed.” | Kept. Only relabelled “Final Test Bank”; all 60 questions in the game still match the paper version. |

**Our own fixes:** Q49 section tag §5.4 → §5.1 (MFU is taught in 5.1); “Stage 2” / “V1.0” labels replaced with “Final” on covers, footers and game titles; page numbers and tap-to-jump links for Dessert (p151) and Bonus Kitchen Lab (p145) on the menu; p4 box renamed “Prerequisites”; “Can't watch?” notes on the video pages for classmates where YouTube is blocked; all 21 web links and 16 YouTube links re-checked.

The full table with how each change was checked is in [`Improvement_Summary.pdf`](Improvement_Summary.pdf).

## ✅ Status

- [x] Final Study Guide, Final Test Bank, Improvement Summary, games: done
- [x] Submission ZIP built and tested from a clean unzip (all PDFs, Word files and games open)
- [ ] **Team to fill in guide p150:** names and roles, and how many testers tried the guide

## Built vs borrowed

**Built by the team:** all guide text, diagrams, examples and exercises; all Test Bank questions and answers; the code, levels and text of all three games; the music and sound effects (made in code).
**Borrowed, with credit:** Cooking Mama characters and pictures © Office Create (class use only, not for sale); photos from Wikimedia Commons (see `Photo_Credits.txt`); Fredoka and Nunito fonts (SIL Open Font License); three.js (MIT License) inside the 24-Hour Diner.

Stage 2 versions stay where they were: [`Study-Guide/`](../Study-Guide/), [`Test-Bank/`](../Test-Bank/) and [`Diner-Game/`](../Diner-Game/).
