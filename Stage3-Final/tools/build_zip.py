"""Assemble and zip the Stage 3 Final package, then verify the ZIP from a clean extract."""
import json, os, shutil, zipfile

BASE = r'C:\Projects\CS-AI-Group\Stage3-Final'
NAME = 'FengweiFriedRice_Stage3_FinalStudyGuide_TestBank'
STAGE = os.path.join(BASE, 'ZIP-staging', NAME)
ZIP = os.path.join(BASE, NAME + '.zip')

FILES = {
    'FengweiFriedRice_Stage3_FinalStudyGuide.pdf': 'FengweiFriedRice_Stage3_FinalStudyGuide.pdf',
    'FengweiFriedRice_Stage3_FinalTestBank.docx': 'FengweiFriedRice_Stage3_FinalTestBank.docx',
    'FengweiFriedRice_Stage3_FinalTestBank.pdf': 'FengweiFriedRice_Stage3_FinalTestBank.pdf',
    'Improvement_Summary.docx': 'Improvement_Summary.docx',
    'Improvement_Summary.pdf': 'Improvement_Summary.pdf',
    r'Supplements\Mamas_Quiz_Kitchen.html': r'Supplements\Mamas_Quiz_Kitchen.html',
    r'Supplements\Pup_Pantry_Maze_offline.html': r'Supplements\Pup_Pantry_Maze_offline.html',
    r'Supplements\Fengwei_24H_Diner_Study_Guide.html': r'Supplements\Fengwei_24H_Diner_Study_Guide.html',
    r'shots\Mamas_Quiz_Kitchen_start.png': r'Supplements\screenshots\Mamas_Quiz_Kitchen_start_screen.png',
    r'shots\why_Pup_Pantry_Maze_offline.png': r'Supplements\screenshots\Pup_Pantry_Maze_why_screen.png',
    r'shots\why_diner2.png': r'Supplements\screenshots\24H_Diner_why_screen.png',
}

README = """# Fengwei Fried Rice · Stage 3 Final Study Guide and Test Bank

Hardware, Networks & Behaviourist AI · Introduction to Artificial Intelligence · 28 November 2026

## What is in this ZIP

| File | What it is |
|------|------------|
| `FengweiFriedRice_Stage3_FinalStudyGuide.pdf` | **Final Study Guide** (168 pages). Open in any PDF reader; the menu, page links and web links are clickable. |
| `FengweiFriedRice_Stage3_FinalTestBank.docx` | **Final Test Bank**, editable Word (60 questions: 25 true/false, 25 multiple choice, 10 open). Questions first, "Mama's Answer Key" after. |
| `FengweiFriedRice_Stage3_FinalTestBank.pdf` | PDF copy of the Test Bank (links work). |
| `Improvement_Summary.docx` / `.pdf` | What we changed since Stage 2, why, where to find it and how we checked it, with before/after pictures. |
| `Supplements/` | The three games, their setup instructions (`Supplements/README.md`) and fallback screenshots. |
| `Photo_Credits.txt` | Source, author and licence of every real photo in the guide. |

Everything needed to study is in these files; no internet link is required for the core content.

Team repository: github.com/Bvmhe/CS-AI-Group
"""

SUPP_README = """# Supplements · Fengwei Fried Rice

Three optional games that go with the Final Study Guide. They are extra practice; the guide and the Test Bank
work without them.

| File | What it is | What learners should do |
|------|------------|-------------------------|
| `Mamas_Quiz_Kitchen.html` | The Final Test Bank as a game: the same 60 questions, with Mama's feedback after every answer. | Pick a station (one per recipe), answer, read the explanation for every wrong answer, then retry the station. |
| `Pup_Pantry_Maze_offline.html` | Bonus Kitchen Lab for Recipe 03 (reinforcement learning), see guide p145. Rule Chef, Map Chef and the Learner Pup (Q-learning) solve the same mazes. | Predict which chef wins each kitchen, run it, compare the steps and rewards, then answer the 3 debrief questions on the start screen. |
| `Fengwei_24H_Diner_Study_Guide.html` | The whole guide as a 3D review game: 5 rooms = Recipes 01–05; customers' orders are Test Bank questions. | Walk into each room after finishing that recipe, serve the orders, then take the Professor's Final Exam. |

## How to open them (setup)

1. Unzip the whole ZIP first (do not open the files from inside the ZIP).
2. Double-click an `.html` file in this folder. It opens in your browser (Edge, Chrome, Firefox or Safari; phone or computer).
3. No install and no account are needed. The game code is inside each file, so they work offline and in mainland
   China. (Pup's Pantry Maze asks Google Fonts for its rounded font; without internet it uses a normal system font.)
4. Each game opens with a start screen. Pup's Pantry Maze and the 24-Hour Diner show **"Why this game?"**: the learning
   goal, what each game part stands for and 3 debrief questions. The **❓ Why this game?** button (bottom left) opens it again.
5. The 24-Hour Diner is a 3D game (about 5 MB). On an old phone it may load slowly; a laptop works best.

## If a game will not open

The `screenshots/` folder shows what each start screen should look like:
`Mamas_Quiz_Kitchen_start_screen.png`, `Pup_Pantry_Maze_why_screen.png`, `24H_Diner_why_screen.png`.
If your browser blocks the file, try another browser. You can always use the paper Test Bank and guide p145–150 instead.

## Built by our team vs borrowed

**Built by the team:** the game code, levels and in-game text of all three games; all 60 questions and answers;
the music and sound effects (made in code); the "Why this game?" screens.

**Borrowed (with credit):**
- Cooking Mama characters and pictures © Office Create. Used for class only, not for sale.
- Fredoka and Nunito fonts (SIL Open Font License), from Google Fonts.
- The 24-Hour Diner uses the three.js 3D library (MIT License), included inside the file.
- Curated YouTube videos in the guide are linked, not copied.
"""


def credits_txt():
    c = json.load(open(os.path.join(BASE, 'photos', 'credits.json'), encoding='utf8'))
    lines = ['Real-world photos in the Final Study Guide', 'All from Wikimedia Commons, free licences (CC BY / CC BY-SA / CC0).', '']
    for k in sorted(c, key=lambda k: int(k.split('_')[0][1:])):
        v = c[k]
        lines += [f'Guide page {k.split("_")[0][1:]}: {v["title"]}',
                  f'  Author: {v["author"]}', f'  Licence: {v["licence"]}', f'  Source: {v["source"]}', '']
    lines.append('Note: photos on p13–15 (GPU, TPU, NPU) appear together on p11 with the CPU photo.')
    return '\n'.join(lines)


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(STAGE)
    for src, dst in FILES.items():
        d = os.path.join(STAGE, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(BASE, src), d)
    open(os.path.join(STAGE, 'README.md'), 'w', encoding='utf8').write(README)
    open(os.path.join(STAGE, 'Supplements', 'README.md'), 'w', encoding='utf8').write(SUPP_README)
    open(os.path.join(STAGE, 'Photo_Credits.txt'), 'w', encoding='utf8').write(credits_txt())
    with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for root, _, files in os.walk(STAGE):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.join(NAME, os.path.relpath(p, STAGE)))
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        for i in z.infolist():
            print(f'{i.file_size // 1024:>7} KB  {i.filename}')
    print('ZIP', os.path.getsize(ZIP) // 1024, 'KB')


if __name__ == '__main__':
    main()
