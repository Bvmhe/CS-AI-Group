"""Write Improvement_Summary.docx in the planner layout: rows grouped by Study Guide / Test Bank / Supplements,
columns Feedback or problem -> What changed and why -> Where to find it and how it was checked,
with before/after evidence directly below the most important rows."""
import os
import pymupdf
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = r'C:\Projects\CS-AI-Group\Stage3-Final'
V1 = BASE + r'\V1-from-GitHub\Fengwei_Study_Guide_HD_FULL.pdf'
FINAL = BASE + r'\FengweiFriedRice_Stage3_FinalStudyGuide.pdf'
IMG = BASE + r'\summary_img'
SHOTS = BASE + r'\shots'
OUT = BASE + r'\Improvement_Summary.docx'
FONT = BASE + r'\tools\FredokaBold.ttf'
PINK = RGBColor(0xD2, 0x38, 0x6F)


def pair(pno, name):
    """Side-by-side BEFORE | AFTER of one guide page (1-based)."""
    ims = []
    for path in (V1, FINAL):
        page = pymupdf.open(path)[pno - 1]
        pix = page.get_pixmap(dpi=100)
        ims.append(Image.frombytes('RGB', (pix.width, pix.height), pix.samples))
    w, h = ims[0].size
    gap, top = 24, 46
    out = Image.new('RGB', (w * 2 + gap, h + top), 'white')
    d = ImageDraw.Draw(out)
    f = ImageFont.truetype(FONT, 30)
    for i, (im, lab) in enumerate(zip(ims, ('BEFORE (V1.0)', 'AFTER (Final)'))):
        out.paste(im, (i * (w + gap), top))
        d.text((i * (w + gap) + 6, 6), lab, font=f, fill=(120, 120, 120) if i == 0 else (210, 56, 111))
    path = os.path.join(IMG, name)
    out.save(path)
    return path


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement('w:shd'); s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto'); s.set(qn('w:fill'), hexcolor)
    tcPr.append(s)


# (group, feedback or problem, what changed and why, where to find it and how it was checked, evidence, teacher row)
ROWS = [
    ('Study Guide',
     'Teacher: “Add images when appropriate, for instance a CPU image is expected when talking about it.”',
     'Added 15 real photos (free licences, Wikimedia Commons) where the guide talks about real hardware, so learners '
     'can connect each idea to the real object: CPU, the four chips together, RAM, iPhone A11 board, data centre, '
     '5G mast, ANYmal robot dog, car camera, LiDAR, parking sensor and radar, Tesla, TPU pod. Each photo has a '
     'one-line caption that links it back to the idea on the page, plus a credit line.',
     'Where: guide p11, p12, p25, p29, p38, p57, p83, p93, p94, p95, p111, p138; all credits on p167 and in '
     'Photo_Credits.txt.\nChecked: licence and author read from each file\'s Commons page (CC BY, CC BY-SA or CC0 '
     'only); every changed page rendered and compared with V1.0 (below) to make sure nothing covers the text.',
     [('pair', 12, 'p012.png'), ('pair', 95, 'p095.png')], True),
    ('Study Guide',
     'Teacher: “I don\'t get the kitchen game\'s relevancy to the learning.”',
     'New “Why this game? · Learning Goal 3” box. It says the maze is Recipe 03 made playable, maps each game part '
     'to a section (agent and Q-learning 3.3, rewards 3.2, reward hacking 3.3) and gives 3 debrief questions, so the '
     'learner knows what to notice before playing.',
     'Where: guide p145 (Bonus Kitchen Lab).\nChecked: every game term in the box (Learner Pup, Map Chef, Cookie Jar, '
     'α γ ε) compared with the game code and with sections 3.2–3.3.',
     [('pair', 145, 'p145.png')], True),
    ('Study Guide',
     'Assignment: the final guide must have no placeholders. The Bonus Kitchen Arcade page still had '
     '“[TEAM: add names and roles]” and “[TEAM: add how many testers…]”.',
     'Filled the “Team contribution” box with real details: Pup\'s Pantry Maze and the Test Bank questions built by '
     '曾富良, all 60 questions checked by 黄施杰 (Darren, Quality Lead), every member\'s role, and the classmate '
     'try-out (2 classmates from team DOOM). Removed the “(fill in before submitting)” note.',
     'Where: guide p150.\nChecked: names and roles match the team roster in our GitHub README; a text search of the '
     'whole Final PDF finds no “[TEAM” placeholder left.',
     [], False),
    ('Study Guide',
     'Own check: the guide still said “Stage 2” / “V1.0”; the final guide must show final version information.',
     'Cover now says “FINAL STUDY GUIDE”, Version: Final (Stage 3), 28 November 2026; the back page footer says '
     '“Final Study Guide”.',
     'Where: guide p1 and p168.\nChecked: text search of the Final PDF for “Stage 2” and “V1.0”: none left on the '
     'cover or in footers.',
     [('pair', 1, 'cover.png')], False),
    ('Study Guide',
     'Own check: the menu (p5) gave page numbers for Recipes 01–05 but not for Dessert and the Bonus Kitchen Lab; '
     'p4 used “You need” where the course template asks for “Prerequisites”.',
     'Added “starts on page 151 / 145 · tap to jump” with working links, so every part of the guide can be reached '
     'from the menu; renamed the p4 box to “Prerequisites”.',
     'Where: guide p4 and p5.\nChecked: clicked every menu link in the Final PDF; each jumps to the right page.',
     [], False),
    ('Study Guide',
     'Own check: a learner reading only the PDF was never told that the whole guide is also a game (24-Hour Diner) '
     'or that the Test Bank can be played (Mama\'s Quiz Kitchen).',
     'Added a “Prefer to play?” card that names both games, where to find them in the ZIP and what each one does, '
     'so learners who get bored of reading can switch to the game version of the same content.',
     'Where: guide p3 (How to use this guide).\nChecked: both file names match the Supplements folder, and both '
     'games open offline from the extracted ZIP.',
     [], False),
    ('Study Guide',
     'Own check: classmates in mainland China cannot open YouTube.',
     'Added a “Can\'t watch?” note: use the English captions, or read the matching guide section and do its '
     '“How to read this” task instead, so no learner is stuck without the video.',
     'Where: video pages p33 and p62; resource pages p88, p116 and p142.\nChecked: all 21 web links and 16 YouTube '
     'links in the guide opened (from outside China).',
     [], False),
    ('Test Bank',
     'Teacher: “The interactive test bank is well designed.”',
     'Not redesigned: the teacher praised it, so we kept the questions, layout and game as they were. Only the '
     'labels changed to “Final Test Bank”, and the Q49 tag was fixed (next row).',
     'Where: FengweiFriedRice_Stage3_FinalTestBank.docx / .pdf and Supplements/Mamas_Quiz_Kitchen.html.\nChecked: '
     'a script compared all 60 questions and answers in the game with the paper Test Bank: all match.',
     [('img', BASE + r'\tb\cover_final.png', 9.0)], True),
    ('Test Bank',
     'Own check: Q49 (raising MFU) pointed to §5.4 (energy and cost), but MFU is taught in §5.1.',
     'Changed the tag to §5.1 so learners who get Q49 wrong are sent to the right section.',
     'Where: Test Bank Q49 (question and answer key) and the same question in Mama\'s Quiz Kitchen.\nChecked: '
     're-read §5.1 “MFU: Are the Cooks Really Cooking?” and the Q49 answer; the tag now matches. Only the section '
     'label changed: the question, options and answer are the ones Darren checked (next row).',
     [], False),
    ('Test Bank',
     'Assignment: every question and answer must be checked by a team member who did not write it.',
     'Darren (黄施杰, Quality Lead), who did not write the questions, checked all 60 questions and their answers (25 true/false, '
     '25 multiple choice, 10 open) between 28 October and 5 November 2026. Result: all correct, so no question or '
     'answer had to change.',
     'Where: the whole Test Bank (.docx / .pdf) and Mama\'s Quiz Kitchen, which uses the same 60 questions.\n'
     'Checked: by Darren (黄施杰), 28 Oct – 5 Nov 2026. The later Q49 change (above) is a section label only.',
     [], False),
    ('Supplements',
     'Teacher: “I don\'t get the kitchen game\'s relevancy to the learning.” (the games themselves)',
     'Pup\'s Pantry Maze and the 24-Hour Diner now open with a “Why this game?” screen: the learning goal, what each '
     'game part stands for, what to notice and 3 debrief questions, with a button to reopen it. The Diner screen '
     'shows that each room is one Recipe / Learning Goal and that customers\' orders are Test Bank questions '
     '(retrieval practice).',
     'Where: Supplements/Pup_Pantry_Maze_offline.html and Supplements/Fengwei_24H_Diner_Study_Guide.html.\n'
     'Checked: both games opened offline (file://, no internet) in Microsoft Edge: the screen appears, '
     '“Start playing” closes it, the button reopens it.',
     [('img', SHOTS + r'\why_Pup_Pantry_Maze_offline.png', 12.6), ('img', SHOTS + r'\why_diner2.png', 12.6)], True),
    ('Supplements',
     'Assignment: supplements need setup instructions, fallback screenshots, and a clear line between what the '
     'team built and what is borrowed.',
     'Added Supplements/README.md (what each game is, how to open it, what learners should do, what to do if it '
     'will not open, built vs borrowed) and three fallback screenshots, one per game.',
     'Where: Supplements/README.md and Supplements/screenshots/.\nChecked: the ZIP was extracted to a new folder '
     'and every game, PDF and Word file opened from there.',
     [], False),
]


def merged_row(t):
    row = t.add_row()
    return row.cells[0].merge(row.cells[1]).merge(row.cells[2])


def main():
    os.makedirs(IMG, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
    sec.orientation = 1
    for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, m, Cm(1.6))
    st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10)

    h = doc.add_heading('Improvement Summary · Stage 3 Final Study Guide and Test Bank', level=1)
    for r in h.runs: r.font.color.rgb = PINK
    doc.add_paragraph('Team Fengwei Fried Rice · Hardware, Networks & Behaviourist AI · 28 November 2026')
    doc.add_paragraph('This table lists the meaningful changes from V1.0 (Stage 2) to the Final version, grouped by '
                      'Study Guide, Test Bank and Supplements: the feedback or problem, what we changed and why, and '
                      'where to find it and how we checked it. Pink rows answer the teacher\'s feedback directly; the '
                      'others come from our own review. The before/after evidence for the most important changes is '
                      'directly below its row.')

    t = doc.add_table(rows=1, cols=3); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = (Cm(7.0), Cm(11.0), Cm(8.4))
    for c, txt, w in zip(t.rows[0].cells,
                         ('Feedback or problem', 'What changed and why', 'Where to find it and how it was checked'),
                         widths):
        c.text = ''; r = c.paragraphs[0].add_run(txt); r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, 'D2386F'); c.width = w
    group = None
    for g, fb, what, where, imgs, teacher in ROWS:
        if g != group:
            group = g
            c = merged_row(t); c.text = ''
            r = c.paragraphs[0].add_run(g); r.bold = True; r.font.size = Pt(12); r.font.color.rgb = PINK
            shade(c, 'FDE3EC')
        cells = t.add_row().cells
        for c, txt, w in zip(cells, (fb, what, where), widths):
            c.text = txt; c.width = w
            if teacher: shade(c, 'FFF1F6')
        cells[0].paragraphs[0].runs[0].bold = True
        if imgs:
            c = merged_row(t); c.text = ''
            p = c.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for k, im in enumerate(imgs):
                if im[0] == 'pair':
                    path, wcm = pair(im[1], im[2]), (12.6 if len(imgs) > 1 else 13.5)
                else:
                    path, wcm = im[1], im[2]
                if k: p.add_run('   ')
                p.add_run().add_picture(path, width=Cm(wcm))
            cap = c.add_paragraph('Before (V1.0, grey label) and after (Final, pink label).' if imgs[0][0] == 'pair'
                                  else 'Evidence: the Final version.')
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True; cap.runs[0].font.size = Pt(9)
    for row in t.rows:  # keep each row on one page
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement('w:cantSplit'); trPr.append(cs)
    doc.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
