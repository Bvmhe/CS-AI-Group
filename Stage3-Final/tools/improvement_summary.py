"""Write Improvement_Summary.docx: teacher feedback -> what changed -> where/how checked, with before/after images."""
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
OUT = BASE + r'\Improvement_Summary.docx'
FONT = BASE + r'\tools\FredokaBold.ttf'
PINK = RGBColor(0xD2, 0x38, 0x6F)


def pair(pno, name, clip=None):
    """Side-by-side BEFORE | AFTER of one guide page (1-based), optionally clipped (pt rect)."""
    ims = []
    for path in (V1, FINAL):
        page = pymupdf.open(path)[pno - 1]
        pix = page.get_pixmap(dpi=100 if clip is None else 110, clip=clip)
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


ROWS = [
    ('Teacher: “Add images when appropriate, for instance a CPU image is expected when talking about it.”',
     'Added 15 real photos (free licences, Wikimedia Commons) where the guide talks about real hardware: CPU (p12), '
     'the four chips together (p11), RAM (p25), iPhone A11 board (p29), data centre (p38), 5G mast (p57), ANYmal robot '
     'dog (p83), car camera (p93), LiDAR (p94), parking sensor and radar (p95), Tesla (p111), TPU pod (p138). '
     'Each photo has a one-line caption that links it back to the idea on the page, plus a credit line. '
     'All credits are listed together on p167.',
     'Licence and author read from each file\'s Commons page (CC BY, CC BY-SA or CC0 only). Every changed page '
     'rendered and compared with V1.0 (images below) to check nothing covers the text.'),
    ('Teacher: “I don\'t get the kitchen game\'s relevancy to the learning.”',
     'Guide p145: new “Why this game? · Learning Goal 3” box. It says the maze is Recipe 03 made playable, maps '
     'each game part to a section (agent and Q-learning 3.3, rewards 3.2, reward hacking 3.3) and gives 3 debrief '
     'questions. Both games now open with a “Why this game?” screen (learning goal, what each part stands for, what to '
     'notice, 3 debrief questions) and keep a button to reopen it. The 24-Hour Diner screen shows that each room is one '
     'Recipe / Learning Goal and that customers\' orders are Test Bank questions (retrieval practice).',
     'Both games opened offline (file://, no internet) in Microsoft Edge: the screen appears, “Start playing” closes it, '
     'the button reopens it. Game terms on the screens (Learner Pup, Map Chef, Cookie Jar, α γ ε, rush hour) checked '
     'against the game code.'),
    ('Teacher: “The interactive test bank is well designed.”',
     'Kept as it is. Only the labels changed to “Final Test Bank”, and the Q49 section tag was fixed (next row).',
     'A script compared all 60 questions and answers in Mama\'s Quiz Kitchen with the paper Test Bank: all match.'),
    ('Own check: Q49 (raising MFU) pointed to §5.4 (energy and cost), but MFU is taught in §5.1.',
     'Changed the tag to §5.1 in the paper Test Bank (.docx/.pdf) and in the quiz game.',
     'Re-read §5.1 “MFU: Are the Cooks Really Cooking?” and the Q49 answer key; tag now matches.'),
    ('Own check: everything still said “Stage 2” / “V1.0”.',
     'Study guide cover now says “FINAL STUDY GUIDE”, Final (Stage 3), 28 November 2026; back page footer says '
     '“Final Study Guide”. Test Bank cover, footer and document title say “Final Test Bank”. Game titles relabelled.',
     'Text search of the Final PDFs and HTML files for “Stage 2”: none left in titles, covers or footers.'),
    ('Own check: the menu (p5) gave page numbers for Recipes 01–05 but not for Dessert and the Bonus Kitchen Lab; '
     'p4 used “You need” where the course template asks for “Prerequisites”.',
     'Added “starts on page 151 / 145 · tap to jump” with working links; renamed the p4 box to “Prerequisites”.',
     'Clicked every menu link in the Final PDF: each jumps to the right page.'),
    ('Own check: classmates in mainland China cannot open YouTube.',
     'Video pages (p33, p62) and the resource pages (p88, p116, p142) now have a “Can\'t watch?” note: use the English '
     'captions, or read the matching guide section and do its “How to read this” task instead.',
     'All 21 web links and 16 YouTube links in the guide opened and checked (from outside China).'),
]

SHOTS = [
    (1, 'cover.png', None, 'Cover: Stage 2 / V1.0 labels replaced by Final.'),
    (12, 'p012.png', None, 'p12 The CPU: real Intel Core i7 photo with caption and credit.'),
    (95, 'p095.png', None, 'p95 Sensors: real parking sensor and radar photos.'),
    (145, 'p145.png', None, 'p145 Bonus Kitchen Lab: new “Why this game?” box.'),
]


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
    doc.add_paragraph('This table lists every change from V1.0 (Stage 2) to the Final version, why we made it and how we '
                      'checked it. The first three rows answer the teacher\'s feedback directly; the rest come from our own review.')

    t = doc.add_table(rows=1, cols=3); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = (Cm(7.0), Cm(12.0), Cm(7.4))
    for c, txt, w in zip(t.rows[0].cells, ('Feedback or problem', 'What changed and why', 'Where and how checked'), widths):
        c.text = ''; r = c.paragraphs[0].add_run(txt); r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, 'D2386F'); c.width = w
    for i, row in enumerate(ROWS):
        cells = t.add_row().cells
        for c, txt, w in zip(cells, row, widths):
            c.text = txt; c.width = w
            if i < 3: shade(c, 'FFF1F6')
        cells[0].paragraphs[0].runs[0].bold = True

    for half in (SHOTS[:2], SHOTS[2:]):
        doc.add_page_break()
        hh = doc.add_heading('Before and after (study guide)', level=2)
        for r in hh.runs: r.font.color.rgb = PINK
        tbl = doc.add_table(rows=2, cols=2)
        for j, (pno, name, clip, cap) in enumerate(half):
            path = pair(pno, name, clip)
            c = tbl.rows[0].cells[j]; c.paragraphs[0].add_run().add_picture(path, width=Cm(12.4))
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            tbl.rows[1].cells[j].text = cap
    doc.add_page_break()
    hh = doc.add_heading('Games and Test Bank', level=2)
    for r in hh.runs: r.font.color.rgb = PINK
    g = doc.add_table(rows=2, cols=3)
    extra = [(BASE + r'\shots\why_Pup_Pantry_Maze_offline.png', 'Pup\'s Pantry Maze: new “Why this game?” start screen.'),
             (BASE + r'\shots\why_diner2.png', '24-Hour Diner: new “Why this game?” start screen (rooms = Recipes 01–05).'),
             (BASE + r'\tb\cover_final.png', 'Test Bank cover relabelled “Final Test Bank”.')]
    for j, (p, cap) in enumerate(extra):
        g.rows[0].cells[j].paragraphs[0].add_run().add_picture(p, width=Cm(8.3))
        g.rows[1].cells[j].text = cap
    doc.save(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
