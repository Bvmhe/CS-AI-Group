"""Fill the guide p150 'Team contribution' box with the team's real details (run after revise_guide.py).

Replaces the "[TEAM: ...]" placeholders and the "(fill in before submitting)" title note in
FengweiFriedRice_Stage3_FinalStudyGuide.pdf. Details come from the user (26 Nov 2026) and the team README roster.
"""
import os, shutil
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, 'FengweiFriedRice_Stage3_FinalStudyGuide.pdf')
CALIBRI_B = r'C:\Windows\Fonts\calibrib.ttf'
FONTS = r'C:\Windows\Fonts'
FREDOKA = os.path.join(ROOT, 'tools', 'FredokaBold.ttf')

HTML = """
<p><b>Who built, tested and reviewed which part:</b> Pup's Pantry Maze (this Bonus Kitchen Arcade) and the
Test Bank questions were built by 曾富良; all 60 Test Bank questions were checked by 黄施杰 (Darren, Quality Lead),
28 Oct – 5 Nov 2026. Team roles: 纪小佩 CEO (plan, coordination) · 林元甲 CIO (repository, packaging) ·
温俊强 and 汤丽嘉 Content Leads · 曾富良 Visual Design Lead · 李秀春 Video Lead · 黄施杰 Quality Lead.</p>
<p><b>Classmate try-out:</b> 2 classmates from team DOOM tried the arcade.</p>
"""
CSS = """
@font-face { font-family: yh; src: url(msyh.ttc); }
* { font-family: sans-serif, yh; color: #4a2a1a; font-size: 14px; line-height: 1.28; margin: 0; }
p { margin-bottom: 5px; }
"""


def main():
    doc = pymupdf.open(PDF)
    p = doc[149]
    if '[TEAM:' not in p.get_text():
        print('p150 already filled; nothing to do'); return
    shutil.copy2(PDF, PDF.replace('.pdf', '_before_p150.pdf'))
    title = p.search_for('Team contribution (fill in before submitting)')[0]
    body = pymupdf.Rect(80, 1058, 730, 1222)          # the three paragraphs inside the card (card is y 1011-1228)
    for r in (title, body):
        p.add_redact_annot(r, fill=False)
    p.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                       text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    # the visible title is drawn art inside a gold banner: paint over it in the banner colour, then write the new title
    pix = p.get_pixmap(clip=pymupdf.Rect(498, 1038, 502, 1042))
    gold = tuple(c / 255 for c in pix.pixel(2, 2))
    p.draw_rect(pymupdf.Rect(114, 1026, 478, 1056), color=None, fill=gold)
    p.insert_text((119, 1048.5), 'Team contribution', fontsize=19, fontfile=FREDOKA, fontname='fredb', color=(1, 1, 1))
    spare, scale = p.insert_htmlbox(pymupdf.Rect(85.5, 1062, 725, 1207), HTML, css=CSS,
                                    archive=pymupdf.Archive(FONTS), scale_low=0.7)
    print('htmlbox spare', round(spare, 1), 'scale', round(scale, 2))
    tmp = PDF + '.tmp'
    doc.save(tmp, garbage=3, deflate=True); doc.close()
    os.replace(tmp, PDF)
    print('p150 filled')


if __name__ == '__main__':
    main()
