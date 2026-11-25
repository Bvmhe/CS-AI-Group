"""Build the Stage 3 final Study Guide from the V1.0 PDF.

Reads  Stage3-Final/V1-from-GitHub/Fengwei_Study_Guide_HD_FULL.pdf (never modified)
Writes Stage3-Final/FengweiFriedRice_Stage3_FinalStudyGuide.pdf

Changes (from the Stage 3 planner + teacher feedback):
  * real-world photos on hardware / sensor pages (teacher: "add images, e.g. a CPU image")
  * cover relabelled as the final version; p4 Prerequisites; p5 page numbers
  * YouTube-in-China notes on video pages; p168 footer
  * "Why this game?" box on p145 (teacher: game relevance unclear)
  * photo credits on p167
"""
import json, os
import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'V1-from-GitHub', 'Fengwei_Study_Guide_HD_FULL.pdf')
OUT = os.path.join(ROOT, 'FengweiFriedRice_Stage3_FinalStudyGuide.pdf')
PHOTOS = os.path.join(ROOT, 'photos')
FREDOKA = os.path.join(ROOT, 'tools', 'FredokaBold.ttf')
ARIAL = r'C:\Windows\Fonts\arial.ttf'

BROWN = (0.29, 0.165, 0.102)       # body text colour in the guide (#4a2a1a)
SOFT = (0.478, 0.29, 0.208)        # footer / caption colour (#7a4a35)
PINK = (0.941, 0.318, 0.545)       # label pill pink
BORDER = (0.957, 0.698, 0.788)     # card border pink
WHITE = (1, 1, 1)
CREDITS = json.load(open(os.path.join(PHOTOS, 'credits.json'), encoding='utf8'))

CSS = """
* { font-family: sans-serif; color: #4a2a1a; font-size: 15.5px; line-height: 1.28; margin: 0; }
b { color: #4a2a1a; }
.cr { font-size: 10px; color: #7a4a35; margin-top: 3px; }
.cap { font-size: 11px; text-align: center; line-height: 1.15; }
"""
SRCDOC = pymupdf.open(SRC)   # untouched copy, used to clone background grid over hidden sprites
GRID = 27                    # background grid repeats every 27 pt


def hide_sprites(page, card_rect):
    """Cartoon sprites partly covered by a new card: hide the parts sticking out by copying
    a strip of plain background grid from a whole number of grid cells to the side."""
    src = SRCDOC[page.number]
    infos = src.get_image_info()
    texts = [pymupdf.Rect(s['bbox']) for b in src.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()]
    for i in infos:
        b = pymupdf.Rect(i['bbox'])
        if b.width > 220 or b.height > 260 or not b.intersects(card_rect):
            continue
        parts = []
        if b.y1 > card_rect.y1:
            parts.append(pymupdf.Rect(b.x0 - 2, card_rect.y1 - 1, b.x1 + 2, b.y1 + 1))
        if b.y0 < card_rect.y0 - 12:
            parts.append(pymupdf.Rect(b.x0 - 2, b.y0 - 1, b.x1 + 2, card_rect.y0 + 1))
        for part in parts:
            for k in (3, -3, 6, -6, 9, -9, 12, -12):
                s = part + (k * GRID, 0, k * GRID, 0)
                if s.x0 < 70 or s.x1 > 740:
                    continue
                clash = any(pymupdf.Rect(o['bbox']).intersects(s) and pymupdf.Rect(o['bbox']).width < 400 for o in infos)
                if clash or any(t.intersects(s) for t in texts):
                    continue
                pix = src.get_pixmap(clip=s, dpi=216)
                page.insert_image(part, stream=pix.tobytes('png'))
                break


def sample(page, x, y):
    pix = page.get_pixmap(clip=pymupdf.Rect(x, y, x + 1, y + 1), dpi=72)
    return tuple(c / 255 for c in pix.pixel(0, 0))


def strip_text(page, rects):
    """Remove text (only text) under the given rectangles; images and drawings stay."""
    for r in rects:
        page.add_redact_annot(pymupdf.Rect(r), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)


def fit_text(page, origin, text, size, maxw, color, fontfile=ARIAL, name='arialx'):
    f = pymupdf.Font(fontfile=fontfile)
    while f.text_length(text, fontsize=size) > maxw and size > 8:
        size -= 0.2
    page.insert_text(origin, text, fontsize=size, fontfile=fontfile, fontname=name, color=color)
    return size


def label(page, x, y, text, fill=PINK, size=11.5):
    f = pymupdf.Font(fontfile=FREDOKA)
    w = f.text_length(text, fontsize=size) + 18
    r = pymupdf.Rect(x, y, x + w, y + size + 9)
    page.draw_rect(r, color=None, fill=fill, radius=0.35)
    page.insert_text((x + 9, y + size + 3.2), text, fontsize=size, fontfile=FREDOKA, fontname='fredb', color=WHITE)


def card(page, rect, title):
    rect = pymupdf.Rect(rect)
    hide_sprites(page, rect)
    page.draw_rect(rect, color=BORDER, fill=WHITE, width=2.2, radius=0.12 if rect.height < 140 else 0.07)
    label(page, rect.x0 + 14, rect.y0 - 10, title)
    return rect


def credit_line(key):
    c = CREDITS[key]
    return f"Photo: {c['author']}, {c['licence']}, Wikimedia Commons"


def photo_card(page, rect, key, html, title='The real thing', maxh=170):
    r = card(page, rect, title)
    inner = pymupdf.Rect(r.x0 + 14, r.y0 + 16, r.x1 - 14, r.y1 - 10)
    img = pymupdf.Pixmap(os.path.join(PHOTOS, CREDITS[key]['file']))
    ar = img.width / img.height
    h = min(inner.height, maxh)
    w = min(h * ar, inner.width * 0.46)
    h = w / ar
    pr = pymupdf.Rect(inner.x0, inner.y0 + (inner.height - h) / 2, inner.x0 + w, inner.y0 + (inner.height - h) / 2 + h)
    page.insert_image(pr, filename=os.path.join(PHOTOS, CREDITS[key]['file']))
    page.draw_rect(pr, color=BORDER, width=1)
    tr = pymupdf.Rect(pr.x1 + 14, inner.y0, inner.x1, inner.y1)
    page.insert_htmlbox(tr, html + f'<p class="cr">{credit_line(key)}</p>', css=CSS, scale_low=0.6)


def last_text_y(page):
    ys = [s['bbox'][3] for b in page.get_text('dict')['blocks'] for l in b.get('lines', [])
          for s in l['spans'] if s['text'].strip() and s['bbox'][1] < 1320]
    return max(ys)


def free_rect(page, pad=38, bottom=1320, x0=62, x1=748):
    return pymupdf.Rect(x0, last_text_y(page) + pad, x1, bottom)


# ---------------------------------------------------------------- edits
def cover(doc):
    p = doc[0]
    band = sample(p, 450, 81)
    # band label is a picture with invisible text on top: remove the invisible text, paint over, redraw
    strip_text(p, [(241, 73, 389, 90), (494, 1056, 668, 1078), (135, 1083, 184, 1105)])
    p.draw_rect(pymupdf.Rect(238, 69, 425, 94), color=None, fill=band)
    p.insert_text((243, 87.5), 'FINAL STUDY GUIDE', fontsize=15.5, fontfile=FREDOKA, fontname='fredb', color=WHITE)
    # V1.0 star badge: find the brown lettering and cover it with the star's yellow
    pix = p.get_pixmap(clip=pymupdf.Rect(625, 205, 720, 285), dpi=144)
    xs, ys = [], []
    for yy in range(pix.height):
        for xx in range(pix.width):
            r, g, b = pix.pixel(xx, yy)[:3]
            if r < 170 and g < 110 and b < 70:
                xs.append(xx); ys.append(yy)
    bx0, bx1, by0, by1 = 625 + min(xs) / 2, 625 + max(xs) / 2, 205 + min(ys) / 2, 205 + max(ys) / 2
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    yellow = sample(p, cx, by0 - 9)
    rad = max(bx1 - bx0, by1 - by0) / 2 + 5
    p.draw_oval(pymupdf.Rect(cx - rad, cy - rad * 0.78, cx + rad, cy + rad * 0.78), color=None, fill=yellow)
    f = pymupdf.Font(fontfile=FREDOKA)
    size = 17
    w = f.text_length('FINAL', fontsize=size)
    piv = pymupdf.Point(cx, cy)
    p.insert_text((cx - w / 2, cy + size * 0.36), 'FINAL', fontsize=size, fontfile=FREDOKA, fontname='fredb',
                  color=(0.45, 0.24, 0.02), morph=(piv, pymupdf.Matrix(-14)))
    # version line in the About box (real text)
    fit_text(p, (492.8, 1074.6), ' Final (Stage 3) ·', 21.6, 200, BROWN)
    fit_text(p, (135.0, 1101.5), '28 November 2026', 21.6, 400, BROWN)


def prerequisites(doc):
    p = doc[3]
    gold = sample(p, 200, 648)
    strip_text(p, [(117, 642, 186, 660)])
    p.draw_polyline([(112, 632), (252, 632), (266, 647.5), (252, 663), (112, 663), (112, 632)], color=None, fill=gold, closePath=True)
    p.insert_text((119, 654.5), 'Prerequisites', fontsize=16.5, fontfile=FREDOKA, fontname='fredb', color=WHITE)


def menu_numbers(doc):
    p = doc[4]
    pink = (0.91, 0.27, 0.5)
    for x, y, text, target in [(232, 1011, 'starts on page 151 · tap to jump', 150), (322, 1135, 'starts on page 145 · tap to jump', 144)]:
        p.insert_text((x, y), text, fontsize=13, fontfile=FREDOKA, fontname='fredb', color=pink)
        w = pymupdf.Font(fontfile=FREDOKA).text_length(text, fontsize=13)
        p.insert_link({'kind': pymupdf.LINK_GOTO, 'from': pymupdf.Rect(x, y - 13, x + w, y + 3), 'page': target})


def real_crew(doc):
    """p11: four real chips under Figure 1.3 (CPU -> GPU -> TPU -> NPU)."""
    p = doc[10]
    r = card(p, free_rect(p, pad=26), 'Meet the real crew')
    cols = [(62, 219, 'p12_cpu', 'Intel Core i7 CPU'), (238, 396, 'p13_gpu', 'NVIDIA RTX 3070 GPU card'),
            (413, 573, 'p14_tpu', 'Google TPU v3 board'), (590, 748, 'p15_npu', 'Apple A14 chip (has an NPU)')]
    top, bottom = r.y0 + 14, r.y1 - 17
    for x0, x1, key, cap in cols:
        x0, x1 = max(x0, r.x0 + 8), min(x1, r.x1 - 8)
        box = pymupdf.Rect(x0 + 6, top, x1 - 6, bottom)
        p.insert_image(box, filename=os.path.join(PHOTOS, CREDITS[key]['file']), keep_proportion=True)
        p.insert_htmlbox(pymupdf.Rect(x0, bottom + 1, x1, r.y1 - 1), f'<p class="cap">{cap}</p>', css=CSS, scale_low=0.6)


PHOTO_PAGES = {
    12: ('p12_cpu', 'An Intel Core i7-6700K desktop CPU (2015). Under the metal lid is one chip with '
                    'just <b>4 cores</b>: four very capable head chefs.'),
    25: ('p25_ram', 'A DDR4 RAM module, the computer\'s main pantry. Under the red heat spreader are memory '
                    'chips; every trip to them costs far more energy than a multiply (Figure 1.9). AI accelerators place stacked '
                    '<b>HBM</b> memory right next to the chip to shorten that trip.'),
    29: ('p29_a11', 'An iPhone 8 main board (ruler in cm). Its <b>A11 Bionic</b> chip was the first '
                    'with Apple\'s Neural Engine (2017).'),
    38: ('p38_datacenter', 'Rows of servers in a data centre. Thousands of machines like these are '
                           'wired together by leaf and spine switches.'),
    57: ('p57_5g', '5G antennas on a mast in Munich: the first radio hop between your phone and an '
                   'AI service in the cloud.'),
    83: ('p83_anymal', 'ANYmal, a four-legged robot from ANYbotics (ETH Zurich). It learned to walk '
                       'with RL in simulation (Hwangbo et al., 2019).'),
    93: ('p93_carcamera', 'A front-view camera built into a car\'s grille badge. It is passive: it only '
                          'collects light onto an image sensor, so distance has to be worked out from the pixels.'),
    94: ('p94_lidar', 'A Velodyne spinning LiDAR, used on many early self-driving test cars. It spins '
                      '64 laser beams to build a 360° point cloud, like the 64-beam example above.'),
    111: ('p111_tesla', 'A Tesla Model 3 (2023). Its cameras sit at the top of the windscreen, in the '
                        'side pillars, the front fenders and above the rear plate. No LiDAR, and no radar since 2021–22.'),
    138: ('p138_tpupod', 'A Google TPU v4 board with 4 TPU chips (Jouppi et al., 2023). 4,096 such '
                         'chips form one pod, linked as a torus.'),
}


def sensors_p95(doc):
    p = doc[94]
    r = card(p, free_rect(p), 'The real thing')
    inner = pymupdf.Rect(r.x0 + 12, r.y0 + 15, r.x1 - 12, r.y1 - 8)
    half = inner.width / 2
    for i, (key, html) in enumerate([
        ('p95_parking', '<b>Ultrasonic:</b> the round disc in this bumper is a parking sensor. It pings sound and times the echo.'),
        ('p95_radar', '<b>Radar:</b> a 60 GHz imaging radar board. Radars for driving use 76–81 GHz.')]):
        x0 = inner.x0 + i * half
        img = pymupdf.Pixmap(os.path.join(PHOTOS, CREDITS[key]['file']))
        ar = img.width / img.height
        h = inner.height; w = min(h * ar, half * 0.48); h = w / ar
        pr = pymupdf.Rect(x0, inner.y0 + (inner.height - h) / 2, x0 + w, inner.y0 + (inner.height - h) / 2 + h)
        p.insert_image(pr, filename=os.path.join(PHOTOS, CREDITS[key]['file']))
        p.draw_rect(pr, color=BORDER, width=1)
        p.insert_htmlbox(pymupdf.Rect(pr.x1 + 8, inner.y0, x0 + half - 8, inner.y1),
                         f'<p style="font-size:13.5px">{html}</p><p class="cr">{credit_line(key)}</p>', css=CSS, scale_low=0.6)


def video_notes(doc):
    # p33 + p62: rewrite the existing "Can't watch?" sentence (real text)
    p = doc[32]
    strip_text(p, [(85, 1242, 711, 1260)])
    fit_text(p, (85.5, 1257.6), 'English captions (CC) on all three. If YouTube is blocked (as in mainland China), re-read', 18.12, 625, BROWN)
    p = doc[61]
    strip_text(p, [(85, 1095, 707, 1113)])
    fit_text(p, (85.5, 1110.7), 'English captions (CC) available. YouTube blocked (as in mainland China)? Use Figures', 18.12, 621, BROWN)
    # p88, p116, p142: add a short note card in the free space
    texts = {
        88: 'YouTube is blocked in mainland China without a VPN. Every <b>Do:</b> task on this page also works from the '
            'matching figure in Recipe 03 and its <b>How to read this</b> note, so you can finish the recipe offline.',
        116: 'YouTube is blocked in mainland China without a VPN. Every <b>Do:</b> task on this page also works from the '
             'matching figure in Recipe 04 and its <b>How to read this</b> note, so you can finish the recipe offline.',
        142: 'YouTube is blocked in mainland China without a VPN: use the <b>No internet? Fallback</b> box above, which '
             'covers every task on this page.',
    }
    for pn, html in texts.items():
        p = doc[pn - 1]
        fr = free_rect(p, pad=42)
        r = card(p, pymupdf.Rect(fr.x0, fr.y0, fr.x1, fr.y0 + 78), "Can't open YouTube?")
        p.insert_htmlbox(pymupdf.Rect(r.x0 + 16, r.y0 + 16, r.x1 - 16, r.y1 - 6), f'<p>{html}</p>', css=CSS, scale_low=0.6)


def game_box(doc):
    p = doc[144]
    r = card(p, free_rect(p, pad=40), 'Why this game? · Learning Goal 3')
    html = ('<p>The maze is <b>Recipe 03 made playable</b>. The Learner Pup is the Q-learning agent of 3.3, the carrot and '
            'treats are its rewards (3.2), the Map Chef shows that search is not learning, and the Cookie Jar kitchen is '
            'reward hacking (3.3).</p><p style="margin-top:4px"><b>After playing, answer:</b> (1) Which chef got better with '
            'practice, and why? (2) Why does the Map Chef need the whole map first? (3) What went wrong in the Cookie Jar, '
            'and how would you change the reward?</p>')
    p.insert_htmlbox(pymupdf.Rect(r.x0 + 16, r.y0 + 15, r.x1 - 16, r.y1 - 6), html, css=CSS.replace('15.5px', '13.5px'), scale_low=0.6)


def footer(doc):
    p = doc[167]
    strip_text(p, [(72, 1179, 738, 1198)])
    text = 'Hardware, Networks & Behaviourist AI · Final Study Guide · November 2026'
    w = pymupdf.Font(fontfile=ARIAL).text_length(text, fontsize=18.96)
    p.insert_text((405 - w / 2, 1195.4), text, fontsize=18.96, fontfile=ARIAL, fontname='arialx', color=SOFT)


def photo_credits(doc):
    p = doc[166]
    r = card(p, free_rect(p, pad=38), 'Real-world photos (Wikimedia Commons, free licences)')
    order = ['p12_cpu', 'p13_gpu', 'p14_tpu', 'p15_npu', 'p25_ram', 'p29_a11', 'p38_datacenter', 'p57_5g',
             'p83_anymal', 'p93_carcamera', 'p94_lidar', 'p95_parking', 'p95_radar', 'p111_tesla', 'p138_tpupod']
    rows = []
    for k in order:
        c = CREDITS[k]
        pg = 'p11–15' if k in ('p13_gpu', 'p14_tpu', 'p15_npu') else 'p' + k.split('_')[0][1:]
        rows.append(f'<li><b>{pg}</b> {c["title"].rsplit(".", 1)[0][:38]}: {c["author"][:28]}, {c["licence"]}</li>')
    half = (len(rows) + 1) // 2
    css = CSS + 'ul{margin:0;padding-left:10px} li{font-size:8.6px;line-height:1.25;margin:0}'
    inner = pymupdf.Rect(r.x0 + 14, r.y0 + 15, r.x1 - 12, r.y1 - 5)
    mid = inner.x0 + inner.width / 2
    p.insert_htmlbox(pymupdf.Rect(inner.x0, inner.y0, mid - 6, inner.y1), '<ul>' + ''.join(rows[:half]) + '</ul>', css=css, scale_low=0.5)
    p.insert_htmlbox(pymupdf.Rect(mid + 6, inner.y0, inner.x1, inner.y1), '<ul>' + ''.join(rows[half:]) + '</ul>', css=css, scale_low=0.5)


def main():
    doc = pymupdf.open(SRC)
    cover(doc); prerequisites(doc); menu_numbers(doc); real_crew(doc)
    for pn, (key, html) in PHOTO_PAGES.items():
        photo_card(doc[pn - 1], free_rect(doc[pn - 1]), key, f'<p>{html}</p>')
    sensors_p95(doc); video_notes(doc); game_box(doc); footer(doc); photo_credits(doc)
    doc.save(OUT, garbage=3, deflate=True)
    print('saved', OUT, round(os.path.getsize(OUT) / 1e6, 1), 'MB', len(doc), 'pages')


if __name__ == '__main__':
    main()
