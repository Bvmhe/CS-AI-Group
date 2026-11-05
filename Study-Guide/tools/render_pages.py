"""Render the 9:16 study-guide pages.

For every page in the content modules this draws a 1080x1920 background PNG (kitchen skin,
chapter badge, section pill, window frames, tables, sprites) and records the editable text
slots that will become Canva text boxes. It also writes a preview PNG with the text drawn in
Arial (the same metrics as Canva's default font) so the layout can be checked before upload.

Usage: python render_pages.py <outdir> [page numbers...]
Content: Study-Guide/content/pages_*.py, each defining PAGES = [ {...}, ... ]
"""
import glob, importlib.util, json, math, os, re, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, 'assets', 'fonts')
SPRITES = os.path.join(ROOT, 'assets', 'sprites')
W, H = 1080, 1920
NAVY, RED, CREAM = (23, 50, 77), (239, 63, 79), (255, 244, 220)
COL = {'yellow': (255, 201, 74), 'mint': (143, 213, 192), 'pink': (255, 141, 161), 'blue': (77, 157, 224),
       'white': (255, 255, 255), 'cream': (255, 236, 200), 'lav': (190, 170, 240), 'orange': (255, 166, 77),
       'red': RED, 'navy': NAVY}
BODY = 30          # default editable text size
LH = 1.4           # Canva default line height
X0, X1 = 60, 1020  # content area
TOP, BOT, BOT_FLOOR = 300, 1765, 1600
GAP = 34

CH_NAMES = {0: 'FENGWEI FRIED RICE', 1: 'THE KITCHEN CREW: AI HARDWARE', 2: 'ORDER UP! NETWORKS & AI INFRA',
            3: 'TRAINING THE KITCHEN PUP: RL', 4: "CHEF'S SENSES: AI SENSORS", 5: 'RUNNING THE CHAIN: INTEGRATION',
            9: "MAMA'S PANTRY: REFERENCES"}
CH_COL = {0: RED, 1: (231, 76, 60), 2: (52, 152, 219), 3: (46, 174, 96), 4: (155, 89, 182), 5: (230, 126, 34), 9: NAVY}


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONTS, name) if not name.endswith('arial.ttf') and '\\' not in name and name not in ('arial.ttf', 'arialbd.ttf', 'consola.ttf') else name, size)
    if var:
        f.set_variation_by_name(var)
    return f


EMOJI = 'C:\\Windows\\Fonts\\seguiemj.ttf'


def label(d, xy, text, f, fill=NAVY, anchor='lm'):
    """Draw text whose first character may be an emoji (drawn with Segoe UI Emoji)."""
    x, y = xy
    text = text.replace('️', '')   # variation selector draws as a box
    if text and ord(text[0]) >= 0x2300:   # emoji / symbols (⏱ ⌚ ☕ ...) that Fredoka lacks
        em, rest = text.split(' ', 1) if ' ' in text else (text, '')
        ef = ImageFont.truetype(EMOJI, int(f.size * 0.9))
        d.text((x, y), em, font=ef, fill=fill, anchor=anchor, embedded_color=True)
        x += ef.getlength(em) + 10
        text = rest
    d.text((x, y), text, font=f, fill=fill, anchor=anchor)


def pix(size):
    return font('PressStart2P.ttf', size)


def fred(size, var='SemiBold'):
    return font('Fredoka.ttf', size, var)


def arial(size, bold=False):
    return ImageFont.truetype('arialbd.ttf' if bold else 'arial.ttf', size)


def wrap(text, f, width):
    lines = []
    for para in text.split('\n'):
        if not para.strip():
            lines.append('')
            continue
        words, cur = para.split(' '), ''
        for w_ in words:
            t = (cur + ' ' + w_) if cur else w_
            if f.getlength(t) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                while f.getlength(w_) > width:  # very long token
                    k = len(w_)
                    while k > 1 and f.getlength(w_[:k]) > width:
                        k -= 1
                    lines.append(w_[:k]); w_ = w_[k:]
                cur = w_
        lines.append(cur)
    return lines


CANVA_W = 1.10     # Canva's text font is ~6-8% wider than Arial (measured); size boxes for it plus margin


def text_h(text, size, width):
    n = len(wrap(text, arial(size), (width - 6) / CANVA_W))
    return math.ceil(n * size * LH) + 4


# ---------------------------------------------------------------- sprites
_cache = {}


def sprite(name, h=None, w=None):
    key = (name, h, w)
    if key in _cache:
        return _cache[key]
    im = Image.open(os.path.join(SPRITES, name + '.png')).convert('RGBA')
    if h and not w:
        w = round(im.width * h / im.height)
    elif w and not h:
        h = round(im.height * w / im.width)
    elif w and h:
        s = min(w / im.width, h / im.height)
        w, h = round(im.width * s), round(im.height * s)
    im = im.resize((w, h), Image.LANCZOS)
    # white sticker outline + soft shadow, like a game sprite
    a = im.getchannel('A')
    pad = 8
    big = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ol = Image.new('L', big.size, 0); ol.paste(a, (pad, pad))
    ol = ol.filter(ImageFilter.MaxFilter(7))
    sh = ol.filter(ImageFilter.GaussianBlur(5)).point(lambda v: int(v * .35))
    out = Image.new('RGBA', (big.width + 8, big.height + 8), (0, 0, 0, 0))
    out.paste((23, 50, 77, 255), (8, 8), sh)
    out.paste((255, 255, 255, 255), (0, 0), ol)
    out.alpha_composite(im, (pad, pad))
    _cache[key] = out
    return out


NOSPRITE = os.environ.get('NOSPRITE') == '1'   # leave pictures out of the background; list them in PLACED instead
PLACED = []        # [{'key': sprite name or photo path, 'kind', 'x', 'y', 'w', 'h'}] for the current page
BIGGEST = {}       # key -> largest rendered version, saved once for upload


def place(bg, im, x, y, key, kind='sprite'):
    """Composite a picture onto the background, or (NOSPRITE) record it as a separate Canva image."""
    if NOSPRITE:
        PLACED.append(dict(key=key, kind=kind, x=int(x), y=int(y), w=im.width, h=im.height))
        if key not in BIGGEST or BIGGEST[key].width < im.width:
            BIGGEST[key] = im
    elif kind == 'photo':
        bg.paste(im, (int(x), int(y)))
    else:
        bg.alpha_composite(im, (int(x), int(y)))


def put(bg, name, x, y, h=None, w=None, anchor='tl'):
    s = sprite(name, h, w)
    if anchor == 'bc':
        x, y = x - s.width // 2, y - s.height
    elif anchor == 'c':
        x, y = x - s.width // 2, y - s.height // 2
    place(bg, s, x, y, name)
    return s.size


# ---------------------------------------------------------------- skin
def gingham(d, y0, y1, a=(255, 214, 222), b=(255, 255, 255), c=(255, 180, 196), s=40):
    for y in range(y0, y1, s):
        for x in range(0, W, s):
            i, j = (x // s) % 2, (y // s) % 2
            col = c if (i and j) else (a if (i or j) else b)
            d.rectangle([x, y, x + s - 1, min(y + s - 1, y1)], fill=col)


def skin(pg, n):
    bg = Image.new('RGBA', (W, H), CREAM + (255,))
    d = ImageDraw.Draw(bg)
    # wall tiles
    for y in range(240, 1800, 90):
        d.line([(0, y), (W, y)], fill=(250, 232, 200), width=3)
    for x in range(0, W, 90):
        d.line([(x, 240), (x, 1800)], fill=(250, 232, 200), width=3)
    # tablecloth header with scalloped edge
    gingham(d, 0, 230)
    for x in range(-30, W + 60, 60):
        d.ellipse([x, 200, x + 60, 260], fill=CREAM)
    # wooden counter
    d.rectangle([0, 1800, W, H], fill=(201, 138, 85))
    d.rectangle([0, 1800, W, 1822], fill=(231, 168, 111))
    for y in (1860, 1895):
        d.line([(0, y), (W, y)], fill=(176, 113, 64), width=3)
    for x in (180, 520, 860):
        d.line([(x, 1822), (x, 1860)], fill=(176, 113, 64), width=3)
    ch = pg.get('ch', 0)
    # chapter badge
    cc = CH_COL.get(ch, RED)
    d.rounded_rectangle([34, 34, 640, 150], 22, fill=NAVY)
    d.rounded_rectangle([46, 46, 146, 138], 16, fill=(255, 201, 74))
    lab = str(ch) if 1 <= ch <= 5 else ('0' if ch == 0 else '+')
    f = pix(52)
    d.text((96, 94), lab, font=f, fill=NAVY, anchor='mm')
    d.polygon([(160, 72), (160, 116), (190, 94)], fill=cc)
    name = pg.get('badge', CH_NAMES.get(ch, ''))
    lines = wrap(name, pix(20), 420)[:2]
    for i, ln in enumerate(lines):
        d.text((206, 94 + (i - (len(lines) - 1) / 2) * 34), ln, font=pix(20), fill='white', anchor='lm')
    # home / wok button
    d.ellipse([930, 34, 1046, 150], fill=NAVY)
    d.polygon([(988, 62), (1018, 90), (958, 90)], fill='white')
    d.rectangle([966, 90, 1010, 124], fill='white')
    d.rectangle([982, 104, 994, 124], fill=NAVY)
    # footer
    d.text((40, 1880), 'Hot Wok, Smart Chips  ·  Fengwei Fried Rice', font=fred(24, 'Medium'), fill=(255, 240, 220), anchor='lm')
    d.rounded_rectangle([930, 1850, 1050, 1910], 14, fill=NAVY)
    d.text((990, 1880), str(n), font=pix(22), fill='white', anchor='mm')
    # section pill
    pill = pg.get('pill')
    if pill:
        pf = pix(24)
        pl = wrap(pill, pf, 760)[:2]
        pw = max(pf.getlength(x) for x in pl) + 90
        ph = 46 * len(pl) + 34
        x0 = (W - pw) / 2
        d.rounded_rectangle([x0 + 8, 184 + 8, x0 + pw + 8, 184 + ph + 8], ph / 2, fill=NAVY)
        d.rounded_rectangle([x0, 184, x0 + pw, 184 + ph], ph / 2, fill=(236, 236, 236), outline=NAVY, width=5)
        d.rounded_rectangle([x0 + 14, 184 + 12, x0 + pw - 14, 184 + ph - 12], (ph - 24) / 2, fill='white')
        for i, ln in enumerate(pl):
            d.text((W / 2, 184 + ph / 2 + (i - (len(pl) - 1) / 2) * 46), ln, font=pf, fill=NAVY, anchor='mm')
        pg['_top'] = max(TOP, 184 + ph + 46)
    else:
        pg['_top'] = 200
    return bg, d


# ---------------------------------------------------------------- blocks
def window(bg, d, x, y, w, h, title, color, icon=None):
    d.rounded_rectangle([x + 10, y + 10, x + w + 10, y + h + 10], 16, fill=NAVY)
    d.rounded_rectangle([x, y, x + w, y + h], 16, fill='white', outline=NAVY, width=5)
    if title is not None:
        d.rounded_rectangle([x, y, x + w, y + 66], 16, fill=COL.get(color, color), outline=NAVY, width=5)
        d.rectangle([x + 5, y + 40, x + w - 5, y + 61], fill=COL.get(color, color))
        d.line([(x, y + 64), (x + w, y + 64)], fill=NAVY, width=5)
        tf = fred(34, 'SemiBold')
        tx = x + 24
        if icon and title and ord(title[0]) >= 0x2300:   # title has its own emoji too: keep one icon only
            title = title.split(' ', 1)[1] if ' ' in title else title
        label(d, (tx, y + 34), (icon + ' ' + title) if icon else title, tf)
        for k, ch in enumerate(('□', '✕')):
            bx = x + w - 92 + k * 44
            d.rounded_rectangle([bx, y + 16, bx + 34, y + 50], 6, fill='white', outline=NAVY, width=3)
            if ch == '✕':
                d.line([(bx + 9, y + 25), (bx + 25, y + 41)], fill=NAVY, width=4)
                d.line([(bx + 25, y + 25), (bx + 9, y + 41)], fill=NAVY, width=4)
            else:
                d.rectangle([bx + 9, y + 25, bx + 25, y + 41], outline=NAVY, width=3)
        return y + 66
    return y


def lay_block(b, size):
    """Return the height a block needs at body size `size`."""
    k = b['kind']
    w = b.get('w', X1 - X0)
    if k == 'win':
        sp = b.get('sprite')
        tw = w - 56 - (b.get('sprite_w', 200) + 10 if sp else 0)
        th = text_h(b['text'], b.get('size', size), tw)
        sph = sprite(sp, w=b.get('sprite_w', 200)).height + 20 if sp else 0
        return 66 * (b.get('title') is not None) + max(th, sph) + 52
    if k == 'bubble':
        tw = w - 230 - 70
        return max(text_h(b['text'], b.get('size', size), tw) + 60, 250)
    if k == 'note':
        return text_h(b['text'], b.get('size', size - 2), w - 70) + 92
    if k == 'big':
        return 120
    if k == 'title':
        f = fred(b.get('size', 96), 'Bold')
        return len(wrap(b['text'], f, w - 40)) * b.get('size', 96) * 1.12 + 30
    if k == 'sprites':
        return b.get('h', 220) + 30
    if k == 'table':
        return table_h(b)
    if k == 'code':
        return len(b['text'].split('\n')) * 34 + 60
    if k == 'row':
        n = len(b['items'])
        cw = (w - b.get('gap', 30) * (n - 1)) / n
        return max(lay_block(dict(c, w=cw), size) for c in b['items'])
    if k == 'photo':
        im = photo_img(b, w)
        return im.height + 40 + (len(wrap(b.get('caption', ''), arial(24), w - 60)) * 30 + 20 if b.get('caption') else 0)
    raise ValueError(k)


REPO = os.path.dirname(ROOT)


def photo_img(b, w):
    """A picture from the repo (path relative to the repo root), scaled to fit the frame."""
    key = ('photo', b['path'], w, b.get('h', 560))
    if key not in _cache:
        im = Image.open(os.path.join(REPO, b['path'])).convert('RGB')
        s = min((w - 40) / im.width, b.get('h', 560) / im.height)
        _cache[key] = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return _cache[key]


def cell_font(b, i, c):
    """The font a body cell is actually drawn with (bold first column, YaHei for Chinese)."""
    fs = b.get('fs', 24)
    bold = i == 0 and b.get('bold_first', True)
    if any(ord(ch) >= 0x2E80 for ch in c):   # Chinese names: Arial has no CJK glyphs
        return ImageFont.truetype('msyhbd.ttc' if bold else 'msyh.ttc', fs)
    return arial(fs, bold)


def table_dims(b):
    """Header height and row heights, measured with the same fonts draw_table uses."""
    ws = b['widths']; fb = arial(b.get('fs', 24), True)
    hh = max(len(wrap(c, fb, ws[i] - 24)) for i, c in enumerate(b['cols'])) * 32 + 26
    rows_h = [max(len(wrap(c, cell_font(b, i, c), ws[i] - 24)) for i, c in enumerate(r)) * 31 + 22 for r in b['rows']]
    return hh, rows_h


def table_h(b):
    hh, rows_h = table_dims(b)
    return hh + sum(rows_h) + 10


def draw_table(d, x, y, b):
    ws = b['widths']; fs = b.get('fs', 24)
    f, fb = arial(fs), arial(fs, True)
    hc = COL.get(b.get('color', 'blue'), b.get('color'))
    tw = sum(ws)
    hh, rows_h = table_dims(b)
    total = hh + sum(rows_h)
    d.rectangle([x + 10, y + 10, x + tw + 10, y + total + 10], fill=NAVY)
    d.rectangle([x, y, x + tw, y + hh], fill=hc)
    cx = x
    for i, c in enumerate(b['cols']):
        for j, ln in enumerate(wrap(c, fb, ws[i] - 24)):
            d.text((cx + 12, y + 13 + j * 32), ln, font=fb, fill='white' if b.get('color', 'blue') in ('blue', 'navy', 'red') else NAVY)
        cx += ws[i]
    yy = y + hh
    for r_i, r in enumerate(b['rows']):
        rh = rows_h[r_i]
        d.rectangle([x, yy, x + tw, yy + rh], fill=(255, 255, 255) if r_i % 2 == 0 else (255, 246, 228))
        cx = x
        for i, c in enumerate(r):
            cf = cell_font(b, i, c)
            for j, ln in enumerate(wrap(c, cf, ws[i] - 24)):
                d.text((cx + 12, yy + 11 + j * 31), ln, font=cf, fill=NAVY)
            cx += ws[i]
        yy += rh
    cx = x
    for wv in ws[:-1]:
        cx += wv
        d.line([(cx, y), (cx, y + total)], fill=NAVY, width=3)
    yy = y + hh
    d.line([(x, yy), (x + tw, yy)], fill=NAVY, width=4)
    for rh in rows_h[:-1]:
        yy += rh
        d.line([(x, yy), (x + tw, yy)], fill=(200, 180, 150), width=2)
    d.rectangle([x, y, x + tw, y + total], outline=NAVY, width=5)
    return total


def draw_block(bg, d, b, x, y, w, size, slots):
    k = b['kind']
    hgt = lay_block(dict(b, w=w), size)
    if k == 'win':
        ty = window(bg, d, x, y, w, hgt, b.get('title'), b.get('color', 'yellow'), b.get('icon'))
        sp = b.get('sprite')
        spw = b.get('sprite_w', 200)
        tx, tw = x + 28, w - 56 - (spw + 10 if sp else 0)
        if sp and b.get('sprite_side', 'right') == 'left':
            tx += spw + 10
        if sp:
            s = sprite(sp, w=spw)
            sx = x + w - 24 - s.width if b.get('sprite_side', 'right') == 'right' else x + 20
            place(bg, s, sx, y + hgt - s.height - 14, sp)
        slots.append(dict(x=tx, y=ty + 24, w=tw, text=b['text'], size=b.get('size', size)))
    elif k == 'bubble':
        sp = b['sprite']; left = b.get('side', 'left') == 'left'
        s = sprite(sp, h=230)
        sx = x if left else x + w - s.width
        bx0 = x + 240 if left else x
        bx1 = x + w if left else x + w - 240
        d.rounded_rectangle([bx0 + 8, y + 8, bx1 + 8, y + hgt + 8], 34, fill=NAVY)
        d.rounded_rectangle([bx0, y, bx1, y + hgt], 34, fill='white', outline=NAVY, width=5)
        ty0 = y + hgt - 120
        if left:
            d.polygon([(bx0 + 3, ty0), (bx0 - 40, ty0 + 40), (bx0 + 3, ty0 + 60)], fill='white', outline=None)
            d.line([(bx0, ty0), (bx0 - 40, ty0 + 40), (bx0, ty0 + 60)], fill=NAVY, width=5)
        else:
            d.polygon([(bx1 - 3, ty0), (bx1 + 40, ty0 + 40), (bx1 - 3, ty0 + 60)], fill='white')
            d.line([(bx1, ty0), (bx1 + 40, ty0 + 40), (bx1, ty0 + 60)], fill=NAVY, width=5)
        place(bg, s, sx, y + hgt - s.height + 10, sp)
        slots.append(dict(x=bx0 + 34, y=y + 28, w=bx1 - bx0 - 68, text=b['text'], size=b.get('size', size)))
    elif k == 'note':
        c = COL.get(b.get('color', 'mint'))
        d.rounded_rectangle([x + 8, y + 8, x + w + 8, y + hgt + 8], 26, fill=NAVY)
        d.rounded_rectangle([x, y, x + w, y + hgt], 26, fill=c, outline=NAVY, width=5)
        lab = b.get('label', '🔬 REAL TECH')
        lf = fred(28, 'Bold')
        lw = lf.getlength(lab) + 60
        d.rounded_rectangle([x + 24, y + 16, x + 24 + lw, y + 62], 23, fill='white', outline=NAVY, width=3)
        label(d, (x + 44, y + 39), lab, lf)
        slots.append(dict(x=x + 34, y=y + 76, w=w - 70, text=b['text'], size=b.get('size', size - 2)))
    elif k == 'big':
        f = pix(66)
        d.text((W / 2 + 6, y + 66), b['text'], font=f, fill=NAVY, anchor='mm')
        d.text((W / 2, y + 60), b['text'], font=f, fill=COL.get(b.get('color', 'red')), anchor='mm')
    elif k == 'title':
        sz = b.get('size', 96)
        f = fred(sz, 'Bold')
        for i, ln in enumerate(wrap(b['text'], f, w - 40)):
            cy = y + 15 + i * sz * 1.12 + sz * 0.56
            for dx, dy in ((-6, 0), (6, 0), (0, -6), (0, 6), (-4, -4), (4, 4), (-4, 4), (4, -4)):
                d.text((W / 2 + dx, cy + dy), ln, font=f, fill=NAVY, anchor='mm')
            d.text((W / 2 + 7, cy + 9), ln, font=f, fill=NAVY, anchor='mm')
            d.text((W / 2, cy), ln, font=f, fill=COL.get(b.get('color', 'yellow'), b.get('color')), anchor='mm')
    elif k == 'sprites':
        hh = b.get('h', 220)
        items = b['items']
        ims = [sprite(n, h=hh) for n in items]
        tot = sum(i.width for i in ims) + 20 * (len(ims) - 1)
        cx = x + (w - tot) / 2
        for nm, im in zip(items, ims):
            place(bg, im, cx, y + 10, nm)
            cx += im.width + 20
    elif k == 'table':
        draw_table(d, x, y, b)
    elif k == 'photo':
        im = photo_img(b, w)
        d.rounded_rectangle([x + 10, y + 10, x + w + 10, y + hgt + 10], 22, fill=NAVY)
        d.rounded_rectangle([x, y, x + w, y + hgt], 22, fill='white', outline=NAVY, width=5)
        ix = int(x + (w - im.width) / 2)
        place(bg, im, ix, y + 20, b['path'], 'photo')
        d.rectangle([ix - 2, y + 18, ix + im.width + 1, y + 21 + im.height], outline=(200, 180, 150), width=2)
        if b.get('caption'):
            f = arial(24)
            for j, ln in enumerate(wrap(b['caption'], f, w - 60)):
                d.text((x + w / 2, y + im.height + 52 + j * 30), ln, font=f, fill=NAVY, anchor='mm')
        if b.get('label'):   # sticker tag on the frame's top-left corner, like the note labels
            lf = fred(26, 'Bold')
            lw = lf.getlength(b['label']) + 56
            d.rounded_rectangle([x - 6 + 4, y - 22 + 4, x - 6 + lw + 4, y + 22 + 4], 22, fill=NAVY)
            d.rounded_rectangle([x - 6, y - 22, x - 6 + lw, y + 22], 22, fill=COL['yellow'], outline=NAVY, width=3)
            label(d, (x + 18, y), b['label'], lf)
    elif k == 'code':
        d.rounded_rectangle([x + 8, y + 8, x + w + 8, y + hgt + 8], 18, fill=(0, 0, 0))
        d.rounded_rectangle([x, y, x + w, y + hgt], 18, fill=(36, 40, 56), outline=NAVY, width=4)
        for i, ln in enumerate(b['text'].split('\n')):
            d.text((x + 28, y + 30 + i * 34), ln, font=ImageFont.truetype('consola.ttf', 26), fill=(180, 240, 200))
    elif k == 'row':
        n = len(b['items'])
        gw = b.get('gap', 30)
        cw = (w - gw * (n - 1)) / n
        for i, c in enumerate(b['items']):
            c2 = dict(c)
            draw_block(bg, d, c2, x + i * (cw + gw), y, cw, size, slots)
    return hgt


def render(pg, n):
    PLACED.clear()
    bg, d = skin(pg, n)
    slots = []
    blocks = pg.get('blocks', [])
    floor = pg.get('floor')
    bottom = BOT_FLOOR if floor else BOT
    top = pg['_top']
    size = pg.get('size', BODY)
    need_at = lambda s: sum(lay_block(dict(b, w=b.get('w', X1 - X0)), s) for b in blocks) + GAP * (len(blocks) - 1)
    if 'size' not in pg:  # short pages: grow text to use the space (up to 36)
        while size < 36 and need_at(size + 1) <= (bottom - top) * 0.92:
            size += 1
    while True:
        need = sum(lay_block(dict(b, w=b.get('w', X1 - X0)), size) for b in blocks) + GAP * (len(blocks) - 1)
        if need <= bottom - top or size <= 22:
            break
        size -= 1
    over = need - (bottom - top)
    gap = GAP if over > 0 or not pg.get('spread', True) else min(GAP + (bottom - top - need) / max(1, len(blocks) + 1), 90)
    y = top + (0 if gap == GAP else gap - GAP)
    for b in blocks:
        w = b.get('w', X1 - X0)
        x = X0 + (X1 - X0 - w) / 2
        hgt = draw_block(bg, d, b, x, y, w, size, slots)
        y += hgt + gap
    if floor:
        xs = [W * (i + 1) / (len(floor) + 1) for i in range(len(floor))]
        for xx, nm in zip(xs, floor):
            nm, fh = (nm, None) if isinstance(nm, str) else nm
            if fh is None:
                fh = 140 if nm in FOODS else pg.get('floor_h', 225)
            put(bg, nm, xx, 1842, h=fh, anchor='bc')
    for (nm, sx, sy, sh) in pg.get('deco', []):
        put(bg, nm, sx, sy, h=sh)
    warn = over > 0
    return bg, slots, size, warn, over


try:
    from fontTools.ttLib import TTFont
    _ARIAL_CMAP = set(TTFont(os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf')).getBestCmap())
except Exception:
    _ARIAL_CMAP = None


def _needs_alt(ch):
    o = ord(ch)
    if _ARIAL_CMAP is not None:
        return o not in _ARIAL_CMAP
    return o >= 0x2190 and not (0x2190 <= o <= 0x2195 or 0x2200 <= o <= 0x22FF)


def draw_runs(d, x, y, text, f, size):
    """Draw a line with Arial, switching to Segoe UI Emoji / Symbol / YaHei for characters Arial lacks."""
    run = ''
    for ch in text + '\0':
        if ch == '\0' or (_needs_alt(ch) and not ch.isspace()):
            if run:
                d.text((x, y), run, font=f, fill=NAVY)
                x += d.textlength(run, font=f)
                run = ''
            if ch == '\0':
                break
            o = ord(ch)
            if o in (0xFE0F, 0x200D):
                continue
            if o >= 0x2E80 and o < 0xA000:
                af = ImageFont.truetype('msyh.ttc', size)
                d.text((x, y), ch, font=af, fill=NAVY)
            elif o >= 0x1F000 or o in range(0x2600, 0x27C0) or o in range(0x2B00, 0x2C00):
                af = ImageFont.truetype(EMOJI, int(size * 0.9))
                d.text((x, y + size * 0.08), ch, font=af, embedded_color=True)
            else:
                af = ImageFont.truetype('seguisym.ttf', size)
                d.text((x, y), ch, font=af, fill=NAVY)
            x += d.textlength(ch, font=af)
        else:
            run += ch


def preview(bg, slots):
    pv = bg.copy()
    d = ImageDraw.Draw(pv)
    for s in slots:
        f = arial(s['size'])
        for i, ln in enumerate(wrap(s['text'], f, (s['w'] - 6) / CANVA_W)):   # same line breaks as Canva's wider font
            draw_runs(d, s['x'], s['y'] + i * s['size'] * LH + s['size'] * 0.15, ln, f, s['size'])
    return pv


FOODS = set()


def load_pages():
    global FOODS
    sys.path.insert(0, os.path.join(ROOT, 'content'))
    import kit
    FOODS = kit.FOODS
    pages = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'content', 'pages_*.py'))):
        spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        pages += m.PAGES
    return pages


if __name__ == '__main__':
    out = sys.argv[1]
    only = set(int(a) for a in sys.argv[2:])
    os.makedirs(os.path.join(out, 'bg'), exist_ok=True)
    os.makedirs(os.path.join(out, 'pv'), exist_ok=True)
    pages = load_pages()
    meta = []
    for i, pg in enumerate(pages, 1):
        if only and i not in only:
            continue
        bg, slots, size, warn, over = render(pg, i)
        bg.convert('RGB').save(os.path.join(out, 'bg', f'{i:03d}.jpg'), quality=90)
        full = preview(bg, slots).convert('RGB')
        os.makedirs(os.path.join(out, 'full'), exist_ok=True)
        full.save(os.path.join(out, 'full', f'{i:03d}.jpg'), quality=94, subsampling=0)
        full.resize((540, 960)).save(os.path.join(out, 'pv', f'{i:03d}.jpg'), quality=85)
        meta.append({'n': i, 'id': pg.get('id', ''), 'slots': slots, 'size': size, 'imgs': [dict(p) for p in PLACED]})
        if warn:
            print(f'! page {i} {pg.get("id","")} overflows by {over:.0f}px at size {size}')
    if NOSPRITE:   # one file per picture, the largest size it is used at
        os.makedirs(os.path.join(out, 'img'), exist_ok=True)
        names = {}
        for k, im in BIGGEST.items():
            fn = re.sub(r'[^\w-]+', '_', os.path.splitext(os.path.basename(k))[0])
            while fn in names.values():
                fn += '_'
            names[k] = fn + '.png'
            im.save(os.path.join(out, 'img', names[k]))
        json.dump(names, open(os.path.join(out, 'img_files.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
        print(len(names), 'separate pictures')
    json.dump(meta, open(os.path.join(out, 'slots.json' if not only else 'slots_part.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print(len(meta), 'pages rendered of', len(pages))
