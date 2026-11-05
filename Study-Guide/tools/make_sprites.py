"""Cut sprite candidates out of their backgrounds and crop them tight.

Reads sprite_scan.json from scan_sprites.py. Class A keeps its own transparency;
class B gets its plain background flood-filled away from the edges.
Writes transparent PNGs to <outdir>/cand/NNNN.png plus contact sheets
(sheet_XX.png, numbered tiles on a checkerboard) for visual review, and
cand_index.json mapping number -> source path.
Usage: python make_sprites.py <Assets_temp> <sprite_scan.json> <outdir>
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MAGIC = (1, 254, 3)
MAXSIDE = 900


def cut_b(im):
    rgb = im.convert('RGB')
    rgb.thumbnail((MAXSIDE, MAXSIDE))
    w, h = rgb.size
    a = np.asarray(rgb).astype(int)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    seeds = [(x, 0) for x in range(0, w, 6)] + [(x, h - 1) for x in range(0, w, 6)] + \
            [(0, y) for y in range(0, h, 6)] + [(w - 1, y) for y in range(0, h, 6)]
    work = rgb.copy()
    px = work.load()
    for s in seeds:
        c = px[s]
        if c == MAGIC:
            continue
        if sum(abs(c[i] - bg[i]) for i in range(3)) < 60:
            ImageDraw.floodfill(work, s, MAGIC, thresh=38)
    m = (np.asarray(work) == np.array(MAGIC)).all(axis=2)
    alpha = Image.fromarray(((~m) * 255).astype('uint8'))
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    out = rgb.convert('RGBA')
    out.putalpha(alpha)
    return out


def cut_a(im):
    out = im.convert('RGBA')
    out.thumbnail((MAXSIDE, MAXSIDE))
    return out


def crop(out, pad=6):
    a = np.asarray(out.getchannel('A'))
    ys, xs = np.where(a > 40)
    if len(xs) == 0:
        return None, 0
    frac = (a > 40).mean()
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    box = (max(0, x0 - pad), max(0, y0 - pad), min(out.width, x1 + pad + 1), min(out.height, y1 + pad + 1))
    return out.crop(box), frac


def checker(size, cell=16):
    bg = Image.new('RGB', size, (235, 235, 235))
    d = ImageDraw.Draw(bg)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(205, 205, 205))
    return bg


if __name__ == '__main__':
    root, scan, outdir = sys.argv[1:4]
    os.makedirs(os.path.join(outdir, 'cand'), exist_ok=True)
    items = [r for r in json.load(open(scan, encoding='utf-8')) if r['cls'] in ('A', 'B')]
    items.sort(key=lambda r: r['path'])
    index, tiles = {}, []
    n = 0
    for r in items:
        try:
            im = Image.open(os.path.join(root, r['path']))
            im.seek(0)
            out = cut_a(im) if r['cls'] == 'A' else cut_b(im)
            sp, frac = crop(out)
        except Exception as e:
            print('fail', r['path'], e)
            continue
        if sp is None or frac < 0.06 or frac > 0.97 or min(sp.size) < 90:
            continue
        n += 1
        name = f'{n:04d}'
        sp.save(os.path.join(outdir, 'cand', name + '.png'))
        index[name] = {'src': r['path'], 'cls': r['cls'], 'size': sp.size, 'fill': round(float(frac), 2)}
        tiles.append((name, sp))
    json.dump(index, open(os.path.join(outdir, 'cand_index.json'), 'w', encoding='utf-8'), indent=0)
    T, cols, per = 190, 8, 48
    font = ImageFont.load_default()
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * T, rows * (T + 18)), 'white')
        for i, (name, sp) in enumerate(chunk):
            cell = checker((T - 6, T - 6))
            t = sp.copy(); t.thumbnail((T - 6, T - 6))
            cell.paste(t, ((T - 6 - t.width) // 2, (T - 6 - t.height) // 2), t)
            x, y = (i % cols) * T, (i // cols) * (T + 18)
            sheet.paste(cell, (x + 3, y + 3))
            ImageDraw.Draw(sheet).text((x + 6, y + T - 2), name, fill='black', font=font)
        sheet.save(os.path.join(outdir, f'sheet_{s // per + 1:02d}.png'))
    print(n, 'sprites,', (len(tiles) + per - 1) // per, 'sheets')
