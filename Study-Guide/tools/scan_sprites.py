"""Scan Assets_temp for pictures that can become clean 'in-game sprites'.

Class A: already has a transparent background (alpha) -> crop to the subject.
Class B: plain, even-coloured border -> background can be flood-filled away.
Everything else (photos, screenshots, busy scenes) is skipped.
Writes sprite_scan.json: [{path, w, h, cls, border_rgb}]
Usage: python scan_sprites.py <Assets_temp> <out.json>
"""
import json, os, sys
from PIL import Image

MIN_SIDE = 140


def border_pixels(im, step=4):
    w, h = im.size
    px = im.load()
    pts = []
    for x in range(0, w, step):
        pts += [px[x, 0], px[x, h - 1]]
    for y in range(0, h, step):
        pts += [px[0, y], px[w - 1, y]]
    return pts


def classify(path):
    try:
        im = Image.open(path)
        im.seek(0)
        w, h = im.size
    except Exception:
        return None
    if min(w, h) < MIN_SIDE:
        return {'w': w, 'h': h, 'cls': 'small'}
    has_alpha = im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info)
    if has_alpha:
        a = im.convert('RGBA').getchannel('A')
        hist = a.histogram()
        transparent = sum(hist[:16]) / (w * h)
        if transparent > 0.08:
            return {'w': w, 'h': h, 'cls': 'A', 'transparent': round(transparent, 2)}
    rgb = im.convert('RGB')
    small = rgb.copy()
    small.thumbnail((400, 400))
    pts = border_pixels(small)
    n = len(pts)
    mean = [sum(p[i] for p in pts) / n for i in range(3)]
    close = sum(1 for p in pts if sum(abs(p[i] - mean[i]) for i in range(3)) < 40) / n
    if close > 0.9:
        return {'w': w, 'h': h, 'cls': 'B', 'border_rgb': [round(m) for m in mean], 'even': round(close, 2)}
    return {'w': w, 'h': h, 'cls': 'C', 'even': round(close, 2)}


if __name__ == '__main__':
    root, out = sys.argv[1], sys.argv[2]
    res = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                p = os.path.join(dp, f)
                c = classify(p)
                if c:
                    c['path'] = os.path.relpath(p, root)
                    res.append(c)
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=0)
    from collections import Counter
    print(Counter((r['path'].split(os.sep)[0], r['cls']) for r in res))
