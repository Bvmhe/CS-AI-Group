"""Editable photos: new sprite-free background + every picture as its own Canva image + text sync.
Usage: python -I photo_ops.py <read-design dump> <first_guide_page> [a b]
Env: SLOTS (render slots.json with 'imgs'), IMGF (img_files.json), MEDIA (bg map), PICS (media_img.json).
Idempotent: skips backgrounds already swapped, pictures already on the page and text already equal.
Run it again after a pass so the newly added text boxes get position + format."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY = '#17324D'
dump, first = sys.argv[1], int(sys.argv[2])
bg = json.load(open(os.path.join(HERE, os.environ.get('MEDIA', 'media_bg7.json'))))
pics = json.load(open(os.path.join(HERE, os.environ.get('PICS', 'media_img.json'))))
imgf = json.load(open(os.environ['IMGF']))
pages_meta = {p['n']: p for p in json.load(open(os.environ['SLOTS'], encoding='utf-8'))}
d = json.loads(open(dump, encoding='utf-8').read())
pages = d['design_content']['pages']
a, b = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, len(pages))


def media_ids(el):
    out = []
    if isinstance(el, dict):
        for k, v in el.items():
            if k == 'mediaId':
                out.append(v)
            else:
                out += media_ids(v)
    elif isinstance(el, list):
        for v in el:
            out += media_ids(v)
    return out


ops = []
for k in range(a, b + 1):
    pg = pages[k - 1]
    pid, n = pg['id'], first + k - 1
    meta = pages_meta.get(n, {})
    els = pg.get('elements', [])
    rects = [e for e in els if e.get('type') == 'rect' and e.get('width', 0) >= 1079 and e.get('height', 0) >= 1919]
    if str(n) in bg and rects:
        r = rects[0]
        if bg[str(n)] not in media_ids(r):
            ops.append(dict(type='update_fill', locator_id=r['locator_id'], asset_type='image', asset_id=bg[str(n)],
                            alt_text=f'Study guide page {n} background'))
            ops.append(dict(type='crop_media', locator_id=r['locator_id'], top=0, left=0, width=1080, height=1920))
        if abs(r['top']) > 0.5 or abs(r['left']) > 0.5:   # background nudged by accident
            ops.append(dict(type='position_element', locator_id=r['locator_id'], top=0, left=0))
    elif str(n) in bg:
        print('no background rect on guide page', n, file=sys.stderr)
    # pictures: count what is already on the page so a rerun adds only the missing ones
    have = {}
    for e in els:
        if e is not (rects[0] if rects else None):
            for m in media_ids(e):
                have[m] = have.get(m, 0) + 1
    for im in meta.get('imgs', []):
        mid = pics[imgf[im['key']]]
        if have.get(mid, 0) > 0:
            have[mid] -= 1
            continue
        ops.append(dict(type='insert_fill', page_id=pid, asset_type='image', asset_id=mid,
                        alt_text=os.path.splitext(os.path.basename(im['key']))[0].replace('_', ' '), top=im['y'], left=im['x'], width=im['w'], height=im['h']))
    # text boxes
    texts = [e for e in els if e.get('type') == 'text']
    sl = meta.get('slots', [])
    for e, s in zip(texts, sl):
        L = e['locator_id']
        cur = ''.join(r.get('characters', '') for r in e.get('textRegions', []))
        if cur.strip() != s['text'].strip():
            ops.append(dict(type='replace_text', locator_id=L, text=s['text']))
        if abs(e['width'] - s['w']) > 1:
            ops.append(dict(type='resize_element', locator_id=L, width=round(s['w'])))
        if abs(e['top'] - s['y']) > 1 or abs(e['left'] - s['x']) > 1:
            ops.append(dict(type='position_element', locator_id=L, top=round(s['y']), left=round(s['x'])))
        f = (e.get('textRegions') or [{}])[0].get('formatting', {})
        if f.get('fontSize') != s['size'] or str(f.get('color', '')).lower() != NAVY.lower() or cur.strip() != s['text'].strip():
            ops.append(dict(type='format_text', locator_id=L,
                            formatting=dict(font_size=s['size'], color=NAVY, line_height=1.4, text_align='start')))
    for s in sl[len(texts):]:
        ops.append(dict(type='add_text', page_id=pid, text=s['text'], top=round(s['y']), left=round(s['x']), width=round(s['w'])))
    if len(texts) > len(sl):
        print('page', n, 'has', len(texts), 'text boxes but', len(sl), 'slots', file=sys.stderr)
print(len(ops), 'ops', file=sys.stderr)
out = json.dumps(ops, ensure_ascii=False, separators=(',', ':'))
if os.environ.get('OUT'):
    open(os.environ['OUT'], 'w', encoding='utf-8').write(out)
else:
    print(out)
