"""Turn page specs into Canva edit-design operations.

Usage: python canva_fill.py <design_content.json> <specs_module> <uploads.jsonl> <outdir> [first_page]
Reads the design JSON returned by Canva read-design (with an open transaction), matches each page
to its spec (by position), and writes one JSON list of operations per page to <outdir>/pNN.json.
"""
import importlib.util, json, math, os, sys

CHAR_W = 0.53   # average glyph width / font size for the template body font
LINE_H = 1.32


def load_specs(path):
    spec = importlib.util.spec_from_file_location('specs', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.PAGES


def media_ids(path):
    ids = {}
    for line in open(path, encoding='utf-8-sig'):
        line = line.strip()
        if line:
            d = json.loads(line); ids[d['name']] = d['resp']['mediaId']
    return ids


def text_of(el):
    return ''.join(r.get('characters', '') for r in el.get('textRegions', []))


def est_lines(txt, width, size):
    per = max(1, int(width / (CHAR_W * size)))
    return sum(max(1, math.ceil(len(p) / per)) for p in txt.split('\n'))


def fit_size(txt, width, height, start=25, lo=15):
    for s in range(start, lo - 1, -1):
        if est_lines(txt, width, s) * s * LINE_H <= height:
            return s, True
    return lo, False


def slots(page):
    top_texts = [e for e in page['elements'] if e['type'] == 'text']
    groups = []
    for e in page['elements']:
        if e['type'] == 'group':
            kids = e.get('elements') or e.get('children') or []
            shapes = [k for k in kids if k['type'] == 'shape']
            texts = sorted([k for k in kids if k['type'] == 'text'], key=lambda k: k['top'])
            box = max(shapes, key=lambda k: k['width'] * k['height']) if shapes else None
            groups.append({'group': e, 'box': box, 'texts': texts, 'shapes': shapes})
    others = [e for e in page['elements'] if e['type'] not in ('text', 'group')]
    return top_texts, groups, others


def build(page, sp, ids):
    ops = []
    pid = page['id']
    texts, groups, others = slots(page)
    tset = sp.get('top', {})           # index of top-level text -> new text ("" deletes)
    for i, el in enumerate(texts):
        if i in tset:
            v = tset[i]
            if v is None or v == '':
                ops.append({'type': 'delete_element', 'locator_id': el['locator_id']})
            else:
                ops.append({'type': 'replace_text', 'locator_id': el['locator_id'], 'text': v})
                fs = sp.get('top_size', {}).get(i)
                if fs:
                    ops.append({'type': 'format_text', 'locator_id': el['locator_id'], 'formatting': {'font_size': fs}})
    # optional re-layout of groups: list of [left, top, width, height] per card
    layout = sp.get('layout')
    cards = sp.get('cards', [])
    for gi, g in enumerate(groups):
        if gi >= len(cards) or cards[gi] is None:
            if sp.get('drop_unused', True):
                ops.append({'type': 'delete_element', 'locator_id': g['group']['locator_id']})
            continue
        head, body = cards[gi]
        box = g['box']; ts = g['texts']
        gx, gy = g['group']['left'], g['group']['top']
        bx = dict(left=box['left'], top=box['top'], width=box['width'], height=box['height'])
        if layout and gi < len(layout) and layout[gi]:
            L, T, W, H = layout[gi]
            if (W, H) != (box['width'], box['height']):   # resize BEFORE moving the group (Canva quirk)
                ops.append({'type': 'resize_element', 'locator_id': box['locator_id'], 'width': W, 'height': H})
            if (L, T) != (gx, gy):
                ops.append({'type': 'position_element', 'locator_id': g['group']['locator_id'], 'left': L, 'top': T})
            dx, dy = L - gx, T - gy
            bx = dict(left=box['left'] + dx, top=box['top'] + dy, width=W, height=H)
        else:
            dx = dy = 0
        right_edge = bx['left'] + bx['width'] - 28
        if len(ts) >= 1:
            h = ts[0]
            ops.append({'type': 'replace_text', 'locator_id': h['locator_id'], 'text': head})
            hw = right_edge - (h['left'] + dx)
            if abs(hw - h['width']) > 60:
                ops.append({'type': 'resize_element', 'locator_id': h['locator_id'], 'width': hw})
        if len(ts) >= 2:
            b = ts[1]
            bw = right_edge - (b['left'] + dx)
            btop = b['top'] + dy
            avail = bx['top'] + bx['height'] - btop - 24
            if abs(bw - b['width']) <= 60:
                bw = b['width']
            size, ok = fit_size(body, bw, avail, start=sp.get('body_max', 25))
            ops.append({'type': 'replace_text', 'locator_id': b['locator_id'], 'text': body})
            ops.append({'type': 'format_text', 'locator_id': b['locator_id'], 'formatting': {'font_size': size, 'line_height': 1.25}})
            if bw != b['width']:
                ops.append({'type': 'resize_element', 'locator_id': b['locator_id'], 'width': bw})
            if not ok:
                print(f"  ! overflow risk page {sp.get('name')} card {gi}: {len(body)} chars", file=sys.stderr)
        for extra in ts[2:]:
            ops.append({'type': 'delete_element', 'locator_id': extra['locator_id']})
    for o in others:
        if o['type'] in sp.get('delete_types', ['line']):
            ops.append({'type': 'delete_element', 'locator_id': o['locator_id']})
    for img in sp.get('imgs', []):
        name, L, T, W, H = img
        ops.append({'type': 'insert_fill', 'page_id': pid, 'asset_type': 'image', 'asset_id': ids[name],
                    'alt_text': name.replace('_', ' '), 'left': L, 'top': T, 'width': W, 'height': H})
    # Canva quirk: shape resizes only stick when they come after all text edits on the page.
    head = [o for o in ops if o['type'] == 'position_element']
    tail = [o for o in ops if o['type'] == 'resize_element' and o.get('height')]
    mid = [o for o in ops if o not in head and o not in tail]
    return head + mid + tail


if __name__ == '__main__':
    design, specs_path, uploads, outdir = sys.argv[1:5]
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    d = json.load(open(design, encoding='utf-8'))
    pages = d['design_content']['pages']
    specs = load_specs(specs_path)
    ids = media_ids(uploads)
    os.makedirs(outdir, exist_ok=True)
    for k, sp in enumerate(specs):
        idx = first + k
        page = pages[idx - 1]
        ops = build(page, sp, ids)
        json.dump(ops, open(os.path.join(outdir, f'p{idx:03d}.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        print(idx, sp.get('name'), len(ops), 'ops')
