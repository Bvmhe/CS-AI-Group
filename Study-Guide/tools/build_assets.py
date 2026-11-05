"""Build the image set for the 'Hot Wok, Smart Chips' study guide.

Reads the raw pictures in Assets_temp/ and writes web-sized copies to Study-Guide/assets/:
  hosts/   hand-picked character and scene images (named by role)
  book/    Recipe Book collage pages (many dish screenshots per page)
Run from the repo root:  python Study-Guide/tools/build_assets.py
"""
import json, os, re
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'Assets_temp')
OUT = os.path.join(ROOT, 'Study-Guide', 'assets')
CH = os.path.join(RAW, '01_Characters')
FD = os.path.join(RAW, '02_Food_and_Recipes')
MG = os.path.join(RAW, '03_Cooking_Techniques_and_Minigames')
EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp')


def listing(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))


def load(path):
    im = Image.open(path)
    im = next(ImageSequence.Iterator(im))
    return im.convert('RGBA')


# --- 1. hand-picked hosts and scenes (index numbers refer to the sorted listing of each folder) ---
CHAR_PICKS = {
    'mama_basket': 192, 'mama_bunny_roses': 193, 'mama_pan_flip': 285, 'mama_cheer': 353,
    'mama_arms_crossed': 162, 'mama_proud': 127, 'mama_small_proud': 338, 'mama_fire_eyes': 359,
    'mama_cry': 358, 'papa_casual': 136, 'papa_suit': 138, 'pup_chef': 161, 'kid_boy': 166,
    'kid_boy_big': 96, 'family_circle': 120, 'family_hotspring': 125, 'family_coins': 126,
    'family_table': 128, 'family_wide': 4, 'mama_and_pup': 300, 'mama_pup_kids': 215,
    'mama_papa_hug': 247, 'grandma': 36, 'grandpa': 37, 'baby_girl': 33, 'mama_face_happy': 99,
    'wok_dish': 65, 'kids_peeking': 6, 'kids_world': 113, 'mama_kid_kitchen': 159,
}
MG_PICKS = {
    'prep_table_1': 1463, 'prep_table_2': 1468, 'prep_table_3': 1476, 'prep_table_4': 1143,
    'prep_table_5': 1193, 'banner_delicious': 200, 'banner_well_done': 201, 'sun_happy': 202,
    'dish_fried_rice': 69, 'dish_bento': 196, 'dish_steak': 44, 'dish_curry': 42,
    'dish_dumplings': 46, 'dish_gyoza_plate': 306, 'dish_feast_cake': 146, 'dish_pizza': 307,
    'dish_noodles': 49, 'dish_onigiri': 97, 'dish_ramen': 235, 'dish_sushi': 304,
    'dish_dog_cookies': 125, 'dish_mama_omelette': 270, 'dish_tempura': 64, 'dish_mapo': 245,
}


def save_pick(src, name, maxsize=1400):
    im = load(src)
    im.thumbnail((maxsize, maxsize))
    has_alpha = im.getextrema()[3][0] < 255
    os.makedirs(os.path.join(OUT, 'hosts'), exist_ok=True)
    if has_alpha:
        im.save(os.path.join(OUT, 'hosts', name + '.png'), optimize=True)
    else:
        im.convert('RGB').save(os.path.join(OUT, 'hosts', name + '.jpg'), quality=85)


def build_picks():
    credits = []
    for folder, picks in ((CH, CHAR_PICKS), (MG, MG_PICKS)):
        files = listing(folder)
        for name, idx in picks.items():
            save_pick(os.path.join(folder, files[idx]), name)
            credits.append({'asset': name, 'source': os.path.relpath(os.path.join(folder, files[idx]), ROOT)})
    return credits


# --- 2. Recipe Book: auto-pick large dish screenshots, drop near-duplicates, lay out collages ---
def ahash(im, n=12):
    g = im.convert('L').resize((n, n))
    px = list(g.getdata()); avg = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def dish_candidates():
    out = []
    for folder in (FD, MG):
        for f in listing(folder):
            if re.search(r'icon|logo|cover|box|title|screenshot|menu', f, re.I):
                continue
            p = os.path.join(folder, f)
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            if w >= 1000 and 1.15 <= w / h <= 1.9:
                out.append(p)
    return out


def build_book(per_page=24, cols=6, max_pages=12):
    seen, kept = [], []
    for p in dish_candidates():
        try:
            im = load(p)
        except Exception:
            continue
        h = ahash(im)
        if any(bin(h ^ s).count('1') <= 10 for s in seen):
            continue
        seen.append(h); kept.append(p)
        if len(kept) >= per_page * max_pages:
            break
    os.makedirs(os.path.join(OUT, 'book'), exist_ok=True)
    W, H, pad, top = 1920, 1080, 18, 150
    cw = (W - pad * (cols + 1)) // cols
    rows = per_page // cols
    chh = (H - top - pad * (rows + 1)) // rows
    pages = 0
    for s in range(0, len(kept), per_page):
        sheet = Image.new('RGB', (W, H), (255, 246, 232))
        d = ImageDraw.Draw(sheet)
        for i, p in enumerate(kept[s:s + per_page]):
            im = load(p).convert('RGB')
            im.thumbnail((cw, chh))
            x = pad + (i % cols) * (cw + pad) + (cw - im.size[0]) // 2
            y = top + pad + (i // cols) * (chh + pad) + (chh - im.size[1]) // 2
            d.rounded_rectangle((x - 6, y - 6, x + im.size[0] + 6, y + im.size[1] + 6), 14, fill='white', outline=(240, 120, 150), width=4)
            sheet.paste(im, (x, y))
        pages += 1
        sheet.save(os.path.join(OUT, 'book', f'recipe_book_{pages:02d}.jpg'), quality=82)
    return kept, pages


if __name__ == '__main__':
    credits = build_picks()
    kept, pages = build_book()
    json.dump({'hosts': credits, 'recipe_book_sources': [os.path.relpath(p, ROOT) for p in kept]},
              open(os.path.join(OUT, 'asset_sources.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(credits), 'host images;', len(kept), 'dishes on', pages, 'Recipe Book pages')
