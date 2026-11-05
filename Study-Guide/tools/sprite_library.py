"""Copy the hand-picked, cleanly cut sprites into Study-Guide/assets/sprites/<name>.png.
'c' = from the cut-out candidates (characters, ingredients, stickers), 'd' = dish cut off its tablecloth.
Source ids refer to cand_index.json (which maps each id back to its Assets_temp file).
Usage: python sprite_library.py <sprites_work_dir>
"""
import json, os, shutil, sys

PICKS = {
    # Mama
    'mama_happy': 'c0027', 'mama_cheer': 'c0021', 'mama_grin': 'c0064', 'mama_fire': 'c0066',
    'mama_fry': 'c0056', 'mama_basket': 'c0030', 'mama_bunny_roses': 'c0031', 'mama_bunny': 'c0032',
    'mama_small': 'c0052', 'mama_ok': 'c0063', 'mama_stand': 'c0424', 'mama_book': 'c0006',
    'mama_tray': 'c0408', 'mama_icon_egg': 'c0372', 'mama_icon_cookie': 'c0373', 'mama_icon_soup': 'c0374',
    'mama_icon_sandwich': 'c0378', 'mama_icon_shake': 'c0379', 'mama_icon_cake': 'c0380',
    'mama_icon_veg': 'c0381', 'mama_icon_lasagna': 'c0383', 'mama_icon_macaron': 'c0390',
    'mama_icon_cheer': 'c0398', 'mama_icon_pan': 'c0402', 'mama_icon_hearts': 'c0399', 'mama_icon_bowl': 'c0385',
    # family & friends
    'papa': 'c0024', 'papa_suit': 'c0025', 'grandpa': 'c0010', 'grandma': 'c0009', 'pup': 'c0026',
    'boy_red': 'c0015', 'boy_grin': 'c0028', 'boy_blond': 'c0029', 'girl_overalls': 'c0036',
    'girl_apron': 'c0038', 'girl_pigtails': 'c0039', 'girl_skirt': 'c0040', 'girl_happy': 'c0035',
    'girl_jump': 'c0037', 'boy_glasses': 'c0041', 'girl_bow': 'c0042', 'girl_blonde': 'c0043',
    'boy_snow': 'c0044', 'boy_snow2': 'c0045', 'baby_crawl': 'c0050', 'baby_toy': 'c0065', 'baby': 'c0067',
    'explorer': 'c0016', 'explorers': 'c0017', 'chef_old': 'c0001', 'lady_sari': 'c0003',
    'girl_uk': 'c0004', 'girl_goggles': 'c0020', 'lady_hat': 'c0022', 'detective': 'c0034',
    'lady_flamenco': 'c0051', 'boy_blue': 'c0053', 'girl_qipao': 'c0054', 'girl_blonde_smile': 'c0055',
    'girl_cross': 'c0058', 'girl_red': 'c0059', 'boy_laugh': 'c0060', 'man_laugh': 'c0061',
    'boy_cap': 'c0062', 'bunny': 'c0008', 'girl_red_dress': 'c0524',
    # stickers & tools
    'sticker_tasty': 'c0068', 'sticker_notgood': 'c0069', 'timer': 'c0282',
    # ingredients
    'carrot': 'c0248', 'cherries': 'c0249', 'golden_apple': 'c0250', 'tomato': 'c0252', 'apple': 'c0253',
    'peppers': 'c0254', 'cabbage': 'c0255', 'cucumber': 'c0257', 'eggplant': 'c0258', 'melon': 'c0259',
    'mint': 'c0260', 'onion': 'c0261', 'peach': 'c0262', 'pear': 'c0263', 'pineapple': 'c0264',
    'pumpkin': 'c0265', 'tomatoes': 'c0266', 'watermelon': 'c0267',
    # dishes
    'salad': 'd0072', 'soup': 'd0073', 'steak_plate': 'd0074', 'pie': 'd0075', 'unagi_box': 'd0076',
    'roast': 'd0077', 'cookies': 'd0078', 'panna_cotta': 'd0079', 'waffle': 'd0080', 'hotdog': 'd0081',
    'meatloaf': 'd0082', 'sweet_potato': 'd0083', 'fish_plate': 'd0084', 'kinpira': 'd0085',
    'tempura': 'd0086', 'bagels': 'd0087', 'chirashi_bowl': 'd0088', 'dorayaki': 'd0089', 'dango': 'd0090',
    'macarons': 'd0092', 'kimchi': 'd0093', 'bread': 'd0094', 'churros': 'd0096', 'sushi_rolls': 'd0097',
    'mapo': 'd0098', 'ice_cream': 'd0101', 'mama_cookies': 'd0102', 'yakiniku': 'd0105',
    'pepper_stirfry': 'd0106', 'goya_stirfry': 'd0109', 'shrimp': 'd0110', 'milk_drink': 'd0111',
    'melon_soda': 'd0112', 'pink_drink': 'd0113', 'orange_drink': 'd0115', 'tartare': 'd0118',
    'roll_cake': 'd0119', 'congee': 'd0120', 'zongzi': 'd0122', 'croquettes': 'd0123', 'nabe': 'd0124',
    'buns': 'd0125', 'popcorn': 'd0127', 'kids_plate': 'd0128', 'shrimp_mayo': 'd0129', 'stew': 'd0131',
    'pita': 'd0132', 'fried_rice': 'd0133', 'pasta': 'd0134', 'heart_cake': 'd0136', 'chive_pancake': 'd0137',
    'fish_fry_plate': 'd0138', 'xiaolongbao': 'd0139', 'soup_noodles': 'd0140', 'donuts': 'd0141',
    'chicken': 'd0142', 'yakisoba': 'd0143', 'dessert_plate': 'd0144', 'mantou': 'd0145',
    'pancakes': 'd0146', 'sesame_balls': 'd0147', 'takoyaki': 'd0148', 'pastry': 'd0149',
    'bibimbap': 'd0150', 'veg_noodles': 'd0151', 'quiche': 'd0152', 'sweet_sour': 'd0153', 'bento': 'd0155',
}

if __name__ == '__main__':
    work = sys.argv[1]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = os.path.join(here, 'assets', 'sprites')
    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(os.path.join(work, 'cand_index.json'), encoding='utf-8'))
    credits = {}
    for name, ref in PICKS.items():
        src = os.path.join(work, 'cand' if ref[0] == 'c' else 'dish', ref[1:] + '.png')
        shutil.copy(src, os.path.join(dst, name + '.png'))
        credits[name] = idx[ref[1:]]['src']
    json.dump(credits, open(os.path.join(dst, '_sources.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(len(PICKS), 'sprites')
