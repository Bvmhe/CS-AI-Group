"""Small helpers so the page modules read like a script."""

FOODS = {
    'carrot', 'cherries', 'golden_apple', 'tomato', 'apple', 'peppers', 'cabbage', 'cucumber', 'eggplant', 'melon',
    'mint', 'onion', 'peach', 'pear', 'pineapple', 'pumpkin', 'tomatoes', 'watermelon', 'salad', 'soup',
    'steak_plate', 'pie', 'unagi_box', 'roast', 'cookies', 'panna_cotta', 'waffle', 'hotdog', 'meatloaf',
    'sweet_potato', 'fish_plate', 'kinpira', 'tempura', 'bagels', 'chirashi_bowl', 'dorayaki', 'dango', 'macarons',
    'kimchi', 'bread', 'churros', 'sushi_rolls', 'mapo', 'ice_cream', 'mama_cookies', 'yakiniku', 'pepper_stirfry',
    'goya_stirfry', 'shrimp', 'milk_drink', 'melon_soda', 'pink_drink', 'orange_drink', 'tartare', 'roll_cake',
    'congee', 'zongzi', 'croquettes', 'nabe', 'buns', 'popcorn', 'kids_plate', 'shrimp_mayo', 'stew', 'pita',
    'fried_rice', 'pasta', 'heart_cake', 'chive_pancake', 'fish_fry_plate', 'xiaolongbao', 'soup_noodles', 'donuts',
    'chicken', 'yakisoba', 'dessert_plate', 'mantou', 'pancakes', 'sesame_balls', 'takoyaki', 'pastry', 'bibimbap',
    'veg_noodles', 'quiche', 'sweet_sour', 'bento', 'timer', 'sticker_tasty', 'sticker_notgood',
}


def P(ch, pill, *blocks, floor=None, deco=None, id='', **kw):
    d = {'id': id, 'ch': ch, 'pill': pill, 'blocks': list(blocks)}
    if floor:
        d['floor'] = floor
    if deco:
        d['deco'] = deco
    d.update(kw)
    return d


def W(title, text, color='yellow', icon=None, **kw):
    return dict(kind='win', title=title, text=text, color=color, icon=icon, **kw)


def N(text, label='🔬 REAL TECH', color='mint', **kw):
    return dict(kind='note', text=text, label=label, color=color, **kw)


def PAPA(text):
    return N(text, label='⚠ PAPA SAYS…', color='yellow')


def TIP(text):
    return N(text, label="💡 MAMA'S TIP", color='pink')


def KEY(text):
    return N(text, label='🔑 KEY IDEA', color='blue')


def B(sprite, text, side='left', **kw):
    return dict(kind='bubble', sprite=sprite, text=text, side=side, **kw)


def T(cols, rows, widths=None, color='blue', fs=24, **kw):
    if widths is None:
        n = len(cols)
        widths = [960 // n] * n
    return dict(kind='table', cols=cols, rows=rows, widths=widths, color=color, fs=fs, **kw)


def S(*names, h=200):
    return dict(kind='sprites', items=list(names), h=h)


def PIC(path, caption='', h=560):
    """An example picture from the repo (path relative to the repo root), framed like a photo."""
    return dict(kind='photo', path=path, caption=caption, h=h)


def C(code):
    return dict(kind='code', text=code)


def R(*items, gap=30):
    return dict(kind='row', items=list(items), gap=gap)


def BIG(text, color='red'):
    return dict(kind='big', text=text, color=color)


def TITLE(text, size=96, color='yellow'):
    return dict(kind='title', text=text, size=size, color=color)


def activity(ch, code, name, goal, prep, task, observe, reflect, feedback, floor1=None, floor2=None, host='mama_happy'):
    """A mini-game in the six-part format, over two pages."""
    p1 = P(ch, f'MINI-GAME {code} · {name}',
           W('Goal', goal, 'mint', '🎯'),
           W('Prep', prep, 'yellow', '🧺'),
           W('Task', task, 'pink', '🔪'),
           floor=floor1 or [host, 'timer'], id=f'mg{code}a')
    p2 = P(ch, f'MINI-GAME {code} · FEEDBACK',
           W('Observe', observe, 'blue', '👀'),
           W('Reflect', reflect, 'lav', '💭'),
           W("Mama's Feedback", feedback, 'orange', '⭐'),
           floor=floor2 or ['mama_cheer', 'sticker_tasty'], id=f'mg{code}b')
    return [p1, p2]


def glossary(ch, rows, color='red', part=''):
    return P(ch, f'CHAPTER {ch} GLOSSARY{part}', T(['Glossary', 'Definition'], rows, [300, 660], color=color),
             floor=['mama_book', 'girl_happy'], id=f'g{ch}{part}')
