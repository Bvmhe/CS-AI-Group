"""Cut a dish (plate + food) out of a Cooking Mama 'result card' (dish on a blue gingham cloth).

Idea: gingham = light, bluish or white pixels. Everything else (food, wooden coasters, pots) is
'foreground'. The plate is a big white ring: the largest white component after an opening.
Keep the plate (holes filled) plus any foreground blob that touches it or sits in the centre.
Usage: python cut_dishes.py <cand_dir> <out_dir> <id> [<id> ...]   (ids like 0072)
"""
import os, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi


def cut(path):
    im = Image.open(path).convert('RGBA')
    a = np.asarray(im).astype(float)
    rgb, al = a[..., :3] / 255.0, a[..., 3]
    mx, mn = rgb.max(2), rgb.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bluish = (b >= r) & (b >= g - 0.02) & (b - r > 0.04)
    gingham = (al > 0) & (mx > 0.62) & (bluish | (sat < 0.10))
    white = (al > 0) & (sat < 0.10) & (mx > 0.80)
    fg = (al > 0) & ~gingham
    h, w = fg.shape
    # plate: biggest white blob after opening (cuts thin links to cloth squares)
    wo = ndi.binary_opening(white, iterations=3)
    lab, n = ndi.label(wo)
    plate = np.zeros_like(fg)
    if n:
        sizes = ndi.sum(wo, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        ys, xs = np.where(lab == k)
        if xs.max() - xs.min() > 0.3 * w:
            plate = ndi.binary_fill_holes(ndi.binary_closing(lab == k, iterations=6))
    # the card's own decorations: bottles (top-left), basket (top-right), striped side edges
    fg[:int(h * .32), int(w * .76):] = False
    fg[:int(h * .30), :int(w * .20)] = False
    fg[:, :int(w * .035)] = False
    fg[:, int(w * .965):] = False
    fg = ndi.binary_opening(fg, iterations=2)
    centre = np.zeros_like(fg)
    centre[int(h * .3):int(h * .8), int(w * .25):int(w * .75)] = True
    seed = plate | (fg & centre)
    lab, n = ndi.label(fg | plate)
    keep = np.zeros_like(fg)
    for k in range(1, n + 1):
        comp = lab == k
        if (comp & seed).any() and comp.sum() > 0.01 * h * w:
            keep |= comp
    keep = ndi.binary_fill_holes(keep)
    keep = ndi.binary_opening(keep, iterations=2)
    alpha = Image.fromarray((keep * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(1.0))
    out = im.copy()
    out.putalpha(Image.fromarray(np.minimum(np.asarray(alpha), al.astype('uint8'))))
    bb = out.getchannel('A').point(lambda v: 255 if v > 40 else 0).getbbox()
    return out.crop(bb) if bb else None


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for i in sys.argv[3:]:
        o = cut(os.path.join(src, i + '.png'))
        if o is not None:
            o.save(os.path.join(dst, i + '.png'))
