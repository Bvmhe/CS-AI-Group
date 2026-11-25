"""Render the Final guide into picture-per-page PDFs (1080x1920 JPEG per page) for Canva, each under 9.8 MB."""
import io, os
import pymupdf
from PIL import Image

BASE = r'C:\Projects\CS-AI-Group\Stage3-Final'
SRC = BASE + r'\FengweiFriedRice_Stage3_FinalStudyGuide.pdf'
OUT = BASE + r'\For-Canva-Final'
PARTS = [('Part0_Front', 1, 5), ('Part1_Recipe01', 6, 35), ('Part2_Recipe02', 36, 64), ('Part3_Recipe03', 65, 90),
         ('Part4_Recipe04', 91, 118), ('Part5_Recipe05', 119, 144), ('Part6_Bonus_Dessert', 145, 168)]
LIMIT = 9.8e6


def jpgs(doc, a, b, q):
    out = []
    for i in range(a - 1, b):
        pg = doc[i]
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(1080 / pg.rect.width, 1920 / pg.rect.height))
        im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q, optimize=True)
        out.append(buf.getvalue())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    doc = pymupdf.open(SRC)
    for name, a, b in PARTS:
        for q in (90, 86, 82, 78, 74, 70):
            data = jpgs(doc, a, b, q)
            new = pymupdf.open()
            for j in data:
                p = new.new_page(width=810, height=1440)
                p.insert_image(p.rect, stream=j)
            path = os.path.join(OUT, f'Final_Fengwei_Guide_{name}_p{a}-{b}.pdf')
            new.save(path, garbage=3, deflate=True)
            if os.path.getsize(path) < LIMIT:
                break
        print(f'{os.path.basename(path)}  q={q}  {os.path.getsize(path) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
