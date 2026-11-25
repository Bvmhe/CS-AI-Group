"""Make the Final Test Bank from the V1.0 paper edition.

Changes: cover pill "STAGE 2 TEST BANK" -> "FINAL TEST BANK", footer and document title,
Q49 section reference 5.4 -> 5.1 (MFU is taught in 5.1). Everything else is kept byte for byte.
"""
import io, re, zipfile
from PIL import Image, ImageDraw, ImageFont

BASE = r'C:\Projects\CS-AI-Group\Stage3-Final'
SRC = BASE + r'\V1-from-GitHub\FengweiFriedRice_Stage2_TestBank_Paper.docx'
OUT = BASE + r'\FengweiFriedRice_Stage3_FinalTestBank.docx'
FONT = BASE + r'\tools\FredokaBold.ttf'


def new_cover(png):
    im = Image.open(io.BytesIO(png)).convert('RGBA')
    px = im.convert('RGB')
    ink = min((px.getpixel((x, y)) for x in range(455, 800) for y in range(88, 116)), key=sum)
    d = ImageDraw.Draw(im)
    d.rectangle((446, 79, 1158, 123), fill=px.getpixel((440, 100)))
    text = 'FENGWEI FRIED RICE · FINAL TEST BANK'
    size = 38
    f = ImageFont.truetype(FONT, size)
    while d.textlength(text, font=f) > 690:
        size -= 1; f = ImageFont.truetype(FONT, size)
    w = d.textlength(text, font=f)
    l, t, r, b = d.textbbox((0, 0), 'FENGWEI', font=f)
    d.text((800 - w / 2, 101.5 - (t + b) / 2), text, font=f, fill=ink)
    out = io.BytesIO(); im.save(out, 'PNG', optimize=True)
    im.save(BASE + r'\tb\cover_final.png')
    return out.getvalue()


def main():
    zin = zipfile.ZipFile(SRC)
    zout = zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED)
    for info in zin.infolist():
        data = zin.read(info.filename)
        n = info.filename
        if n == 'word/media/image2.png':
            data = new_cover(data)
        elif n in ('word/footer1.xml', 'word/header1.xml', 'docProps/core.xml'):
            x = data.decode('utf8')
            x = x.replace('Stage 2 Test Bank', 'Final Test Bank')
            data = x.encode('utf8')
        elif n == 'word/document.xml':
            x = data.decode('utf8')
            # Q49 question header: the run after "Q49  " / "MULTIPLE CHOICE" holds "·   §5.4  ·  Understand"
            i = x.find('>Q49  <')
            j = x.find('§5.4', i)
            assert i > 0 and 0 < j - i < 3000, (i, j)
            x = x[:j] + '§5.1' + x[j + 4:]
            assert x.count('Stage 2') == 0
            data = x.encode('utf8')
        zout.writestr(info, data)
    zout.close()
    print('wrote', OUT)


if __name__ == '__main__':
    main()
