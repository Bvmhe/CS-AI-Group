"""Render the ```mermaid blocks of the chapter .md files to PNG with headless Edge (no Node needed).
Usage: python mermaid_png.py <out_dir> [chapter_number ...]
Writes <out_dir>/chNN_KK.png (KK = order of the block in the chapter) plus index.json (code + line + nearby heading)."""
import glob, html, json, os, re, subprocess, sys, tempfile
from PIL import Image, ImageChops

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
PROXY = '--proxy-server=http://127.0.0.1:7890'
# Cooking Mama palette, the same colours as the guide pages (render_pages.COL): yellow, mint, pink, blue, lavender, orange
_PAL = ['#FFC94A', '#8FD5C0', '#FF8DA1', '#7FB8EA', '#BEAAF0', '#FFA64D']
_CYCLE = ''.join(f'.node:nth-of-type(6n+{i + 1}) :is(rect,polygon,circle,ellipse,path.basic){{fill:{c} !important}}' for i, c in enumerate(_PAL))
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&display=block" rel="stylesheet">
<style>body{margin:0;background:#fff;font-family:Fredoka,Arial} #d{display:inline-block;padding:24px} #d svg{transform:scale(2.5);transform-origin:0 0}
#d svg .node :is(rect,polygon,circle,ellipse,path.basic){stroke:#17324D !important;stroke-width:3px !important}
#d svg .node rect{rx:12px;ry:12px}
#d svg .node, #d svg rect.actor, #d svg g.plot{filter:drop-shadow(3px 4px 0 #17324D)}
#d svg .cluster rect{fill:#FFF4DC !important;stroke:#17324D !important;stroke-width:2.5px !important;stroke-dasharray:8 5;rx:14px;ry:14px}
#d svg .cluster text, #d svg .cluster-label text{font-weight:600}
#d svg text, #d svg tspan{fill:#17324D !important;font-family:Fredoka,Arial !important;font-weight:500}
#d svg .flowchart-link{stroke:#17324D !important;stroke-width:3px !important}
#d svg marker path{fill:#17324D !important;stroke:#17324D !important}
#d svg .edgeLabel rect, #d svg .labelBkg{fill:#FFF4DC !important;opacity:1 !important}
#d svg rect.actor{fill:#FF8DA1 !important;stroke:#17324D !important;stroke-width:3px !important}
#d svg .messageLine0, #d svg .messageLine1{stroke:#17324D !important;stroke-width:2.5px !important}
#d svg .note{fill:#FFC94A !important;stroke:#17324D !important}
#d svg g.plot path{stroke-width:5px !important}
#d svg g.plot rect{stroke:#17324D !important;stroke-width:2px !important}
""" + _CYCLE + """</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script></head>
<body><div id="d"><pre class="mermaid">%s</pre></div>
<script>mermaid.initialize({startOnLoad:false,htmlLabels:false,flowchart:{useMaxWidth:false,htmlLabels:false,curve:'basis',nodeSpacing:45,rankSpacing:55,padding:14},sequence:{useMaxWidth:false},
theme:'base',themeVariables:{primaryColor:'#FFC94A',primaryBorderColor:'#17324D',primaryTextColor:'#17324D',lineColor:'#17324D',
secondaryColor:'#8FD5C0',tertiaryColor:'#FFF4DC',fontFamily:'Fredoka, Arial',fontSize:'18px',
xyChart:{plotColorPalette:'#FF8DA1, #7FB8EA, #8FD5C0, #FFC94A',backgroundColor:'#FFFFFF',titleColor:'#17324D',xAxisLabelColor:'#17324D',yAxisLabelColor:'#17324D',xAxisLineColor:'#17324D',yAxisLineColor:'#17324D'}}});
document.fonts.load('500 18px Fredoka').then(() => document.fonts.ready).then(() => mermaid.run());</script>
</body></html>"""


def blocks(md):
    lines = open(md, encoding='utf-8').read().split('\n')
    head, out, i = '', [], 0
    while i < len(lines):
        if lines[i].startswith('#'):
            head = lines[i].lstrip('#').strip()
        if lines[i].strip().startswith('```mermaid'):
            j = i + 1
            while not lines[j].strip().startswith('```'):
                j += 1
            out.append(dict(line=i + 1, heading=head, code='\n'.join(lines[i + 1:j])))
            i = j
        i += 1
    return out


def to_tb(code):
    """Top-to-bottom layout for diagrams too wide for a portrait page."""
    return re.sub(r'direction\s+LR', 'direction TB', re.sub(r'flowchart\s+LR', 'flowchart TB', code, count=1))


def shot(code, png):
    code = re.sub(r'</?(b|i|strong|em)>', '', code)   # SVG labels show these tags literally
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, 'm.html')
        open(f, 'w', encoding='utf-8').write(PAGE % html.escape(code))
        subprocess.run([EDGE, '--headless=new', '--disable-gpu', PROXY, '--hide-scrollbars', 
                        '--virtual-time-budget=8000', '--window-size=6000,5000', f'--screenshot={png}', 'file:///' + f.replace('\\', '/')],
                       capture_output=True, timeout=90, cwd=td)
    im = Image.open(png).convert('RGB')
    bb = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox()
    if bb:
        im = im.crop((max(bb[0] - 20, 0), max(bb[1] - 20, 0), min(bb[2] + 20, im.width), min(bb[3] + 20, im.height)))
    im.save(png)
    return im.size


if __name__ == '__main__':
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    chs = sys.argv[2:] or ['1', '2', '3', '4', '5']
    idx = json.load(open(os.path.join(out, 'index.json'))) if os.path.exists(os.path.join(out, 'index.json')) else {}
    for ch in chs:
        md = glob.glob(os.path.join(REPO, 'Main', f'Chapter-0{ch}', f'0{ch}-*.md'))[0]
        for k, b in enumerate(blocks(md), 1):
            name = f'ch{int(ch):02d}_{k:02d}'
            size = shot(b['code'], os.path.join(out, name + '.png'))
            if size[0] / size[1] > 2.4 and re.search(r'flowchart\s+LR', b['code']):
                # too wide for a portrait page: lay it out top-to-bottom instead
                size = shot(to_tb(b['code']), os.path.join(out, name + '.png'))
            idx[name] = dict(b, size=size)
            print(name, size, b['heading'][:50])
    json.dump(idx, open(os.path.join(out, 'index.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)



