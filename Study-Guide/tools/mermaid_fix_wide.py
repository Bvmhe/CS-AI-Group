"""Re-render diagrams that came out too wide for a portrait page with a top-to-bottom layout.
Usage: python mermaid_fix_wide.py <diagram_dir>"""
import json, os, re, sys
from mermaid_png import shot

d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'index.json'), encoding='utf-8'))
for name, b in idx.items():
    w, h = b['size']
    if w / h > 2.4 and re.search(r'flowchart\s+LR', b['code']):
        b['size'] = shot(re.sub(r'flowchart\s+LR', 'flowchart TB', b['code'], count=1), os.path.join(d, name + '.png'))
        print(name, (w, h), '->', b['size'])
json.dump(idx, open(os.path.join(d, 'index.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
