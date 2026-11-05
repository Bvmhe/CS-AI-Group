"""Upload every file in DIR matching GLOB to Canva (one fresh upload URL per file), skipping files already done.
Usage: set DIR=<folder> GLOB=*.png OUT=media_img.json ; python -I up4.py
OUT maps file stem -> mediaId (for page backgrounds the stem is the page number, e.g. '005' -> stored as '5').
Fresh upload URLs come from the newest session transcript (create-upload-url results)."""
import base64, glob, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DIR, PAT = os.environ['DIR'], os.environ.get('GLOB', '*.png')
OUT = os.path.join(HERE, os.environ['OUT'])
USED = os.path.join(HERE, 'used.txt')
_proj = r'C:\Users\LimunZ\.claude\projects\C--Users-LimunZ-OneDrive-Documents-GitHub-CS-AI-Group'
TRANSCRIPT = max(glob.glob(os.path.join(_proj, '*.jsonl')), key=os.path.getmtime)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7890'}))


def key(f):
    s = os.path.splitext(os.path.basename(f))[0]
    return str(int(s)) if s.isdigit() else os.path.basename(f)


def claims(u):
    tok = u.split('_sat=')[1].split('.')[1]
    return json.loads(base64.urlsafe_b64decode(tok + '=' * (-len(tok) % 4)))


media = json.load(open(OUT)) if os.path.exists(OUT) else {}
files = {key(f): f for f in sorted(glob.glob(os.path.join(DIR, PAT)))}
todo = [k for k in files if not str(media.get(k, '')).startswith('MA')]
txt = open(TRANSCRIPT, encoding='utf-8', errors='ignore').read()
urls = list(dict.fromkeys(re.findall(r'https://www\.canva\.com/_blob/upload/ingress/upload/file/agent-tool\?_sat=[\w.\-]+', txt)))
used = set(open(USED).read().split()) if os.path.exists(USED) else set()
fresh = sorted([u for u in urls if u not in used and claims(u)['exp'] > time.time() + 60], key=lambda u: claims(u)['iat'])
pairs = list(zip(todo, fresh))


def post(p):
    k, u = p
    req = urllib.request.Request(u, data=open(files[k], 'rb').read(), method='POST', headers={'Content-Type': 'application/octet-stream'})
    try:
        return k, u, json.loads(opener.open(req, timeout=180).read())['mediaId']
    except Exception as e:
        return k, u, 'ERR ' + str(e)[:80]


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(post, pairs))
with open(USED, 'a') as f:
    for k, u, m in res:
        f.write(u + '\n')
        if m.startswith('MA'):
            media[k] = m
        else:
            print('FAILED', k, m)
json.dump(media, open(OUT, 'w'), indent=0)
left = [k for k in files if not str(media.get(k, '')).startswith('MA')]
print(f'in Canva: {len(files) - len(left)} of {len(files)} | left: {len(left)}' + (f' (next {left[0]})' if left else ''))
