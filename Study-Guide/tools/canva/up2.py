"""Take fresh Canva upload URLs from this session's transcript and POST the next page images to them.
Usage: python -I up2.py   (state kept in media.json / used.txt next to this script)"""
import base64, glob, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.environ.get('PAGES_DIR', os.path.join(HERE, 'render2', 'full'))   # set PAGES_DIR to <render out>\full
MEDIA = os.path.join(HERE, 'media.json')
USED = os.path.join(HERE, 'used.txt')
_proj = r'C:\Users\LimunZ\.claude\projects\C--Users-LimunZ-OneDrive-Documents-GitHub-CS-AI-Group'
TRANSCRIPT = max(glob.glob(os.path.join(_proj, '*.jsonl')), key=os.path.getmtime)   # newest session transcript
CUTOFF = 1793803100          # URLs issued before this were used by earlier uploads
TOTAL = len(glob.glob(os.path.join(FULL, '*.jpg')))
opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:7890'}))


def claims(u):
    tok = u.split('_sat=')[1].split('.')[1]
    return json.loads(base64.urlsafe_b64decode(tok + '=' * (-len(tok) % 4)))


txt = open(TRANSCRIPT, encoding='utf-8', errors='ignore').read()
urls, seen = [], set()
for u in re.findall(r'https://www\.canva\.com/_blob/upload/ingress/upload/file/agent-tool\?_sat=[\w.\-]+', txt):
    if u not in seen:
        seen.add(u)
        urls.append(u)
used = set(open(USED).read().split()) if os.path.exists(USED) else set()
now = time.time()
fresh = [u for u in urls if u not in used and claims(u)['iat'] >= CUTOFF and claims(u)['exp'] > now + 60]
fresh.sort(key=lambda u: claims(u)['iat'])
media = json.load(open(MEDIA)) if os.path.exists(MEDIA) else {}
todo = [n for n in range(1, TOTAL + 1) if not str(media.get(str(n), 'ERR')).startswith('MA')]
pairs = list(zip(todo, fresh))
print(f'{len(fresh)} fresh urls, {len(todo)} pages left, uploading {len(pairs)}')


def post(p):
    n, u = p
    data = open(os.path.join(FULL, f'{n:03d}.jpg'), 'rb').read()
    req = urllib.request.Request(u, data=data, method='POST', headers={'Content-Type': 'application/octet-stream'})
    try:
        return n, u, json.loads(opener.open(req, timeout=180).read())['mediaId']
    except Exception as e:
        return n, u, 'ERR ' + str(e)[:80]


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(post, pairs))
with open(USED, 'a') as f:
    for n, u, m in res:
        f.write(u + '\n')
        media[str(n)] = m
        if not m.startswith('MA'):
            print('FAILED', n, m)
json.dump(media, open(MEDIA, 'w'), indent=0)
ok = sorted(int(k) for k, v in media.items() if v.startswith('MA'))
print('uploaded ok:', len(ok), 'of', TOTAL, '| last batch', res[0][0] if res else '-', '-', res[-1][0] if res else '-')
