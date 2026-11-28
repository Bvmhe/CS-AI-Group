"""Build the readable PDF copy of the final Markdown content.

Usage (from the repository root):
    python tools/build_pdf.py

What it does:
  1. Reads Main/00-Module-Overview.md and the five chapter files, in reading order.
  2. Writes tools/build/module.html, which renders the Markdown in the browser with
     marked.js and draws the Mermaid diagrams with mermaid.js (both in tools/vendor/,
     MIT licence, from the official npm packages via jsDelivr).
  3. Prints that page to Main/FengweiFriedRice_Stage3_FinalMarkdown.pdf with
     Microsoft Edge in headless mode.

Nothing is downloaded at build time; the PDF uses only files inside this repository.
"""
import json, os, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "Main"
OUT_PDF = MAIN / "FengweiFriedRice_Stage3_FinalMarkdown.pdf"
BUILD = ROOT / "tools" / "build"
CHAPTERS = [
    ("c0", "00-Module-Overview.md"),
    ("c1", "Chapter-01/01-AI-Hardware-Architecture.md"),
    ("c2", "Chapter-02/02-Computer-Networks-and-AI-Infrastructure.md"),
    ("c3", "Chapter-03/03-Behaviourist-AI-and-Reinforcement-Learning.md"),
    ("c4", "Chapter-04/04-AI-Sensors-and-Intelligent-Systems.md"),
    ("c5", "Chapter-05/05-AI-System-Integration-and-Optimization.md"),
]
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

docs = []
for cid, rel in CHAPTERS:
    text = (MAIN / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
    folder = str(pathlib.PurePosixPath(rel).parent)
    docs.append({"id": cid, "dir": "" if folder == "." else folder + "/", "md": text})

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Hardware, Networks & Behaviourist AI — Final Markdown Content (Fengwei Fried Rice)</title>
<base href="__BASE__">
<style>
 @page { size: A4; margin: 16mm 15mm 18mm 15mm; }
 body { font-family: "Segoe UI", Arial, "Microsoft YaHei", sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1d1d1f; }
 h1 { font-size: 20pt; color: #b3261e; border-bottom: 2px solid #b3261e; padding-bottom: 4px; margin-top: 1.4em; }
 h2 { font-size: 14.5pt; color: #7a1c15; margin-top: 1.3em; border-bottom: 1px solid #e3d5d3; }
 h3 { font-size: 12pt; } h4 { font-size: 11pt; }
 h1, h2, h3, h4 { break-after: avoid; }
 .chapter { break-before: page; }
 .chapter:first-of-type { break-before: auto; }
 table { border-collapse: collapse; margin: 0.6em 0; width: 100%; font-size: 9.5pt; break-inside: auto; }
 th, td { border: 1px solid #cfcfcf; padding: 4px 6px; vertical-align: top; text-align: left; }
 th { background: #f6ecea; }
 tr { break-inside: avoid; }
 code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f4; padding: 0 3px; border-radius: 3px; }
 pre { background: #f7f7f7; border: 1px solid #e2e2e2; padding: 8px 10px; white-space: pre-wrap; break-inside: avoid; }
 pre code { background: none; padding: 0; }
 pre.mermaid { background: none; border: none; text-align: center; }
 img { max-width: 78%; max-height: 95mm; display: block; margin: 0.6em auto 0.2em; break-inside: avoid; }
 blockquote { border-left: 4px solid #d9a19b; background: #fbf3f2; margin: 0.8em 0; padding: 4px 12px; break-inside: avoid; }
 blockquote.note { border-color: #2f6fb3; background: #eef4fb; }
 blockquote.tip { border-color: #2e8540; background: #eef8f0; }
 blockquote.warning, blockquote.caution { border-color: #c77700; background: #fff6e6; }
 .alert-title { font-weight: 700; text-transform: uppercase; font-size: 8.5pt; letter-spacing: .05em; }
 a { color: #1a5fa8; text-decoration: none; }
 .cover { text-align: center; padding-top: 70mm; break-after: page; }
 .cover h1 { border: none; font-size: 26pt; }
 .cover p { font-size: 12pt; color: #555; }
</style></head><body>
<div class="cover">
 <h1>Hardware, Networks &amp; Behaviourist AI</h1>
 <p><b>Final Markdown Content — readable copy</b></p>
 <p>Team Fengwei Fried Rice · Introduction to Artificial Intelligence</p>
 <p>Version: Final (Stage 3) · 28 November 2026</p>
 <p>Source: github.com/Bvmhe/CS-AI-Group, folder <code>Main/</code></p>
</div>
<div id="out"></div>
<script src="__VENDOR__/marked.min.js"></script>
<script src="__VENDOR__/mermaid.min.js"></script>
<script>
const DOCS = __DOCS__;
function slug(t){ return t.toLowerCase().replace(/<[^>]+>/g,'').replace(/[^\p{L}\p{N}\s_-]/gu,'').trim().replace(/\s/g,'-'); }
function alerts(md){ // GitHub "> [!NOTE]" alerts -> marker paragraph
  return md.replace(/^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$/gim, (m,t)=>'> %%ALERT-'+t.toUpperCase()+'%%');
}
const out = document.getElementById('out');
for (const d of DOCS) {
  const seen = {};
  const r = new marked.Renderer();
  r.heading = function({tokens, depth}) {
    const html = this.parser.parseInline(tokens);
    let s = slug(html); if (seen[s] !== undefined) { seen[s]++; s = s + '-' + seen[s]; } else seen[s] = 0;
    return `<h${depth} id="${d.id}-${s}">${html}</h${depth}>`;
  };
  r.code = function({text, lang}) {
    if ((lang||'').trim() === 'mermaid') return `<pre class="mermaid">${text.replace(/</g,'&lt;')}</pre>`;
    return `<pre><code>${text.replace(/&/g,'&amp;').replace(/</g,'&lt;')}</code></pre>`;
  };
  r.image = function({href, title, text}) {
    if (!/^(https?:|data:|\/)/.test(href)) href = d.dir + href;
    return `<img src="${href}" alt="${text.replace(/"/g,'&quot;')}">`;
  };
  r.link = function({href, title, tokens}) {
    const inner = this.parser.parseInline(tokens);
    if (href.startsWith('#')) href = '#' + d.id + '-' + href.slice(1);
    else { const m = href.match(/^(?:\.\/)?(Chapter-0\d\/[^#]+\.md|\.\.\/Chapter-0\d\/[^#]+\.md)$/);
           if (m) { const n = m[1].match(/Chapter-0(\d)/)[1]; href = '#c' + n + '-top'; } }
    return `<a href="${href}">${inner}</a>`;
  };
  const sec = document.createElement('section');
  sec.className = 'chapter'; sec.id = d.id + '-top';
  sec.innerHTML = marked.parse(alerts(d.md), {renderer: r, gfm: true});
  sec.querySelectorAll('blockquote').forEach(b => {
    const p = b.querySelector('p'); if (!p) return;
    const m = p.innerHTML.match(/^%%ALERT-(\w+)%%/); if (!m) return;
    b.classList.add(m[1].toLowerCase());
    p.innerHTML = p.innerHTML.replace(/^%%ALERT-\w+%%\s*/, `<span class="alert-title">${m[1]}</span><br>`);
  });
  out.appendChild(sec);
}
mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose', flowchart: { htmlLabels: true } });
mermaid.run({ querySelector: 'pre.mermaid' }).then(() => { document.title += ' '; window.__done = true; })
  .catch(e => { document.body.insertAdjacentHTML('afterbegin', '<p style="color:red">Mermaid error: '+e+'</p>'); window.__done = true; });
</script></body></html>"""

BUILD.mkdir(parents=True, exist_ok=True)
html = (HTML.replace("__BASE__", MAIN.as_uri() + "/")
            .replace("__VENDOR__", (ROOT / "tools" / "vendor").as_uri())
            .replace("__DOCS__", json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")))
page = BUILD / "module.html"
page.write_text(html, encoding="utf-8")
print("wrote", page)

if "--html-only" in sys.argv:
    sys.exit(0)
edge = next((p for p in EDGE if os.path.exists(p)), None)
if not edge:
    sys.exit("Microsoft Edge not found; open tools/build/module.html in a browser and print it to PDF.")
subprocess.run([edge, "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                "--no-pdf-header-footer", "--virtual-time-budget=60000",
                f"--print-to-pdf={OUT_PDF}", page.as_uri()], check=True)
print("wrote", OUT_PDF, OUT_PDF.stat().st_size // 1024, "KB")
