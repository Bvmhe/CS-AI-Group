"""Add a 'Why this game?' start screen to the two supplement games (Final edition).

Reads the V1.0 copies kept in Supplements/_v1/ and writes the Final files into Supplements/.
The overlay is self-contained (inline CSS/JS, no network), opens on load and can be reopened
from a floating button.
"""
import os, shutil

SUP = r'C:\Projects\CS-AI-Group\Stage3-Final\Supplements'
V1 = os.path.join(SUP, '_v1')

CSS = """
<style id="why-css">
#why-ov{position:fixed;inset:0;z-index:2147483000;display:flex;align-items:center;justify-content:center;
 background:rgba(10,12,30,.72);backdrop-filter:blur(3px);font:15px/1.5 "Segoe UI",system-ui,sans-serif;padding:16px}
#why-ov[hidden]{display:none}
#why-card{background:#fffaf2;color:#2b2233;max-width:640px;width:100%;max-height:92vh;overflow:auto;border-radius:22px;
 border:4px solid #f06a9b;box-shadow:0 18px 50px rgba(0,0,0,.45);padding:22px 26px}
#why-card h2{margin:0 0 4px;font-size:24px;color:#d2386f}
#why-card .goal{display:inline-block;background:#ffe27a;border-radius:99px;padding:3px 12px;font-weight:700;font-size:13px;margin:2px 0 10px}
#why-card h3{margin:14px 0 4px;font-size:16px;color:#2b2233}
#why-card ul,#why-card ol{margin:4px 0 0;padding-left:22px}
#why-card li{margin:3px 0}
#why-card table{border-collapse:collapse;width:100%;font-size:14px;margin-top:4px}
#why-card td{border-top:1px solid #f3d3df;padding:5px 6px;vertical-align:top}
#why-card td:first-child{font-weight:700;white-space:nowrap;color:#d2386f}
#why-go{margin-top:16px;background:#f06a9b;color:#fff;border:0;border-radius:99px;padding:10px 26px;font:700 16px "Segoe UI",system-ui,sans-serif;cursor:pointer}
#why-go:hover{background:#d2386f}
#why-btn{position:fixed;left:12px;bottom:12px;z-index:2147482999;background:#ffe27a;color:#2b2233;border:2px solid #f06a9b;
 border-radius:99px;padding:6px 14px;font:700 13px "Segoe UI",system-ui,sans-serif;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.3)}
</style>
"""

JS = """
<script id="why-js">
(function(){
  var ov=document.getElementById('why-ov'), go=document.getElementById('why-go'), b=document.getElementById('why-btn');
  function close(){ov.hidden=true;b.focus&&b.blur();}
  go.addEventListener('click',close);
  b.addEventListener('click',function(){ov.hidden=false;go.focus({preventScroll:true});});
  ov.addEventListener('click',function(e){if(e.target===ov)close();});
  document.addEventListener('keydown',function(e){if(e.key==='Escape'&&!ov.hidden)close();},true);
  go.focus({preventScroll:true});
})();
</script>
"""

MAZE = """
<div id="why-ov" role="dialog" aria-modal="true" aria-labelledby="why-h">
 <div id="why-card">
  <h2 id="why-h">Why this game?</h2>
  <span class="goal">Learning Goal 3 · Recipe 03 “Training the Pup”</span>
  <p>Pup's Pantry Maze is <b>Recipe 03 made playable</b>. You watch three chefs try to reach the ingredient,
  and only one of them <b>learns from rewards</b>, which is exactly what the chapter is about.</p>
  <h3>What each part stands for</h3>
  <table>
   <tr><td>Learner Pup</td><td>The reinforcement-learning <b>agent</b> using Q-learning (guide 3.3). Its Q-table fills up as it practises.</td></tr>
   <tr><td>Carrot / treats</td><td>The <b>reward</b> signal (3.2). The pup is never told the path, only “good” or “bad” afterwards.</td></tr>
   <tr><td>α, γ, ε sliders</td><td>Learning rate, discount and exploration rate from the Q-update formula (3.3). Change them and watch learning speed up, slow down or stall.</td></tr>
   <tr><td>Map Chef</td><td>Search with a full map. Fast, but it is <b>not learning</b>: give it a new kitchen and it needs a new map.</td></tr>
   <tr><td>Cookie Jar kitchen</td><td><b>Reward hacking</b> (3.3): the pup finds a way to collect reward that is not what Mama wanted.</td></tr>
  </table>
  <h3>While you play, notice</h3>
  <ul>
   <li>How many episodes the pup needs before its path gets short and steady.</li>
   <li>What happens with ε = 0 (no exploring) versus a high ε (always exploring).</li>
  </ul>
  <h3>After playing, answer (debrief)</h3>
  <ol>
   <li>Which chef got better with practice, and why?</li>
   <li>Why does the Map Chef need the whole map first?</li>
   <li>What went wrong in the Cookie Jar, and how would you change the reward?</li>
  </ol>
  <button id="why-go" type="button">Start playing ▶</button>
 </div>
</div>
<button id="why-btn" type="button" aria-label="Why this game? Learning goal and debrief questions">❓ Why this game?</button>
"""

DINER = """
<div id="why-ov" role="dialog" aria-modal="true" aria-labelledby="why-h">
 <div id="why-card">
  <h2 id="why-h">Why this game?</h2>
  <span class="goal">Learning Goals 1–5 · Recipes 01–05</span>
  <p>The 24-Hour Diner is <b>the whole study guide as a review game</b>. Nothing in it is new content:
  every mission is a page of the guide, and every customer's order is a question from our Test Bank.
  Playing it is <b>retrieval practice</b>: you recall the idea instead of re-reading it.</p>
  <h3>Each room is one recipe</h3>
  <table>
   <tr><td>Room 1</td><td>Recipe 01 The Kitchen Crew: CPU, GPU, TPU, NPU and memory (Goal 1)</td></tr>
   <tr><td>Room 2</td><td>Recipe 02 Order Up!: networks, latency, cloud vs edge (Goal 2)</td></tr>
   <tr><td>Room 3</td><td>Recipe 03 Training the Pup: rewards, Q-learning, explore vs exploit (Goal 3)</td></tr>
   <tr><td>Room 4</td><td>Recipe 04 Chef's Senses: camera, LiDAR, ultrasonic, radar, sensor fusion (Goal 4)</td></tr>
   <tr><td>Room 5</td><td>Recipe 05 Running the Chain: putting hardware, networks and software together at scale (Goal 5)</td></tr>
  </table>
  <h3>How the game parts help you learn</h3>
  <ul>
   <li><b>Missions</b> = read a short part of the chapter, then try the side task.</li>
   <li><b>Serving customers</b> = answering Test Bank questions; a wrong answer shows the explanation.</li>
   <li><b>Rush hour</b> = timed recall for each room, like exam pressure.</li>
   <li><b>The Professor's Final Exam</b> = all 60 Test Bank questions as a final check.</li>
  </ul>
  <h3>After playing, answer (debrief)</h3>
  <ol>
   <li>Which room had the most customers you could not serve? Re-read that recipe in the guide.</li>
   <li>Pick one wrong answer and explain in your own words why the right answer is right.</li>
   <li>Which idea did you only understand after seeing it in the game, and which page of the guide explains it?</li>
  </ol>
  <button id="why-go" type="button">Open the diner ▶</button>
 </div>
</div>
<button id="why-btn" type="button" aria-label="Why this game? Learning goals and debrief questions">❓ Why this game?</button>
"""

DINER_LABELS = [
    ('<title>Fengwei 24-Hour Diner · Stage 2 Study Guide</title>', '<title>Fengwei 24-Hour Diner · Final Study Guide</title>'),
    ('content="Fengwei Fried Rice Stage 2 study guide', 'content="Fengwei Fried Rice Final study guide'),
    ('<small>Stage 2 Diner</small>', '<small>Final Study Guide</small>'),
    ('<span class="dot"></span>Stage 2 study guide · open 24 hours', '<span class="dot"></span>Final study guide · open 24 hours'),
    ('Every mission is the real Stage 2 course.', 'Every mission is the real course.'),
    ('Questions: Stage 2 Test Bank</div>', 'Questions: Final Test Bank</div>'),
    ('Every mission teaches one part of the Stage 2 course.', 'Every mission teaches one part of the course.'),
    ('All <b>60 questions</b> from our Stage 2 Test Bank', 'All <b>60 questions</b> from our Final Test Bank'),
    ('Your Stage 2 review starts', 'Your final review starts'),
    ('Questions: Fengwei Fried Rice Stage 2 Test Bank.', 'Questions: Fengwei Fried Rice Final Test Bank.'),
    ('Hardware, Networks & Behaviourist AI · Stage 2 Study Guide · November 2026', 'Hardware, Networks & Behaviourist AI · Final Study Guide · November 2026'),
    ('"About this guide","text":"Stage 2 Study Guide,', '"About this guide","text":"Final Study Guide,'),
]


def inject(name, overlay, labels=()):
    src = os.path.join(V1, name)
    h = open(src, encoding='utf8').read()
    assert 'id="why-ov"' not in h
    for old, new in labels:
        assert h.count(old) == 1, (old, h.count(old))
        h = h.replace(old, new)
    h = h.replace('</head>', CSS + '</head>', 1)
    i = h.rfind('</body>')
    h = h[:i] + overlay + JS + h[i:]
    open(os.path.join(SUP, name), 'w', encoding='utf8', newline='').write(h)
    print(name, 'ok', len(h))


if __name__ == '__main__':
    os.makedirs(V1, exist_ok=True)
    for n in ('Pup_Pantry_Maze_offline.html', 'Fengwei_24H_Diner_Study_Guide.html'):
        if not os.path.exists(os.path.join(V1, n)):
            shutil.copy2(os.path.join(SUP, n), os.path.join(V1, n))
    inject('Pup_Pantry_Maze_offline.html', MAZE)
    inject('Fengwei_24H_Diner_Study_Guide.html', DINER, DINER_LABELS)
