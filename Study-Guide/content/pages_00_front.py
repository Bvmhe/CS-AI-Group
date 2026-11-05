"""Front matter: cover, intro, topics, table of contents, team, how to play, goals, score card, cast."""
import glob
import importlib.util
import os

from kit import *

C0 = 0
HERE = os.path.dirname(os.path.abspath(__file__))
FRONT_LEN = 11   # number of pages in this file (checked below)


def _starts():
    """Page number where each later module starts, so the contents page is always right."""
    n = FRONT_LEN + 1
    starts = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'pages_*.py'))):
        name = os.path.basename(p)
        if name.startswith('pages_00'):
            continue
        spec = importlib.util.spec_from_file_location('_toc_' + name[:-3], p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        starts[name[6:-3]] = (n, len(m.PAGES))
        n += len(m.PAGES)
    return starts, n - 1


S, TOTAL = _starts()


def pg(key):
    return S[key][0]


def rng(key, vkey):
    a = S[key][0]
    b = S[vkey][0] + S[vkey][1] - 1
    return f'p. {a}-{b}'


PAGES = [
    P(C0, 'STAGE 2 STUDY GUIDE',
      TITLE('Fengwei', 120, 'yellow'),
      TITLE('Fried Rice', 120, 'red'),
      BIG("We're Coooking!", 'blue'),
      W('A cook-along study guide', 'Module: Hardware, Networks & Behaviourist AI\nCourse: Introduction to Artificial Intelligence\nTeam: Fengwei Fried Rice\nNovember 2026\n' + f'{TOTAL} pages · 5 recipes · 20 mini-games · 5 kitchen labs', 'mint', '🍳', w=640),
      floor=[('mama_fry', 300), ('fried_rice', 170), ('pup', 200)],
      deco=[('qr_repo', 790, 1240, 200)], id='cover'),
    P(C0, 'WELCOME TO THE KITCHEN',
      B('mama_happy', 'Welcome, new apprentice! Did you know that the phone in your pocket does billions of tiny maths steps every time it unlocks with your face? This guide shows how AI really works: the chips that cook the maths, the networks that deliver it, and the agents that learn by trial and error.'),
      W('💡 Did you know?', '• Moving one number from memory can cost ~170× more energy than multiplying it (Recipe 01).\n• Light needs ~40 ms to go from Beijing to New York and back through fibre: no network can beat that (Recipe 02).\n• Robot dogs learn to walk in about 8 minutes by training 4,096 virtual dogs at once on one GPU (Recipe 03).\n• At 150 m a child is only about five pixels wide in a car camera (Recipe 04).\n• A 4,096-chip AI job breaks roughly every 11 hours, so it saves itself every ~51 minutes (Recipe 05).', 'yellow'),
      N('Scan the QR code on the cover to open our team repository (github.com/Bvmhe/CS-AI-Group) with the source chapters, the Test Bank and the AI log.', label='📱 SCAN HERE', color='blue'),
      floor=['mama_cheer', 'boy_grin', 'girl_happy'], id='intro'),
    P(C0, 'TOPICS COVERED',
      W('🍳 Recipe 01 · The Kitchen Crew', 'AI hardware: MACs, CPU vs GPU vs TPU vs NPU, systolic arrays, parallelism and Amdahl\'s law, memory hierarchy and the roofline, Apple Neural Engine.', 'red'),
      W('🛵 Recipe 02 · Order Up!', 'Networks and AI infrastructure: layers and packets, TCP/IP, UDP and QUIC, cloud vs edge, latency and bandwidth, all-reduce, 5G and AI.', 'blue'),
      W('🐶 Recipe 03 · Training the Kitchen Pup', 'Behaviourist AI and reinforcement learning: rewards, the agent-environment loop, return and value, Q-learning, exploration, robot dogs.', 'mint'),
      W('👀 Recipe 04 · Chef\'s Senses', 'AI sensors: cameras, LiDAR, ultrasonic and radar, real-time data, sensor fusion and Kalman filters, decisions and safety, Tesla.', 'lav'),
      W('🏭 Recipe 05 · Running the Chain', 'Integration and optimisation: compilers and fusion, distributed training, serving, energy, cost and failures, Google TPU pods.', 'orange'),
      floor=['mama_ok', 'pup', 'papa'], id='topics'),
    P(C0, 'TODAY\'S MENU (CONTENTS)',
      T(['Course', 'Page'],
        [['Starter · Welcome, topics, team, how to play, goals, score card', 'p. 1-11'],
         ['Recipe 01 · The Kitchen Crew: AI Hardware', rng('01_hardware', '01v_videos')],
         ['   1.1 What is AI hardware? · 1.2 CPU vs GPU vs TPU · 1.3 Parallel processing', f'p. {pg("01_hardware") + 2}'],
         ['   1.4 Memory hierarchy · 1.5 Case: Apple Neural Engine · Watch & Learn', ''],
         ['Recipe 02 · Order Up! Networks & AI Infrastructure', rng('02_networks', '02v_videos')],
         ['   2.1 Basics · 2.2 TCP/IP · 2.3 Cloud vs edge · 2.4 Latency & bandwidth', f'p. {pg("02_networks") + 2}'],
         ['   2.5 Case: 5G and AI · Watch & Learn', ''],
         ['Recipe 03 · Training the Kitchen Pup: RL', rng('03_rl', '03v_videos')],
         ['   3.1 Principles · 3.2 Agent & environment · 3.3 Rewards', f'p. {pg("03_rl") + 2}'],
         ['   3.4 Explore vs exploit · 3.5 Case: robot dogs · Watch & Learn', ''],
         ['Recipe 04 · Chef\'s Senses: AI Sensors', rng('04_sensors', '04v_videos')],
         ['   4.1 Sensors · 4.2 Real-time data · 4.3 Fusion', f'p. {pg("04_sensors") + 2}'],
         ['   4.4 Hardware + decisions · 4.5 Case: Tesla · Watch & Learn', ''],
         ['Recipe 05 · Running the Chain: Integration', rng('05_integration', '05v_videos')],
         ['   5.1 Hardware-software · 5.2 Distributed · 5.3 Optimisation', f'p. {pg("05_integration") + 2}'],
         ['   5.4 Energy, cost, scale · 5.5 Case: Google TPU · Watch & Learn', ''],
         ['Dessert · Word Soup, references, credits, AI use, back cover', f'p. {pg("09_end")}-{TOTAL}']],
        [780, 180], color='yellow', fs=22),
      TIP('Each recipe ends with a Kitchen Lab (live platform + offline fallback), an Exercise with answers, a two-page Glossary, a Recipe Card summary and curated videos. The separate Test Bank (Word file) has 50+ questions with an answer key.'),
      floor=['mama_book', 'mama_icon_pan'], id='toc'),
    P(C0, 'THE KITCHEN BRIGADE',
      T(['Member', 'Kitchen role', 'Team role'],
        [['纪小佩', 'Head Chef', 'CEO: planning, coordination, final sign-off'],
         ['林元甲', 'Pantry Keeper', 'CIO: repository, tools, file management'],
         ['温俊强', 'Recipe Tester', 'Content Lead: worked examples and answer checks'],
         ['汤丽嘉', 'Recipe Writer', 'Content Lead: explanations and activities'],
         ['曾富良', 'Plating Artist', 'Visual Design Lead: layout, sprites, identity'],
         ['李秀春', 'Camera Crew', 'Video Lead: curated videos and timestamps'],
         ['黄施杰', 'Taste Inspector', 'Quality Lead: accuracy, references, Test Bank review']],
        [200, 280, 480], color='pink'),
      W('About this guide', 'Stage 2 Study Guide, built from the team\'s shared Markdown content (Main/Chapter-01 to 05). Repository: github.com/Bvmhe/CS-AI-Group. Deadline: 7 November 2026, 23:59 Beijing time.', 'mint', '🏷'),
      floor=['mama_happy', 'papa', 'grandpa', 'grandma'], id='team'),
    P(C0, 'HOW TO PLAY',
      W('Your route through the guide', '1. Pick a Recipe from the Menu. Recipes build on each other, so 01 → 05 is the best route.\n2. Every step follows the same cook-along: Mama\'s Problem → Definition → Worked example → Real tech → Mini-game.\n3. Mini-games have six parts: 🎯 Goal · 🧺 Prep · 🔪 Task · 👀 Observe · 💭 Reflect · ⭐ Mama\'s Feedback (the answers).\n4. Stamp your Score Card: Gold = right first time · Silver = right after a hint · Bronze = read the answer and redo it.\n5. Finish a Recipe → unlock its Recipe Card!', 'yellow', '🎮'),
      T(['Box', 'What it means'],
        [['MAMA\'S TIP (pink)', 'A kitchen analogy to help you remember'],
         ['REAL TECH (mint)', 'The exact technical meaning behind the analogy'],
         ['PAPA SAYS (yellow)', 'A common mistake: Papa loves shortcuts!'],
         ['KEY IDEA (blue)', 'The one thing to remember from the page'],
         ['BRIDGE (mint)', 'How this links to another recipe']],
        [300, 660], color='mint'),
      N('Kitchen tools: pen, paper, a calculator, and a phone or laptop with internet for the 🌐 Kitchen Labs (an offline fallback is always included).', label='🧺 YOU NEED', color='pink'),
      floor=['mama_ok', 'sticker_tasty'], id='howto'),
    P(C0, 'GOALS & PREREQUISITES',
      W('What you will be able to cook', '1. Name CPU, GPU, TPU/NPU and memory, and explain what each does for AI. (Recipe 01)\n2. Explain how networks move AI data, and choose cloud vs edge using latency and bandwidth. (Recipe 02)\n3. Describe how an agent learns from rewards, and balance exploration and exploitation. (Recipe 03)\n4. Compare cameras, LiDAR, ultrasonic and radar, and explain sensor fusion. (Recipe 04)\n5. Explain how hardware, networks and software are combined and optimised at scale. (Recipe 05)', 'yellow', '🎯'),
      W('Before you start', 'Basic arithmetic and powers of ten; what a variable and a loop are; a rough idea of what a CPU and the internet are. No electronics or advanced maths needed.\nWho it is for: 1st/2nd-year CS and IT undergraduates.\nTime: about 2-3 hours per Recipe, including mini-games.', 'blue', '🧺'),
      floor=['girl_overalls', 'boy_glasses', 'girl_pigtails'], id='goals'),
    P(C0, 'MY SCORE CARD',
      T(['Recipe', 'Mini-games and labs (stamp Gold, Silver or Bronze)'],
        [['01', '1.1 Stir-Fry Maths · 1.2 Hire the Chef · 1.3 Chopping Race · 1.4 Batch Cooking · Lab: Race the Crew · Taste Test'],
         ['02', '2.1 Bento Packing · 2.2 Careful vs Shouting Waiter · 2.3 Truck or Kitchen? · 2.4 Delivery Timer · Lab: Ping & Tracert'],
         ['03', '3.1 Good Pup! · 3.2 Pup Gridworld · 3.3 Reward Hacking · 3.4 Noodle Stall Gamble · Lab: GridWorld'],
         ['04', '4.1 Sensor Taste Test · 4.2 Beat the Timer · 4.3 Mix the Senses · 4.4 Brake or Bake? · Lab: Phone Sensor Kit'],
         ['05', '5.1 One Trip to the Fridge · 5.2 Split the Order · 5.3 Rush Hour · 5.4 Pay the Bills · Lab: Run a Real TPU']],
        [140, 820], color='orange'),
      W('Your rank', '0-10 Gold stamps = Kitchen Helper\n11-20 Gold stamps = Line Cook\n21+ Gold stamps = Head Chef of the whole chain!', 'yellow', '🏅'),
      floor=['mama_cheer', 'sticker_tasty', 'timer'], id='score'),
    P(C0, 'MEET THE KITCHEN',
      B('mama_stand', 'Mama: your teacher. She cheers when you get it right, gets fire-eyes when something burns, and always helps you fix it.'),
      B('papa', 'Papa: loves shortcuts. When Papa says "Easy! Just…", watch out: it is a common mistake.', side='right'),
      B('pup', 'The Kitchen Pup: learns tricks from treats. The star of Recipe 03.'),
      B('boy_grin', 'The kids: ask the "but why?" questions any beginner would ask. And you, the new apprentice: by the end you run the whole restaurant chain!', side='right'),
      floor=None, id='cast'),
    P(C0, 'ANALOGY vs REAL TECH',
      T(['In the kitchen', 'In real AI'],
        [['Adding ingredient × amount to the wok', 'A MAC (multiply-accumulate)'],
         ['Head Chef / Line Cooks / Conveyor / Rice Cooker', 'CPU / GPU / TPU / NPU'],
         ['Chopping board vs fridge vs supermarket', 'Cache vs DRAM vs storage'],
         ['Waiters carrying bento boxes', 'Packets over a network'],
         ['Central kitchen vs food truck', 'Cloud vs edge computing'],
         ['Training the pup with treats', 'Reinforcement learning with rewards'],
         ['A chef\'s eyes, ears and nose', 'Cameras, ultrasonic, LiDAR, radar'],
         ['A restaurant chain', 'A distributed AI system']],
        [480, 480], color='blue'),
      PAPA('"So a GPU really is 1,000 cooks!" Not quite! Analogies help you remember, but they are not exact. Always check the 🔬 REAL TECH box for the precise meaning.'),
      floor=['mama_fry', 'pup', 'boy_cap'], id='analogy'),
    P(C0, 'IMAGE & COPYRIGHT NOTE',
      W('About the pictures', 'The character and food artwork comes from the Cooking Mama video-game series (© Office Create; published by Majesco / Nintendo platforms). We cut the images into game-style sprites for a class project only. This guide is NOT for sale or public distribution, and no ownership of the artwork is claimed. Full credits are on the Image Credits page at the end.', 'pink', '©'),
      W('About the content', 'All explanations, worked examples, mini-games and numbers are the team\'s own work, written from our shared Markdown chapters and the cited sources. Illustrative numbers are marked "illustrative"; company claims are marked "reported" or "company claim".', 'mint', '📝'),
      floor=['mama_small', 'mama_icon_hearts'], id='copy'),
]

assert len(PAGES) == FRONT_LEN, len(PAGES)
