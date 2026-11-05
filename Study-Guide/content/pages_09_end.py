"""End matter: Word Soup puzzle + answers, references, image credits, AI use, contributions, back cover."""
import json
import os
import random

from kit import *

C9 = 9
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ word soup
CLUES = [
    ('MAC', 'Multiply, then add to the running total (1.1)'),
    ('GPU', 'Thousands of simple cores: the 1,000 line cooks (1.2)'),
    ('CACHE', 'Small fast memory near the cores: the chopping board (1.4)'),
    ('ROOFLINE', 'Model: min(peak, intensity × bandwidth) (1.4)'),
    ('PACKET', 'A small labelled piece of data on a network (2.1)'),
    ('LATENCY', 'How long one message takes to arrive (2.4)'),
    ('EDGE', 'Computing close to where the data is made (2.3)'),
    ('REWARD', 'The number that says how well the agent did (3.2)'),
    ('POLICY', 'The agent\'s rule for choosing actions (3.2)'),
    ('AGENT', 'The learner and decision-maker in RL (3.2)'),
    ('LIDAR', 'Laser time-of-flight sensor that makes point clouds (4.1)'),
    ('KALMAN', 'Predict-and-update filter for tracking (4.3)'),
    ('FUSION', 'Combining several sensors, or several operations, into one (4.3, 5.1)'),
    ('TORUS', 'Grid network whose edges wrap around (5.5)'),
    ('SHARD', 'One slice of a tensor or state split across chips (5.2)'),
]
GN = 14
DIRS = [(0, 1), (1, 0), (1, 1), (-1, 1)]


def make_grid(seed=7):
    rnd = random.Random(seed)
    g = [[''] * GN for _ in range(GN)]
    where = {}
    for word, _ in sorted(CLUES, key=lambda c: -len(c[0])):
        for _ in range(2000):
            dr, dc = rnd.choice(DIRS)
            r, c = rnd.randrange(GN), rnd.randrange(GN)
            cells = [(r + dr * i, c + dc * i) for i in range(len(word))]
            if all(0 <= a < GN and 0 <= b < GN and g[a][b] in ('', word[i]) for i, (a, b) in enumerate(cells)):
                for i, (a, b) in enumerate(cells):
                    g[a][b] = word[i]
                where[word] = (r, c, (dr, dc), cells)
                break
        else:
            raise RuntimeError(word)
    fill = 'ABCDEFGHIKLMNOPRSTUWY'
    sol = [[g[r][c] or '·' for c in range(GN)] for r in range(GN)]
    for r in range(GN):
        for c in range(GN):
            if not g[r][c]:
                g[r][c] = rnd.choice(fill)
    return g, sol, where


GRID, SOL, WHERE = make_grid()
COLS = [chr(65 + i) for i in range(GN)]
DNAME = {(0, 1): 'across', (1, 0): 'down', (1, 1): 'diag. down', (-1, 1): 'diag. up'}


def grid_table(g, color):
    return T([''] + COLS, [[str(r + 1)] + g[r] for r in range(GN)], [64] + [64] * GN, color=color, fs=24, center=True)


# ------------------------------------------------------------------ references
REFS = {
    1: [['Amdahl (1967)', 'Validity of the single processor approach… AFIPS Conf. Proc.'],
        ['Apple (2017; 2024)', 'iPhone X press release; Platform Security guide; Core ML docs'],
        ['Hennessy & Patterson (2019)', 'A new golden age for computer architecture. CACM 62(2)'],
        ['Horowitz (2014)', 'Computing\'s energy problem. IEEE ISSCC'],
        ['Jouppi et al. (2017)', 'In-datacenter performance analysis of a TPU. ISCA'],
        ['Sze et al. (2017)', 'Efficient processing of deep neural networks. Proc. IEEE'],
        ['Williams et al. (2009)', 'Roofline: an insightful visual performance model. CACM 52(4)'],
        ['NVIDIA; Qualcomm; Google', 'Tensor Core, Hexagon NPU and Ironwood (2025) documentation']],
    2: [['Kurose & Ross (2020)', 'Computer Networking: A Top-Down Approach, 8th ed. Pearson'],
        ['Cerf & Kahn (1974)', 'A protocol for packet network intercommunication. IEEE Trans. Comm.'],
        ['Postel (1980; 1981); Eddy (2022)', 'UDP (RFC 768), IP (RFC 791), TCP (RFC 9293)'],
        ['Iyengar & Thomson (2021)', 'QUIC (RFC 9000); Rescorla (2018) TLS 1.3 (RFC 8446)'],
        ['Mathis et al. (1997); Jacobson (1988)', 'TCP throughput model; congestion control'],
        ['Dean & Barroso (2013)', 'The tail at scale. CACM 56(2)'],
        ['Mell & Grance (2011)', 'The NIST Definition of Cloud Computing (SP 800-145)'],
        ['Satyanarayanan (2017); Shi et al. (2016)', 'Edge computing'],
        ['ITU-R (2015; 2017); 3GPP; ETSI (2014); O-RAN', '5G requirements, architecture, MEC, RIC']],
    3: [['Sutton & Barto (2018)', 'Reinforcement Learning: An Introduction, 2nd ed. MIT Press'],
        ['Thorndike (1911); Pavlov (1927)', 'Animal Intelligence; Conditioned Reflexes'],
        ['Skinner (1938); Watson (1913); Chomsky (1959)', 'Behaviourism and its critique'],
        ['Watkins & Dayan (1992); Sutton (1988)', 'Q-learning; temporal-difference learning'],
        ['Mnih et al. (2015)', 'Human-level control through deep RL. Nature 518'],
        ['Ng, Harada & Russell (1999)', 'Policy invariance under reward transformations. ICML'],
        ['Auer et al. (2002); Clark & Amodei (2016)', 'UCB bandits; faulty reward functions'],
        ['Hwangbo et al. (2019); Lee et al. (2020)', 'Legged robot locomotion. Science Robotics'],
        ['Rudin et al. (2022); Makoviychuk et al. (2021)', 'Walking in minutes; Isaac Gym'],
        ['Tobin et al. (2017); Tan et al. (2018); Kumar et al. (2021)', 'Domain randomisation, sim-to-real, RMA']],
    4: [['Thrun, Burgard & Fox (2005)', 'Probabilistic Robotics. MIT Press'],
        ['Kalman (1960)', 'A new approach to linear filtering and prediction problems'],
        ['Winner et al. (2016)', 'Handbook of Driver Assistance Systems. Springer'],
        ['Rosique et al. (2019); Feng et al. (2021)', 'Perception systems; multi-modal detection surveys'],
        ['Hall & Llinas (1997)', 'An introduction to multisensor data fusion. Proc. IEEE'],
        ['Philion & Fidler (2020); Li et al. (2022); Liu et al. (2023)', 'BEV: Lift-Splat-Shoot, BEVFormer, BEVFusion'],
        ['Geiger et al. (2012); Caesar et al. (2020); Sun et al. (2020)', 'KITTI, nuScenes, Waymo Open datasets'],
        ['SAE (2021); ISO 26262 (2018); ISO 21448 (2022)', 'Automation levels; functional safety; SOTIF'],
        ['Talpes et al. (2020); Tesla AI Day (2021, 2022)', 'Tesla FSD computer; Tesla Vision'],
        ['Waymo (2024); Electrek; TechCrunch; NHTSA', '6th-gen Driver; reported news; investigations']],
    5: [['Barroso, Hölzle & Ranganathan (2018)', 'The Datacenter as a Computer, 3rd ed.'],
        ['Jouppi et al. (2023)', 'TPU v4: an optically reconfigurable supercomputer. ISCA'],
        ['Kaplan et al. (2020); Hoffmann et al. (2022)', 'Scaling laws; compute-optimal training'],
        ['Narayanan et al. (2021); Rajbhandari et al. (2020)', 'Megatron-LM; ZeRO'],
        ['Patarasuk & Yuan (2009)', 'Bandwidth-optimal all-reduce algorithms'],
        ['Kwon et al. (2023)', 'PagedAttention (LLM serving). SOSP'],
        ['Xu et al. (2021)', 'GSPMD: general and scalable parallelization'],
        ['Patterson et al. (2021)', 'Carbon emissions and large neural network training'],
        ['Young (1974)', 'Optimum checkpoint interval. CACM 17(9)'],
        ['Google; Google Cloud', 'Ironwood (2025); PUE reports; Cloud TPU documentation']],
}
NAMES = {1: 'The Kitchen Crew', 2: 'Order Up!', 3: 'Training the Kitchen Pup', 4: "Chef's Senses", 5: 'Running the Chain'}
RCOL = {1: 'red', 2: 'blue', 3: 'mint', 4: 'lav', 5: 'orange'}

# ------------------------------------------------------------------ image credits
try:
    SRC = json.load(open(os.path.join(HERE, '..', 'assets', 'sprites', '_sources.json'), encoding='utf-8'))
except OSError:
    SRC = {}
folders = sorted({v.split('\\')[0].split('/')[0] for v in SRC.values()})

PAGES = [
    P(C9, "DESSERT · MAMA'S WORD SOUP",
      W('How to play', 'Read each clue, write the term, then find it in the grid. Words go across, down or diagonally (down-right or up-right). 15 words, one from every corner of the course.', 'yellow', '🥣'),
      grid_table(GRID, 'yellow'),
      floor=['soup', 'mama_happy', 'congee'], id='ws'),
    P(C9, 'WORD SOUP · CLUES',
      T(['#', 'Clue', 'Answer'], [[str(i + 1), c, '__________'] for i, (_, c) in enumerate(CLUES)], [70, 650, 240], color='mint', fs=23),
      floor=['mama_book', 'girl_bow'], id='wsc'),
    P(C9, 'WORD SOUP · ANSWERS',
      grid_table(SOL, 'pink'),
      T(['Word', 'Start · dir', 'Word', 'Start · dir', 'Word', 'Start · dir'],
        [sum(([w, f'{COLS[WHERE[w][1]]}{WHERE[w][0] + 1} ' + DNAME[WHERE[w][2]]] for w, _ in CLUES[r::5]), []) for r in range(5)],
        [170, 150, 170, 150, 170, 150], color='pink', fs=20),
      floor=None, id='wsa'),
    *[P(C9, f'REFERENCES · RECIPE 0{k}',
        W(f'Recipe 0{k} · {NAMES[k]}', 'Key sources used for this recipe (APA style, shortened). The full reference list, with titles, venues and access dates (early November 2026), is in the shared chapter Main/Chapter-0' + f'{k} of our repository.', RCOL[k], '📚'),
        T(['Source', 'Work'], REFS[k], [400, 560], color=RCOL[k], fs=21),
        floor=['mama_book', 'grandpa'] if k % 2 else ['mama_small', 'grandma'], id=f'ref{k}') for k in range(1, 6)],
    P(C9, 'VIDEO & PLATFORM SOURCES',
      T(['Resource', 'Creator / owner'],
        [['But what is a neural network? (aircAruvnKk)', '3Blue1Brown'],
         ['How do Graphics Cards Work? (h9Z4oGN89MU)', 'Branch Education'],
         ['Tensor Processing Units (eyAjbgkBdjU); TPU data center (FsxthdQ_sL4); Scaling AI with TPUs (8zBaa20mvoc)', 'Google Cloud'],
         ['A very short intro to the Roofline model (IrkNZG8MJ64)', 'NHR@FAU'],
         ['How does the INTERNET work? (x3c1ih2NJEg)', 'Lesics'],
         ['What is bandwidth? (QLJCqY-7Vrw)', 'internet-class'],
         ['Inside a Google data center (XZmGGAbHqa0)', 'Google Workspace'],
         ['RL: Machine Learning Meets Control Theory (0MNVhXEX9to)', 'Steve Brunton'],
         ['Markov Decision Processes (2iF9PRriA7w)', 'Computerphile'],
         ['AI Learns to Walk (L_4BPjLBF4E)', 'AI Warehouse'],
         ['How do self-driving cars "see"? (PRg5RNU_JLk)', 'TED-Ed'],
         ['How Does LiDAR Remote Sensing Work? (EYbhNSUnIdU)', 'NEON Science'],
         ['Visually Explained: Kalman Filters (IFeCIbljreY)', 'Visually Explained'],
         ['Tesla AI Day 2021 (j0z4FweCy4M)', 'Tesla'],
         ['A friendly introduction to distributed training (S1tN9a4Proc)', 'TensorFlow'],
         ['Platforms: Google Colab, REINFORCEjs GridWorld, phyphox, ping/tracert', 'Google; A. Karpathy; RWTH Aachen; OS tools']],
        [660, 300], color='blue', fs=20),
      N('All YouTube links have the form youtube.com/watch?v=<ID>; IDs, titles and creators were checked on 4 November 2026.', label='🔗 LINKS', color='blue'),
      floor=None, id='vsrc'),
    P(C9, 'IMAGE CREDITS',
      W('Cooking Mama artwork', 'Characters, dishes, ingredients and stickers: Cooking Mama series © Office Create Corp.; published on Nintendo platforms by Majesco Entertainment / Taito / 505 Games and others. ' + f'{len(SRC)} sprites were cut from the team\'s collected game images (folders: {", ".join(folders) if folders else "Assets_temp"}) by removing backgrounds and cropping. Low-quality images were not used.', 'pink', '🎨'),
      W('Fair-use statement', 'Used only for a non-commercial university class assignment, for illustration and commentary. Not for sale, not for public distribution. If the guide is ever shared beyond the class, the artwork must be replaced with original or licensed images.', 'yellow', '⚖'),
      W('Other design assets', 'Fonts: Fredoka and Press Start 2P (Google Fonts, SIL Open Font License); Segoe UI Emoji (Microsoft) for icons. Layout, wall tiles, gingham header, kitchen counter, windows and badges: drawn by the team. QR code: generated by the team for our repository.', 'mint', '🖌'),
      floor=['mama_happy', 'sticker_tasty'], id='credits'),
    P(C9, 'HOW WE COOKED THIS GUIDE',
      W('Contributions', 'Content: written from the team\'s five shared Markdown chapters (Main/Chapter-01 to 05), which every member reviewed. Visual identity, sprites and layout: Visual Design Lead with the team. Videos: selected and timestamped by the Video Lead. Accuracy check of all worked examples and answer keys: Quality Lead and Content Leads.', 'blue', '👩‍🍳'),
      W('AI-use statement', 'We used an AI assistant (Claude) to help turn our chapters into page layouts, draft kitchen analogies, write the image-processing scripts for sprite cut-outs, and check arithmetic. Every number was re-checked against our chapters\' worked examples. AI did not invent sources; all references come from our chapters. Details of prompts and edits are in the AI-Logs folder of our repository.', 'lav', '🤖'),
      W('Known limits', 'Tesla, Waymo and TPU figures change quickly and some are "reported" rather than independently verified. Numbers marked illustrative are chosen for teaching. Check current sources before quoting.', 'pink', '⚠'),
      floor=['mama_tray', 'papa', 'grandma'], id='how'),
    P(C9, 'RECIPE BOOK · ALL DISHES UNLOCKED!',
      TITLE('Thanks for cooking!', 84, 'yellow'),
      B('mama_cheer', 'You did it, Head Chef! You know which cook does the maths, how orders travel, how the pup learns, how the chef senses, and how to run the whole chain. Now try the Test Bank and see how many Gold stamps you get!'),
      S('fried_rice', 'xiaolongbao', 'bibimbap', 'donuts', h=130),
      W('Fengwei Fried Rice', 'Hot Wok, Smart Chips: We\'re Coooking!\nHardware, Networks & Behaviourist AI · Stage 2 Study Guide · November 2026\ngithub.com/Bvmhe/CS-AI-Group', 'mint', '🍳'),
      floor=[('mama_fry', 260), 'pup', 'boy_grin', 'girl_happy'], id='back'),
]
