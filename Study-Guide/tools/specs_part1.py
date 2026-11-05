# Page specs for Canva pages 1-25 (Opening + Recipe 01). Text source: Study-Guide/content/00-opening.md and 01-recipe-hardware.md
# Layout helpers ---------------------------------------------------------------
HOST = lambda name: [name, 1590, 12, 260, 215]          # character in the top-right corner
TWO = [[80, 240, 870, 800], [970, 240, 870, 800]]         # two big cards
SIX = [[80, 250, 560, 370], [680, 250, 560, 370], [1280, 250, 560, 370],
       [80, 650, 560, 395], [680, 650, 560, 395], [1280, 650, 560, 395]]
FOUR = [[80, 250, 870, 380], [970, 250, 870, 380], [80, 660, 870, 380], [970, 660, 870, 380]]
THREE = [[80, 250, 570, 560], [675, 250, 570, 560], [1270, 250, 570, 560]]
THREE_BAND = THREE + [[80, 840, 1760, 200]]


def activity(name, title, sub, goal, prep, task, observe, reflect, feedback, host='mama_cheer'):
    return dict(name=name, top={0: title, 1: sub}, layout=SIX, body_max=22,
                cards=[['🎯 Goal', goal], ['🧺 Prep', prep], ['🔪 Task', task],
                       ['👀 Observe', observe], ['💭 Reflect', reflect], ["⭐ Mama's Feedback", feedback]],
                imgs=[HOST(host)])


def explain(name, title, sub, left, right, host='mama_pan_flip'):
    return dict(name=name, top={0: title, 1: sub}, layout=TWO, body_max=25, cards=[left, right], imgs=[HOST(host)])


PAGES = [
    # 1 cover --------------------------------------------------------------
    dict(name='cover', top={0: 'HOT WOK, SMART CHIPS\nWe\'re Coooking!', 1: 'Fengwei Fried Rice presents',
                            2: 'A cook-along, play-along study guide to Hardware, Networks & Behaviourist AI — taught by cooking.',
                            3: 'Stage 2 · Study Guide V1.0 · November 2026 · Introduction to Artificial Intelligence'},
         imgs=[['mama_basket', 110, 600, 300, 300]], delete_types=[]),
    # 2 team ---------------------------------------------------------------
    dict(name='team', top={0: 'The Kitchen Brigade', 1: 'Team Fengwei Fried Rice · Module: Hardware, Networks & Behaviourist AI',
                           2: 'Stage 2 · Study Guide V1.0 · November 2026 · Built from Shared Markdown Content V2.0 (Main/Chapter-01 … 05)'},
         body_max=24,
         cards=[['Who cooked this guide?',
                 '纪小佩 — Head Chef (CEO)\n林元甲 — Pantry Keeper (CIO)\n温俊强 — Recipe Tester (Content Lead)\n汤丽嘉 — Recipe Writer (Content Lead)\n'
                 '曾富良 — Plating Artist (Visual Design Lead)\n李秀春 — Camera Crew (Video Lead)\n黄施杰 — Taste Inspector (Quality Lead)'],
                ['Version', 'V1.0 · first complete version for Stage 2'],
                ['Source recipe book', 'Shared Markdown Content V2.0 · Main/Chapter-01 … 05'],
                ['Kitchen address', 'github.com/Bvmhe/CS-AI-Group']],
         imgs=[HOST('family_wide')]),
    # 3 menu ---------------------------------------------------------------
    explain('menu', "Today's Menu", 'Pick a dish — 01 → 05 is the chef-recommended route. Page numbers are on the right.',
            ['Main Courses',
             'Recipe 01 · The Kitchen Crew — AI Hardware ........ p.8\n'
             'Recipe 02 · Order Up! — Networks & AI Infrastructure ........ p.26\n'
             'Recipe 03 · Training the Kitchen Pup — Behaviourist AI & RL ........ p.44\n'
             'Recipe 04 · Chef\'s Senses — AI Sensors & Intelligent Systems ........ p.62\n'
             'Recipe 05 · Running the Restaurant Chain — Integration & Optimization ........ p.78'],
            ['Starters & Desserts',
             'Starter · How to Play ........ p.4\nStarter · Learning Goals ........ p.5\nStarter · Score Card ........ p.6\nStarter · Meet the Kitchen ........ p.7\n'
             "Dessert · Mama's Pantry (glossary) · Word Soup · Videos · References · Credits ........ p.94\n"
             'Treat · The Recipe Book (unlockable dishes) ........ p.104'],
            host='mama_basket'),
    # 4 how to play ----------------------------------------------------------
    dict(name='howto', top={0: 'How to Play', 1: 'Every page is a game screen. Here is how a round works.'}, body_max=22,
         layout=[[80, 250, 330, 520], [432, 250, 330, 520], [784, 250, 330, 520], [1136, 250, 330, 520], [1488, 250, 352, 520], [80, 800, 1760, 240]],
         cards=[['1 · PICK', 'Choose a Recipe from the Menu. 01 → 05 is best: each one builds on the last.'],
                ['2 · COOK ALONG', "Each Step: Mama's Problem → Predict → Try it → Look → Why it works → Use it → Taste Test."],
                ['3 · PLAY', 'Mini-games have six parts: Goal · Prep · Task · Observe · Reflect · Mama\'s Feedback. Try before you peek!'],
                ['4 · STAMP', 'Score Card: Gold = right first time · Silver = right after a hint · Bronze = read the answer, then redo it.'],
                ['5 · UNLOCK', 'Finish a Recipe → unlock its page in the Recipe Book at the back!'],
                ['Kitchen tools you need',
                 'Pen, paper and a calculator. A phone or laptop with internet for the 🌐 Kitchen Lab activities — every lab has an offline back-up (sample output + questions), so you can always finish.']],
         imgs=[HOST('mama_arms_crossed')]),
    # 5 goals ----------------------------------------------------------------
    dict(name='goals', top={0: 'What You Will Be Able to Cook', 1: 'Learning goals, who this is for, and what to know first'}, body_max=24,
         layout=[[80, 250, 1100, 790], [1220, 250, 620, 560]],
         cards=[['LEARNING GOALS',
                 '1. Name CPU, GPU, TPU/NPU and memory, and explain what each does for AI. (Recipe 01)\n'
                 '2. Explain how networks move AI data, and choose cloud vs edge using latency and bandwidth. (Recipe 02)\n'
                 '3. Describe how an agent learns from rewards, and explain exploration vs exploitation. (Recipe 03)\n'
                 '4. Compare cameras, LiDAR and ultrasonic sensors, and explain sensor fusion. (Recipe 04)\n'
                 '5. Explain how hardware, networks and software are combined and optimised at scale. (Recipe 05)'],
                ['BEFORE YOU START',
                 'For 1st/2nd-year CS & IT students.\nYou need: basic arithmetic, what a variable and a loop are, a rough idea of what a CPU and the internet are.\nNo electronics or advanced maths.\nTime: about 1.5–2 hours per Recipe.']],
         delete_types=['shape'], imgs=[['kid_boy', 1400, 830, 210, 210], HOST('mama_proud')]),
    # 6 score card -------------------------------------------------------------
    dict(name='score', top={0: 'My Score Card', 1: 'Stamp one medal per mini-game: 🥇 Gold · 🥈 Silver · 🥉 Bronze'}, body_max=19,
         layout=[[80, 250, 560, 380], [680, 250, 560, 380], [1280, 250, 560, 380], [80, 660, 560, 380], [680, 660, 560, 380], [1280, 660, 560, 380]],
         cards=[['RECIPE 01', '1.1 Stir-Fry Maths ○\n1.2 Hire the Chef ○\n1.3 Conveyor Belt ○\n1.4 Chopping Race ○\n1.5 Pantry Run ○\n1.6 Batch Cooking ○'],
                ['RECIPE 02', '2.1 Bento Packing ○\n2.2 Careful vs Shouting Waiter ○\n2.3 Delivery Timer ○\n2.4 Truck or Kitchen? ○\n2.5 Kitchen Lab: ping ○'],
                ['RECIPE 03', '3.1 Good Pup! ○\n3.2 Noodle Stall Gamble ○\n3.3 Pup Gridworld ○\n3.4 Reward Hacking ○\n3.5 Kitchen Lab: GridWorld ○'],
                ['RECIPE 04', '4.1 Sensor Match ○\n4.2 Is the Steak Done? ○\n4.3 Kitchen Lab: phone sonar ○'],
                ['RECIPE 05', '5.1 Fusion Saves Trips ○\n5.2 Does It Fit? ○\n5.3 Kitchen Manager ○\n5.4 Energy Bill ○'],
                ['MY RANK', '0–10 Gold = Kitchen Helper\n11–20 Gold = Line Cook\n21+ Gold = HEAD CHEF!\n\nTotal Gold: ____']],
         imgs=[HOST('mama_cheer')]),
    # 7 cast -----------------------------------------------------------------------
    dict(name='cast', top={0: 'Meet the Kitchen', 1: 'You are the new apprentice. By the end, you run the whole restaurant chain!'}, body_max=21,
         layout=[[80, 250, 420, 790], [525, 250, 420, 790], [970, 250, 420, 790], [1415, 250, 425, 790]],
         cards=[['MAMA', 'Your teacher. Cheers when you get it right, gets fire-eyes when something burns — then always helps you fix it.'],
                ['THE KIDS', 'Ask the "but why?" questions a beginner would ask. If you wonder it, they say it.'],
                ['PAPA', 'Loves shortcuts. When Papa says "Easy! Just…", watch out: it is a common mistake.'],
                ['THE KITCHEN PUP', 'Learns tricks from treats. Star of Recipe 03 (Behaviourist AI).']],
         delete_types=['shape'],
         imgs=[['mama_arms_crossed', 140, 560, 300, 460], ['kid_boy', 585, 560, 300, 460], ['papa_casual', 1030, 560, 300, 460], ['pup_chef', 1470, 600, 320, 400]]),
    # 8 Recipe 01 opener ----------------------------------------------------------------
    dict(name='r01_open', top={0: 'RECIPE 01', 1: 'The Kitchen Crew —\nAI Hardware Architecture',
                               2: 'Mama\'s Problem: "1,000 plates of fried rice in one minute! Who should cook — one master chef, a thousand helpers, or a machine?"'},
         top_size={2: 30}, delete_types=[], imgs=[['dish_fried_rice', 1240, 560, 560, 420], ['mama_pan_flip', 1500, 120, 300, 420]]),
    # 9 step 1 ----------------------------------------------------------------------------
    explain('r01_s1', 'Step 1 · What Does AI Cook?', 'Predict first: does a neural network "think" with (a) logic puzzles (b) one tiny maths move, billions of times (c) internet searches?',
            ['The answer is (b): the MAC',
             'Almost every layer multiplies inputs by learned weights and adds them up.\n\nOne "multiply, then add to the running total" = a MAC (multiply-accumulate). 🍳 Like adding one ingredient × its amount to the wok.\n\n'
             'Scale it up: a layer with 1,024 inputs and 4,096 outputs = 1,024 × 4,096 = 4,194,304 MACs — for ONE input. Big language models need billions to trillions per answer.'],
            ['Worked example: one neuron',
             'weights w = [0.5, −1.0, 2.0]   inputs x = [4, 3, 1]\n\ntotal = 0\n+ 0.5 × 4 = 2.0\n+ (−1.0 × 3) = −1.0\n+ 2.0 × 1 = 1.0  → output 1.0 (3 MACs)\n\n'
             '🔬 Real tech: 1 MAC = 2 FLOPs (one multiply + one add). A matrix groups the weights so hardware can do many MACs at once; a tensor is a many-dimensional matrix — the "T" in TPU.']),
    # 10 MG 1.1 -------------------------------------------------------------------------------
    activity('r01_g1', 'Mini-game 1.1 · Stir-Fry Maths', 'Cook a neuron by hand, then count the work in a whole layer.',
             'Compute a neuron output with MACs and count the work in a layer.',
             'Pen, paper, calculator. The worked example on the previous page.',
             '(1) w = [2, −1, 0.5], x = [1, 4, 6]: write each MAC step and the output.\n(2) A layer has 512 inputs and 2,048 outputs. How many MACs per input? How many FLOPs?',
             'Write your running total after every single step.',
             'Why would a chip that does MACs very fast speed up almost the whole model?',
             '(1) 0 → 2 → −2 → 1.0 (3 MACs).\n(2) 512 × 2,048 = 1,048,576 MACs = 2,097,152 FLOPs.\nReflect: MACs are nearly all of a network\'s work, so speeding them up speeds up nearly everything.'),
    # 11 step 1b ------------------------------------------------------------------------------
    explain('r01_s1b', 'Step 1b · Writing the Recipe vs Cooking the Order', 'Training and inference use the same model in two very different ways.',
            ['📖 Training = writing the recipe',
             'Adjust the model\'s weights using lots of data.\n• Rare, huge jobs (days to months) in data centres\n• Higher precision (FP32 / BF16 / FP16)\n• Big batches\n\n'
             '🍽 Inference = cooking each order\nUse the finished weights to answer a request.\n• Millions of times a day\n• Latency, energy and cost per request matter\n• Often batch size 1 and low precision (INT8, 4-bit)'],
            ['Taste Test',
             'Training or inference?\n(a) Face ID unlocking a phone\n(b) fine-tuning a model on 100 GPUs for 3 weeks\n(c) a chatbot answering 10,000 requests per second\n→ Answers: (a) inference (b) training (c) inference\n\n'
             '🔬 Precision = bits per number: FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte. Fewer bits → less memory and energy, a little less accuracy.\n\n'
             'Papa says: "Easy! The quoted peak speed is what I\'ll get." ✗ Peak = every unit busy every cycle. Real programs reach much less (utilisation).'],
            host='papa_casual'),
    # 12 crew ---------------------------------------------------------------------------------
    dict(name='r01_crew', top={0: 'Step 2 · Meet the Crew', 1: 'Rule of the kitchen: the more specialised the cook, the more efficient at that dish — and the less useful for anything else.'},
         layout=FOUR, body_max=24,
         cards=[['CPU · THE HEAD CHEF', 'A few very skilled cores that can cook ANY dish, one step at a time each. Flexible, low latency. Best for mixed tasks, data loading and small models.'],
                ['GPU · 1,000 LINE COOKS', 'Thousands of simple cores all doing "chop these onions" together. High parallel throughput. Best for training and serving large models.'],
                ['TPU · THE DUMPLING CONVEYOR', "Google's chip built around a grid of MAC units. Highest efficiency for neural-network maths, but the least flexible."],
                ['NPU · THE HOME RICE COOKER', 'A small, low-power AI engine inside a phone or laptop chip (Apple Neural Engine, Qualcomm Hexagon). 🔬 CPUs spend chip area on caches and control; GPUs on many arithmetic units; TPUs on the MAC grid.']],
         imgs=[HOST('family_wide')]),
    # 13 MG 1.2 ---------------------------------------------------------------------------------
    activity('r01_g2', 'Mini-game 1.2 · Hire the Chef', 'Four restaurants need a cook. Pick the right chip and defend your choice.',
             'Match a chip to a workload and justify it.',
             'The crew cards on the previous page.',
             'Choose CPU, GPU, TPU or NPU:\n(a) web server running mixed tasks\n(b) training a large image model\n(c) fixed, high-volume inference service in a data centre\n(d) voice assistant in a battery smart speaker — rank NPU, CPU, GPU',
             'For each, write the ONE feature that decided it (flexibility, parallel throughput, efficiency per watt …).',
             'Why might a start-up still choose a GPU even though the TPU was faster in Google\'s tests?',
             '(a) CPU — flexibility. (b) GPU (or TPU on Google Cloud). (c) TPU / fixed accelerator — best per watt. (d) NPU > CPU > GPU — a GPU drains a small battery.\nReflect: unsupported ops fall back to the CPU; team already uses GPU software; availability/cost; memory-bound jobs gain little.',
             host='mama_proud'),
    # 14 systolic -------------------------------------------------------------------------------
    explain('r01_s2b', 'Step 2b · The Dumpling Conveyor (Systolic Array)', 'Pass the dumplings hand-to-hand instead of running back to the fridge each time.',
            ['How it works',
             'A systolic array is a grid of MAC cells. Each cell keeps ONE weight. Inputs and partial sums are passed to the neighbour — like dumplings on a conveyor — instead of being fetched from memory again and again.\n\n'
             '🔬 First TPU (Jouppi et al., 2017): 65,536 8-bit MACs, 92 TOPS peak, ~15–30× faster and ~30–80× more TOPS per watt than 2015 CPUs/GPUs — but 4 of 6 workloads were limited by memory bandwidth. A historical comparison, not today\'s ranking.'],
            ['Worked example: a 2 × 2 conveyor',
             'W = [[1, 2], [3, 4]], x = [5, 6]\n\nCycle 1: x1 = 5 enters column 1\n→ 1 × 5 = 5   and   3 × 5 = 15\n\nCycle 2: x2 = 6 reaches column 2\n→ 2 × 6 + 5 = 17   and   4 × 6 + 15 = 39\n\nResult y = [17, 39] with 4 MACs.\n\n'
             'Memory reads: systolic = 4 weights + 2 inputs = 6; without passing = 8. The saving grows with array size.'],
            host='mama_and_pup'),
    # 15 MG 1.3 -----------------------------------------------------------------------------------
    activity('r01_g3', 'Mini-game 1.3 · Conveyor Belt Trace', 'Be the conveyor belt: trace a 2 × 2 systolic array cycle by cycle.',
             'Trace a weight-stationary systolic array cycle by cycle.',
             'The worked example on the previous page.',
             'W = [[2, 1], [0, 3]], x = [4, 5].\nCycle 1: row-1 sum = __, row-2 sum = __\nCycle 2: row-1 sum = __, row-2 sum = __\nCount memory reads with and without neighbour passing.',
             'Which numbers were read from memory, and which were handed along?',
             'Why does reusing data matter more than adding more MAC cells?',
             'Cycle 1: 2×4 = 8 and 0×4 = 0. Cycle 2: 1×5 + 8 = 13 and 3×5 + 0 = 15 → y = [13, 15].\nReads: 6 (systolic) vs 8 (no reuse).\nReflect: fetching data costs far more energy and time than multiplying it (Step 4).'),
    # 16 parallel -----------------------------------------------------------------------------------
    explain('r01_s3', 'Step 3 · Cooking in Parallel', "Mama's Problem: one chef chops 4,194,304 onions one by one; 4,096 helpers chop 4,096 at a time. How much faster?",
            ['The race',
             '• Idealised CPU core: 1 MAC per cycle at 3 GHz → 4,194,304 ÷ 3×10⁹ ≈ 1.4 ms\n• Idealised accelerator: 4,096 MAC units at 1 GHz → 4,194,304 ÷ 4.096×10¹² ≈ 1.0 µs\n• ≈ 1,365× faster, even with a slower clock — it does 4,096 MACs per cycle.\n\n'
             'Four levels of parallelism: instruction-level (one core overlaps steps) · data-level SIMD/SIMT (GPU warps) · matrix-level (systolic arrays) · across chips (Recipes 02 & 05).'],
            ['But… four things stop the speed-up',
             '1. Amdahl\'s law — the part that can\'t be shared caps everything\n2. Idle units — not enough independent MACs\n3. Feeding the units — memory can\'t keep up\n4. Communication — chips swapping results\n\n'
             '🔬 Other boosters: quantisation (smaller numbers), specialised units (Tensor Cores, Neural Engine), batching (reuse weights).'],
            host='kid_boy'),
    # 17 MG 1.4 ----------------------------------------------------------------------------------------
    activity('r01_g4', "Mini-game 1.4 · Chopping Race (Amdahl's Law)", 'Add more cooks… and discover the one wok that slows everyone down.',
             "Discover why adding cooks stops helping.",
             'speed-up = 1 ÷ ((1 − p) + p ÷ N)\np = share of work that can be split; N = number of cooks.',
             'An order is 90% chopping (p = 0.9) and 10% waiting for the one wok. PREDICT the speed-up with 100 cooks. Then fill a table for N = 1, 2, 4, 8, 16, 100, ∞ and plot it.',
             'N=1 → 1.0× · 2 → 1.8× · 4 → 3.1× · 8 → 4.7× · 16 → 6.4× · 100 → 9.2× · ∞ → 10×. Where does the curve flatten?',
             'A job is 80% parallel. Speed-up on 8 units? Limit with unlimited units?',
             '1 ÷ (0.2 + 0.8 ÷ 8) = 1 ÷ 0.3 ≈ 3.3×; limit 1 ÷ 0.2 = 5×.\nThe serial 20% (the single wok!) caps everything. In AI pipelines the "wok" is data loading, pre-processing or steps that cannot be split.',
             host='mama_fire_eyes'),
    # 18 memory -------------------------------------------------------------------------------------------
    explain('r01_s4', 'Step 4 · The Pantry Problem (Memory Hierarchy)', 'Moving food costs more than cooking it. Keep the ingredients on the chopping board!',
            ['From hands to supermarket',
             '✋ In your hands = registers (tiny, instant)\n🔪 Chopping board = cache / on-chip SRAM\n🧊 Fridge across the kitchen = DRAM / HBM\n🛒 Supermarket = storage (huge, slow)\n\n'
             'Faster memory is smaller and more expensive; bigger memory is slower and cheaper. A good cook plans so everything needed is already on the board — good AI hardware and software do the same.'],
            ['The surprising fact (Horowitz, 2014)',
             '45 nm chip, energy per operation:\n• 32-bit multiply ≈ 3.7 pJ\n• Read from small SRAM ≈ 5 pJ\n• Read from DRAM ≈ 640 pJ\n\n→ A DRAM read costs ≈ 173× a multiply and ≈ 128× an SRAM read.\n\n'
             '🔬 HBM = DRAM stacked right next to the chip for very high bandwidth. Unified memory (Apple silicon) = CPU, GPU and NPU share one pool, so nothing is copied.'],
            host='mama_basket'),
    # 19 MG 1.5 --------------------------------------------------------------------------------------------
    activity('r01_g5', 'Mini-game 1.5 · Pantry Run', 'How far is each ingredient — and how big is the shopping bag?',
             'Feel why data movement and number size matter.',
             'The energy table on the previous page; 1 byte = 8 bits.',
             '(1) Rank, cheapest first: DRAM read, register access, small SRAM read.\n(2) A 7-billion-parameter model: memory for its weights in FP16? INT8? 4-bit?\n(3) A chip doubles its MAC units but keeps the same bandwidth. Effect on a memory-bound job?',
             'Compare your answers to a phone\'s memory (≈ 8 GB, shared with other apps).',
             'Which helps a memory-bound workload more: more MACs, or smaller weights?',
             '(1) register < SRAM (~5 pJ) < DRAM (~640 pJ).\n(2) 14 GB · 7 GB · 3.5 GB.\n(3) Almost nothing — the units just wait for data.\nReflect: smaller weights (fewer bytes to move).'),
    # 20 roofline -------------------------------------------------------------------------------------------
    explain('r01_s4b', 'Step 4b · Cook Speed vs Delivery Speed (Roofline)', 'Is the cook too slow — or are the groceries arriving too slowly?',
            ['The roofline model',
             'Arithmetic intensity = FLOPs ÷ bytes moved (how much cooking per bag of groceries).\n\nAttainable = min(peak compute, intensity × memory bandwidth). Plotted, it looks like a roof: a slope (memory limit) meeting a flat top (compute limit).\n\n'
             'Ridge point = peak ÷ bandwidth. Below it: memory-bound (cook waits). Above it: compute-bound (cook is busy).'],
            ['Worked example: one layer',
             'Chip: 10 TFLOP/s, 500 GB/s → ridge 20 FLOP/byte. Layer 1,024 × 4,096 in FP16:\n• Batch 1: intensity ≈ 1.0 → 0.5 TFLOP/s = 5% of peak (memory-bound; ~95% of the time waiting)\n• Batch 32: intensity ≈ 30.8 → 10 TFLOP/s = 100% (compute-bound)\n\n'
             '🍱 Batching = cook 32 plates from one grocery delivery: weights fetched once, used 32 times.\n⚠ But each order may wait longer (latency). Phones answer one user at a time → memory-bound.'],
            host='mama_proud'),
    # 21 MG 1.6 -----------------------------------------------------------------------------------------------
    activity('r01_g6', 'Mini-game 1.6 · Batch Cooking', 'Find the batch size where the kitchen stops waiting for groceries.',
             'Use the roofline to decide memory-bound vs compute-bound.',
             'Same chip (10 TFLOP/s, 500 GB/s, ridge 20). Layer: 512 inputs × 2,048 outputs, FP16. Weights fetched once per batch; inputs/outputs once per sample.',
             '(1) weights and weight bytes (2) FLOPs at batch 1 (3) intensity at batch 1, 8, 32 — memory- or compute-bound? (4) attainable TFLOP/s and % of peak (5) smallest compute-bound batch',
             'Record a table: batch | intensity | bound | attainable | % peak.',
             'A voice assistant answers one user at a time within milliseconds. What hardware would you choose?',
             '(1) 1,048,576 weights = 2,097,152 bytes (2) 2,097,152 FLOPs (3–4) b1 ≈ 1.0 memory 0.5 TF 5% · b8 ≈ 7.9 memory 3.9 TF 39% · b32 ≈ 29.7 compute 10 TF 100% (5) batch 22.\nReflect: batch 1 is memory-bound → high bandwidth, on-chip memory, low-precision weights (an NPU).'),
    # 22 case study --------------------------------------------------------------------------------------------
    dict(name='r01_case', top={0: 'Step 5 · Case: The Home Rice Cooker (Apple Neural Engine)', 1: 'Training happens in the big data-centre kitchen; the finished recipe is cooked at home, on your phone.'},
         layout=THREE_BAND, body_max=22,
         cards=[['What is it?', 'The NPU inside Apple\'s chips. First shipped in 2017 in the A11 Bionic (iPhone X), described as 600 billion operations per second. It shares one system-on-a-chip and unified memory with the CPU and GPU; Core ML decides where each part of a model runs.'],
                ['Face ID: sense → decide → act', '1 Scan (TrueDepth camera)\n2 Turn the face into numbers\n3 AI inference on the Neural Engine\n4 Compare with your saved face\n5 Unlock\nFace data stays on the device (Secure Enclave).'],
                ['Pros & cons', '✅ fast · energy-efficient · private · real-time · works offline\n\n❌ limited compute and memory · not every operation is supported (falls back to the CPU) · big models still need the cloud'],
                ['Why it fits Recipe 01', 'Face ID runs at batch size 1 → memory-bound. So the Neural Engine relies on on-chip memory, unified memory and low-precision (INT8/FP16) weights — Steps 1–4 in one real device.']],
         imgs=[HOST('mama_arms_crossed')]),
    # 23 taste test ---------------------------------------------------------------------------------------------
    dict(name='r01_quiz', top={0: 'Taste Test 01 🍽', 1: 'Choose A, B or C for each dish, then turn the page to see if Mama is proud.'},
         layout=THREE_BAND, body_max=24,
         cards=[['Q1 🍙', 'Face ID processes one face at a time. On the roofline it is most likely…\n\nA) compute-bound\nB) memory-bound\nC) network-bound'],
                ['Q2 🍜', 'A 3-billion-parameter model quantised to 4-bit needs about…\n\nA) 12 GB\nB) 6 GB\nC) 1.5 GB'],
                ['Q3 🥟', 'One unsupported operation sits in the middle of a model. Why can it make the NPU slow?\n\nA) the NPU overheats\nB) data must move to the CPU and back\nC) Core ML deletes the operation'],
                ['Scoring', 'Write your answers here: Q1 __  Q2 __  Q3 __   →   answers on the next page. All right = 🥇 Gold.']],
         imgs=[HOST('mama_cheer')]),
    # 24 answers ---------------------------------------------------------------------------------------------------
    dict(name='r01_ans', top={0: 'Taste Test 01 · Answers', 1: 'All right? Mama is proud! Missed one? Mama will fix it — re-read Steps 4–5 and try again for Silver.'},
         layout=THREE_BAND, body_max=24,
         cards=[['Q1 = B', 'Batch 1 means low arithmetic intensity: the weights are fetched for just one input, so memory is the limit.'],
                ['Q2 = C', '3 × 10⁹ × 0.5 bytes = 1.5 GB. It fits on a phone with 8 GB of shared memory — but it is still a big share.'],
                ['Q3 = B', 'Each switch to the CPU and back moves data between processors and loses the accelerator\'s efficiency. The data movement, not the operation, is the cost.'],
                ['Mama says', '"Delicious! Stamp your Score Card, then serve the Recipe Card on the next page."']],
         imgs=[['banner_delicious', 1330, 860, 300, 110], ['mama_fire_eyes', 1640, 845, 170, 190], HOST('mama_proud')]),
    # 25 recipe card -------------------------------------------------------------------------------------------------
    dict(name='r01_card', top={0: 'Recipe Card 01 · The Kitchen Crew', 1: 'Cut out and keep: everything from Recipe 01 on one card.'}, body_max=23,
         layout=[[80, 250, 1150, 790], [1270, 250, 570, 520]],
         cards=[['METHOD (big ideas)',
                 '1. AI is mostly MACs — speed those up and you speed up the model.\n2. Specialised chips trade flexibility for efficiency (CPU → GPU → TPU/NPU).\n'
                 '3. Parallelism helps until the serial part, idle units, memory or communication get in the way (Amdahl).\n4. Moving data costs more than computing on it — keep ingredients on the chopping board (memory hierarchy, roofline, batching).\n'
                 '5. On-device AI (NPU) wins on privacy, latency and battery, but is limited by memory and supported operations.'],
                ['INGREDIENTS (key terms)', 'MAC · FLOP · training / inference · CPU / GPU / TPU / NPU · systolic array · Amdahl\'s law · memory hierarchy · arithmetic intensity · roofline · batching · quantisation\n\n🔓 Unlocked: Recipe Book page 1!']],
         delete_types=['shape'], imgs=[['dish_dumplings', 1300, 790, 300, 230], ['banner_well_done', 1610, 830, 230, 150]]),
]
