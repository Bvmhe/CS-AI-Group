"""'See it' picture pages: diagrams from the chapter notes (Main/Chapter-0X/*.md, Mermaid blocks)
rendered to Study-Guide/assets/diagrams by tools/mermaid_png.py, placed right after the page they illustrate.

SEE[after_page_id] = (page title, [(diagram name, caption), ...])   (one or two diagrams per page)"""
import json, os
from kit import B, P, PIC, R

_DIR = 'Study-Guide/assets/diagrams'
_INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'diagrams', 'index.json')

SEE = {
    # Recipe 01
    '1.1c': ('ONE NEURON, STEP BY STEP', [('ch01_03', 'Worked example: a single neuron computed one multiply-add (MAC) at a time.')]),
    '1.2d': ('THE 2 × 2 CONVEYOR', [('ch01_12', 'A 2 × 2 systolic array: inputs flow right, partial sums flow down.'),
                                    ('ch01_13', 'The same example worked out: W = [[1, 2], [3, 4]], x = [5, 6].')]),
    '1.3b': ("AMDAHL'S LAW", [('ch01_21', 'Speed-up levels off when 10% of the job cannot be shared (p = 0.9).'),
                              ('ch01_22', 'The formula with numbers plugged in.')]),
    '1.3c': ('SHRINKING THE NUMBERS', [('ch01_25', 'Quantisation: FP32 weights mapped to 8-bit integers and back.'),
                                       ('ch01_24', 'Original vs dequantised weights: almost the same values.')]),
    '1.4b': ('THE ROOFLINE', [('ch01_29', 'The roofline for a 10 TFLOP/s chip with 500 GB/s of memory bandwidth.'),
                              ('ch01_31', 'Batch 1 vs batch 32: the same layer, memory-bound vs compute-bound.')]),
    '1.5c': ('FACE ID IN FIVE STEPS', [('ch01_37', 'Scan, process, infer on the Neural Engine, compare, unlock.')]),
    # Recipe 02
    '2.1e': ('PACKING THE BENTO', [('ch02_02', 'Encapsulation: each layer adds its own header around the data.'),
                                   ('ch02_03', 'Worked example: how much of a packet is useful data?')]),
    '2.2b': ('THE TCP HANDSHAKE', [('ch02_06', 'TCP: handshake, data with acknowledgements, then closing.')]),
    '2.2d': ('TIME TO FIRST BITE', [('ch02_07', 'Worked example: round trips add up before the first byte arrives.')]),
    '2.2e': ('FAST ROAD, SLOW DELIVERY', [('ch02_08', 'Only one window of data can be "in flight" per round trip.'),
                                          ('ch02_09', 'Worked example: a 1 Gbit/s link limited by a 64 KB window.')]),
    '2.2f': ('LOST BOXES', [('ch02_10', 'Worked example: how packet loss limits TCP throughput.')]),
    '2.3d': ('THE LATENCY BUDGET', [('ch02_12', 'End-to-end latency split into its parts (ms).'),
                                    ('ch02_13', 'Upload, network, inference and download, worked out.')]),
    '2.3e': ('THE EDGE SAVES THE ROAD', [('ch02_14', 'Worked example: 100 cameras, raw video vs results sent from the edge.')]),
    '2.4c': ('THE SLOWEST COOK', [('ch02_17', 'Tail latency: the chance that at least one server is slow.'),
                                  ('ch02_18', 'With more servers, a slow one becomes almost certain (p = 1%).')]),
    '2.4d': ('RUSH HOUR', [('ch02_19', 'Queueing: response time grows as utilisation rises.'),
                           ('ch02_20', 'Response time vs utilisation (service time 10 ms).')]),
    '2.4f': ('TRAINING TRAFFIC', [('ch02_21', 'Worked example: how much data each GPU must exchange.'),
                                  ('ch02_22', 'Training efficiency on different networks.')]),
    '2.5c': ('4G vs 5G vs EDGE', [('ch02_24', 'Latency budget of the street-sign translator (ms).'),
                                  ('ch02_25', 'The same request worked out step by step.')]),
    '2.5e': ('A TINY FORECAST', [('ch02_27', 'Worked example: exponential smoothing decides when a cell can sleep.')]),
    # Recipe 03
    '3.1d': ('THE LAW OF EFFECT', [('ch03_02', 'Worked example: rewarded actions grow stronger, punished ones weaker.')]),
    '3.2a': ('THE LOOP', [('ch03_03', 'Agent and environment: state, action, reward, repeat.')]),
    '3.2c': ('RETURN & DISCOUNT', [('ch03_04', 'Worked example: the same reward is worth less the later it comes.')]),
    '3.2d': ('VALUE OF EACH SQUARE', [('ch03_05', 'Worked example: value of each square in a corridor (γ = 0.9).'),
                                      ('ch03_06', 'How the values are computed, step by step.')]),
    '3.3d': ('Q-LEARNING STEP BY STEP', [('ch03_08', 'Worked example: Q-learning on the corridor (γ = 0.9, α = 0.5).')]),
    '3.4c': ('THE BANDIT', [('ch03_10', 'Worked example: a bandit simulation with epsilon-greedy.'),
                            ('ch03_11', 'Average reward for different epsilon values (1,000 pulls).')]),
    '3.4d': ('UCB', [('ch03_12', 'Worked example: the UCB rule favours the arm we know least about.')]),
    '3.5c': ('SCORING A STEP', [('ch03_14', 'Worked example: scoring three moments of the robot dog\'s walk.')]),
    '3.5d': ('A VIRTUAL KITCHEN', [('ch03_15', 'Why simulation: 100 million steps on one real robot would take years.'),
                                   ('ch03_16', '4,096 simulated robots in parallel cut weeks to minutes.')]),
    # Recipe 04
    '4.1c': ('PIXELS ON A CHILD', [('ch04_03', 'Worked example: how many pixels a child covers at a distance.'),
                                   ('ch04_05', 'The expected output at 50 m and 150 m.')]),
    '4.1d': ('TIME OF FLIGHT', [('ch04_06', 'Worked example: a laser pulse returns after 400 ns. How far away?')]),
    '4.1e': ('LiDAR DATA RATE', [('ch04_07', 'Worked example: how much data a 128-beam LiDAR produces.')]),
    '4.1f': ('ECHO TO DISTANCE', [('ch04_08', 'Worked example: echo to distance, and the error from temperature.')]),
    '4.2b': ('CAMERA DATA RATE', [('ch04_11', 'Worked example: the raw data rate of an 8-camera array.')]),
    '4.2d': ('DISTANCE WHILE THINKING', [('ch04_13', 'Worked example: distance travelled while thinking, at 0.1 s and 0.3 s of latency.')]),
    '4.2f': ('CAN THE COMPUTER KEEP UP?', [('ch04_15', 'Worked example: compute needed for a camera array vs what the chip offers.')]),
    '4.3c': ('RADAR PLUS CAMERA', [('ch04_18', 'Worked example: fusing two measurements by their confidence.'),
                                   ('ch04_19', 'Expected output: the fused estimate is better than either.')]),
    '4.3d': ('THE KALMAN FILTER', [('ch04_20', 'Worked example: one predict step and two updates.'),
                                   ('ch04_21', 'The uncertainty shrinks after each update.')]),
    '4.3f': ('DEPTH FROM TWO EYES', [('ch04_23', 'Worked example: stereo depth and why it gets worse with range.')]),
    '4.4d': ('CAN THE CAR STOP IN TIME?', [('ch04_25', 'Worked example: detection range decides the outcome.')]),
    '4.4e': ('BRAKE OR NOT?', [('ch04_26', 'Worked example: when should the car brake? Weighing the two kinds of error.')]),
    '4.5b': ('PIXELS TO STEERING', [('ch04_28', 'From 8 cameras to a driving decision.')]),
    '4.5d': ('HOW MANY TOPS?', [('ch04_30', 'Worked example: the TOPS of one accelerator, from its MACs and clock.')]),
    # Recipe 05
    '5.1c': ('ONE TRIP TO THE FRIDGE', [('ch05_02', 'Worked example: fusion saves memory traffic.')]),
    '5.2a': ('DOES IT FIT?', [('ch05_03', 'Worked example: does a 7B model fit on one chip for training?')]),
    '5.2c': ('KEEPING IN SYNC', [('ch05_05', 'Worked example: communication overhead for the 7B model.'),
                                 ('ch05_15', 'Why communicate hierarchically: fast links first, slow links last.')]),
    '5.3b': ('THE PRICE OF RECOMPUTATION', [('ch05_06', 'Worked example: recomputing saves memory but costs time.')]),
    '5.3c': ('ONE TOKEN AT A TIME', [('ch05_07', 'Worked example: the speed limit of one conversation.')]),
    '5.4a': ('WHAT PUE IS WORTH', [('ch05_09', 'Worked example: the energy bill with different PUE values.')]),
    '5.4b': ('THE FULL BILL', [('ch05_10', 'Worked example: FLOPs, time, energy and cost of the training run.')]),
    '5.4c': ('COST PER PLATE', [('ch05_08', 'Worked example: why utilisation is a cost issue.'),
                                ('ch05_11', 'The cost of serving 5,000 tokens per second.')]),
    '5.4d': ('WHEN STOVES BREAK', [('ch05_12', 'Worked example: failures in a 4,096-chip job.')]),
    '5.5c': ('THE 3D TORUS', [('ch05_14', 'Worked example: how much wraparound links help.')]),
}


# Mama's "spot it" question at the top of each See-it page: something to look for in the picture.
HOOK = {
    '1.1c': 'Follow the running total from Start to the end. How many MACs does one neuron need?',
    '1.2d': 'Watch each input pass through two cooks. Which weight does each input meet first?',
    '1.3b': 'Look at the curve: why does it flatten even with 128 cooks?',
    '1.3c': 'Compare the original and squeezed numbers. How big is the biggest error?',
    '1.4b': 'Find the bend in the roof. Which batch sits on the slope, and which on the flat top?',
    '1.5c': 'Count the steps from scan to unlock. Which one runs on the Neural Engine?',
    '2.1e': 'Count the wrappers! How many bytes of headers wrap one small message?',
    '2.2b': 'How many trips across the table before the first real data arrives?',
    '2.2d': 'Spot the difference: why does QUIC save a whole round trip?',
    '2.2e': 'The road is fast, so why is the delivery slow? Look at the window size.',
    '2.2f': 'Follow the three branches: what happens to speed as more boxes get lost?',
    '2.3d': 'Which slice of the latency is biggest? That is the one to fix first!',
    '2.3e': 'Compare the two roads: how much traffic does the edge save?',
    '2.4b': 'Even light needs time! How long does it take to cross the ocean?',
    '2.4c': 'One slow cook ruins the meal. Watch the chance grow with more servers.',
    '2.4d': 'Look at the curve near 100% busy. What happens to the wait?',
    '2.4f': 'Same job, two networks: how much training time is lost to talking?',
    '2.5c': 'Which path answers the street-sign photo fastest, and why?',
    '2.5e': 'Follow the loop: at which step does the base station go to sleep?',
    '3.1d': 'Trace the pup\'s scores trial by trial. Which trick wins, and why?',
    '3.2a': 'Find the three arrows. What does the agent send, and what comes back?',
    '3.2c': 'Same reward of 10, three answers. What makes them different?',
    '3.2d': 'Watch the value shrink as you walk away from the goal. By how much each step?',
    '3.3d': 'Follow one update: which numbers go into the new Q value?',
    '3.4c': 'Which epsilon earns the most? Too little or too much exploring both cost you!',
    '3.4d': 'Why does UCB pick the arm with the LOWER average? Look at the bonus.',
    '3.5c': 'Three moments of the walk: which one scores best, and what lost points?',
    '3.5d': 'One real robot vs 4,096 virtual pups: how much time is saved?',
    '4.1c': 'A child far away is only a few pixels wide. How many at 150 m?',
    '4.1d': 'The pulse goes there AND back. Where does the "÷ 2" come in?',
    '4.1e': 'Multiply along the arrows: how many points per second does the LiDAR make?',
    '4.1f': 'Cold air slows sound. How big is the error at 0 °C?',
    '4.2b': 'Cameras are hungry! How many bits per second do they produce together?',
    '4.2d': 'Every 0.1 s of thinking costs metres. How far does the car roll?',
    '4.2f': 'Can the computer keep up? Compare what is needed with what the chip offers.',
    '4.3c': 'Two sensors, two guesses. Why is the fused answer closer to the radar?',
    '4.3d': 'Watch P (the uncertainty) after each update. Up or down?',
    '4.3f': 'Two eyes give depth. Why does the error grow so fast with distance?',
    '4.4d': 'See it early, stop in time! Which detection range avoids the crash?',
    '4.4e': 'A false alarm vs a missed child: which costs more, and how does that move the line?',
    '4.5b': 'Follow the pixels from 8 cameras all the way to the steering wheel.',
    '4.5d': 'MACs × 2 × clock: can you get the TOPS before reading the answer?',
    '5.1c': 'Fused or not fused: how many trips to the fridge does each make?',
    '5.2a': '16 bytes per parameter! Does a 7B model fit on one chip?',
    '5.2c': 'Fast links first, slow links last. How much time does that save?',
    '5.3b': 'Recompute instead of storing: what do you pay in time?',
    '5.3c': 'Every token reads all the weights. What is the speed limit?',
    '5.4a': 'PUE is the overhead on top of computing. How many dollars does 0.5 save?',
    '5.4b': 'From FLOPs to dollars: follow each arrow of the bill.',
    '5.4c': 'An idle chip still costs money. Compare 35% and 70% busy.',
    '5.4d': 'Thousands of chips means something always breaks. How often?',
    '5.5c': 'Wraparound links shorten the trip. How many hops are saved?',
}
_HOSTS = ['mama_happy', 'mama_ok', 'mama_grin', 'mama_cheer', 'mama_stand']


def see_it(ch, pages, floor=('mama_book', 'girl_happy')):
    """Insert the 'See it' pages after the pages they illustrate. Skips diagrams that were not rendered."""
    have = json.load(open(_INDEX, encoding='utf-8')) if os.path.exists(_INDEX) else {}
    out = []
    for pg in pages:
        out.append(pg)
        pid = pg.get('id', '')
        if pid not in SEE:
            continue
        title, items = SEE[pid]
        items = [(n, c) for n, c in items if n in have]
        if not items:
            continue
        pill = f'{pid.split(".")[0]}.{pid.split(".")[1][0]} SEE IT: {title}'
        tall = [have[n]['size'][1] > 0.9 * have[n]['size'][0] for n, _ in items]
        pic = lambda n, c, h: dict(PIC(f'{_DIR}/{n}.png', c, h=h), label='📸 SEE IT')
        hook = lambda: B(_HOSTS[len(out) % len(_HOSTS)], '🔍 Spot it! ' + HOOK.get(pid, 'What does each box add? Follow the arrows.'))
        if len(items) == 1:
            layouts = [[pic(*items[0], 860)]]
        elif all(tall):                      # two tall diagrams: side by side
            layouts = [[R(*(pic(n, c, 820) for n, c in items))]]
        elif any(tall):                      # a tall one would be squashed: one diagram per page
            layouts = [[pic(n, c, 860)] for n, c in items]
        else:                                # two wide diagrams: stacked
            layouts = [[pic(n, c, 370) for n, c in items]]
        for k, blocks in enumerate(layouts):
            out.append(P(ch, pill if k == 0 else pill + ' (2)', hook(), *blocks,
                         floor=list(floor), id=pid + 's' + ('' if k == 0 else str(k + 1))))
    return out
