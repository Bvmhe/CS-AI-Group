"""Curated videos (links, titles, creators, lengths and chapter times checked on YouTube, 4 Nov 2026)."""
from kit import *

COL = ['yellow', 'mint', 'pink', 'blue']


def V(title, creator, vid, length, why, times, task, limits):
    return dict(title=title, creator=creator, url=f'youtube.com/watch?v={vid}', length=length,
                why=why, times=times, task=task, limits=limits)


def pages(ch, videos, floors):
    out = []
    for k in range(0, len(videos), 2):
        blocks = []
        for j, v in enumerate(videos[k:k + 2]):
            blocks.append(W(v['title'],
                            f'Creator: {v["creator"]} · {v["length"]}\nLink: {v["url"]}\n'
                            f'Why watch: {v["why"]}\nTimestamps: {v["times"]}\n'
                            f'Task: {v["task"]}\nLimitations: {v["limits"]}', COL[(k + j) % 4], '🎬'))
        out.append(P(ch, f'WATCH & LEARN · RECIPE 0{ch}' + (f' ({k // 2 + 1}/{(len(videos) + 1) // 2})' if len(videos) > 2 else ''),
                     *blocks, floor=floors[(k // 2) % len(floors)], id=f'{ch}.vid{k // 2 + 1}'))
    return out


CH1 = [
    V('But what is a neural network?', '3Blue1Brown (Deep learning ch. 1)', 'aircAruvnKk', '18:40',
      'Shows, with animation, that a network is layers of weighted sums: the MACs of 1.1.',
      '2:42 neurons · 3:35 layers · 11:34 counting weights and biases',
      'At 11:34, count the weights of the 784-16-16-10 network yourself and check his total.',
      'About recognising digits, not hardware; no chips are discussed.'),
    V('How do Graphics Cards Work? Exploring GPU Architecture', 'Branch Education', 'h9Z4oGN89MU', '28:29',
      'A 3-D tour inside a real GPU: thousands of cores, memory chips and SIMD.',
      '2:15 GPU vs CPU · 4:56 GA102 architecture · 8:48 CUDA core · 12:04 GDDR6X/GDDR7 memory',
      'Write down how many cores the GPU has and link it to "1,000 Line Cooks" (1.2).',
      'Focuses on graphics/gaming; little about AI training or TPUs. Includes a sponsor segment.'),
    V('Tensor Processing Units (TPUs)', 'Google Cloud', 'eyAjbgkBdjU', '1:12',
      'A one-minute official summary of why Google built an AI-only chip.',
      'Whole video (about 1 min)',
      'List two claims it makes and check each against 1.2 ("What the first TPU measured").',
      'Marketing video by the chip\'s maker: no numbers to test, not independent.'),
    V('A very short intro to the Roofline model', 'NHR@FAU', 'IrkNZG8MJ64', '18:36',
      'A university HPC centre explains intensity, bandwidth and the roof, exactly as in 1.4.',
      'Whole video; pause at each roofline plot',
      'Redraw the roof for 10 TFLOP/s and 500 GB/s and mark batch 1 and batch 32 from 1.4.',
      'Aimed at HPC programmers; uses CPU examples and some jargon beyond this guide.'),
]
CH2 = [
    V('How does the INTERNET work? | ICT #2', 'Lesics (Sabin Civil Engineering channel)', 'x3c1ih2NJEg', '9:00',
      'Animated path of data through cables, routers, IP addresses and DNS (2.1-2.2).',
      'Whole video (9 min)',
      'Draw the path a YouTube video takes to your phone and label each hop.',
      'Simplified; no TCP/UDP detail, no AI workloads.'),
    V('What is bandwidth? How is it different than latency?', 'internet-class', 'QLJCqY-7Vrw', '3:24',
      'A short, clear split between "how much" and "how fast" (2.4).',
      'Whole video (3 min)',
      'Give one AI example that needs high bandwidth and one that needs low latency.',
      'Very short; no numbers for real networks.'),
    V('Inside a Google data center', 'Google Workspace', 'XZmGGAbHqa0', '5:28',
      'See the racks, cooling and security of the "cloud kitchen" of 2.3 and Recipe 05.',
      'Whole video (5 min)',
      'Spot three things a data centre needs besides computers (power, cooling, security…).',
      'Company tour video; older footage; shows no AI accelerators.'),
]
CH3 = [
    V('Reinforcement Learning: Machine Learning Meets Control Theory', 'Steve Brunton', '0MNVhXEX9to', '26:00',
      'A clear lecture on agent, environment, reward, MDPs and Q-learning (3.2-3.3).',
      '3:34 RL overview · 12:32 Markov decision process · 13:33 credit assignment · 21:50 Q-learning',
      'At 21:50, compare his Q-update with the 0.2 → 0.325 worked example.',
      'University level; some equations go beyond this guide.'),
    V('Markov Decision Processes - Computerphile', 'Computerphile', '2iF9PRriA7w', '17:40',
      'A friendly whiteboard explanation of states, actions, rewards and policies.',
      'Whole video; pause at each diagram',
      'Write the S, A, R and policy for "the pup fetching an ingredient".',
      'Talk format, no code; long for a single idea.'),
    V('AI Learns to Walk (deep reinforcement learning)', 'AI Warehouse', 'L_4BPjLBF4E', '8:36',
      'Watch a simulated creature go from falling to walking: rewards and shaping in action (3.5).',
      'Whole video (9 min)',
      'Note one moment where the agent "hacks" the reward and suggest a better reward.',
      'Entertainment style; simulation only (no sim-to-real); training details are not shown.'),
]
CH4 = [
    V('How do self-driving cars "see"?', 'TED-Ed (Sajan Saini)', 'PRg5RNU_JLk', '5:25',
      'An animated intro to cameras, radar and LiDAR, the sensors of 4.1.',
      'Whole video (5 min)',
      'Fill a 3-column table: sensor · what it measures · one weakness.',
      'Beginner level; no fusion maths; does not cover Tesla\'s camera-only choice.'),
    V('How Does LiDAR Remote Sensing Work?', 'NEON Science', 'EYbhNSUnIdU', '7:42',
      'Time of flight and point clouds explained by a science organisation (4.1).',
      'Whole video (8 min)',
      'Use distance = c × t ÷ 2 for one example shown in the video.',
      'Airborne mapping LiDAR, not car LiDAR.'),
    V('Visually Explained: Kalman Filters', 'Visually Explained', 'IFeCIbljreY', '11:18',
      'Animated predict-update cycle: the exact steps of 4.3.',
      '0:00 intro · 4:30 Kalman filters · 5:37 prediction step · 7:14 update step',
      'At 7:14, match each symbol to x, P, K, z and R from the worked example.',
      'Uses matrix notation; beginners can skip the last minutes.'),
    V('Tesla AI Day 2021', 'Tesla', 'j0z4FweCy4M', '3:03:20',
      'Primary source for 4.5: Tesla\'s own explanation of camera-only vision and BEV fusion.',
      '46:54 event begins · 48:44 Tesla Vision · 1:13:12 planning and control · 1:35:15 simulation',
      'Watch 48:44-1:13:12 and list three claims; mark which are "company claims".',
      'Very long; a company presentation, not independent evidence; 2021 technology.'),
]
CH5 = [
    V('A friendly introduction to distributed training', 'TensorFlow (ML Tech Talks)', 'S1tN9a4Proc', '24:20',
      'Data vs model parallelism and synchronous all-reduce, as in 5.2.',
      '0:37 why distributed · 1:49 data vs model parallelism · 6:05 synchronous data parallelism · 18:20 asynchronous',
      'At 6:05, connect the ring all-reduce to traffic = 2(N−1)/N × S.',
      'Framework-focused (TensorFlow); no cost or energy discussion.'),
    V('Scaling AI with Google Cloud\'s TPUs', 'Google Cloud Tech', '8zBaa20mvoc', '5:54',
      'HBM, inter-chip interconnect, pods and multislice: the TPU case study of 5.5 in 6 minutes.',
      '0:26 TPUs explained · 1:11 HBM · 2:18 ICI and resiliency · 3:44 multislice · 4:40 frameworks',
      'At 3:44, explain why multislice needs hierarchical communication (5.5).',
      'Vendor video; performance claims are not independently tested.'),
    V('Inside a Google Cloud TPU Data Center', 'Google Cloud', 'FsxthdQ_sL4', '2:42',
      'See real TPU racks and liquid cooling: the energy and scaling challenges of 5.4.',
      'Whole video (3 min)',
      'Name two parts of the building that exist only to remove heat or supply power.',
      'Short marketing tour; no numbers for PUE or cost.'),
]
