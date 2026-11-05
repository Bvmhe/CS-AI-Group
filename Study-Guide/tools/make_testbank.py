"""Build the editable Word Test Bank (Stage 2, section F).

25 True/False, 25 multiple-choice (21 single-answer, 4 select-all) and 10 open questions:
5 T/F + 5 MC + 2 open per recipe, every Kitchen Lab used at least once (with its data given,
so no external platform is needed). Questions first, answer key after.
Answer positions of single-answer MC questions are spread over A-D by rotating the options."""
import os
from collections import Counter
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
TB = os.path.join(HERE, '..', '..', 'Test-Bank')
OUT = os.path.join(TB, 'FengweiFriedRice_Stage2_TestBank_V1.docx')
KEY_OUT = os.path.join(TB, 'FengweiFriedRice_Stage2_TestBank_AnswerKey.docx')
MD_OUT = os.path.join(TB, 'Questions.md')
NAVY = RGBColor(0x17, 0x32, 0x4D)
RED = RGBColor(0xEF, 0x3F, 0x4F)
LAB = 'Kitchen Lab'

# ---------------------------------------------------------------- True / False
# (tag, statement, answer, explanation; false statements are corrected)
TF = [
    # Recipe 01
    ('R1 · 1.1 · R', 'One MAC (multiply-accumulate) counts as two FLOPs.', True,
     'One multiply plus one add = 2 FLOPs.'),
    ('R1 · 1.2 · U', 'A TPU is more flexible than a CPU for general programs.', False,
     'Correct: the CPU is the most flexible chip; the TPU is the least flexible but the most efficient at matrix work.'),
    ('R1 · 1.3 · A', 'If 90% of a job can be parallelised, 100 cooks give a speed-up of 100×.', False,
     "Correct: Amdahl's law gives 1 ÷ (0.1 + 0.9/100) ≈ 9.2×; the limit is 10×, set by the serial 10%."),
    ('R1 · 1.4 · U', 'At batch size 1, a large layer on a typical accelerator is usually memory-bound.', True,
     'Arithmetic intensity is about 1 FLOP/byte, far below a ridge point of about 20 FLOP/byte.'),
    ('R1 · Lab 1 · A', f'In {LAB} 1 (Race the Crew), the GPU\'s speed-up over the CPU is LARGER for 256 × 256 matrices '
     'than for 4,096 × 4,096 matrices.', False,
     'Correct: the speed-up is much SMALLER at size 256 (only a few times), because most GPU cores sit idle and '
     'launch overhead dominates; at 4,096 the GPU is roughly 50-100× faster.'),
    # Recipe 02
    ('R2 · 2.1 · U', 'Encapsulation means each layer adds its own header to the data from the layer above.', True,
     'Like boxes inside boxes: application data → TCP/UDP header → IP header → link header.'),
    ('R2 · 2.2 · U', 'UDP guarantees that every packet arrives in order.', False,
     'Correct: UDP gives no delivery or ordering guarantee; TCP provides acknowledgements, retransmission and ordering.'),
    ('R2 · 2.4 · U', 'Bandwidth and latency are two names for the same quantity.', False,
     'Correct: bandwidth is how much data per second; latency is how long one message takes to arrive.'),
    ('R2 · 2.4 · A', 'In a ring all-reduce, the traffic per chip approaches 2 × S as the number of chips grows.', True,
     'Traffic = 2(N − 1)/N × S, which tends to 2S for large N.'),
    ('R2 · Lab 2 · A', f'In {LAB} 2, a server answers ping with an average round-trip time of 12 ms. Using the '
     'speed-of-light floor of 5 µs per km (one way), the server can be at most about 1,200 km away.', True,
     'One way = 6 ms = 6,000 µs; 6,000 ÷ 5 = 1,200 km. Real paths are slower, so the server is usually much closer.'),
    # Recipe 03
    ('R3 · 3.2 · A', 'With rewards [0, 0, 0, 10] and γ = 0.9, the discounted return is 7.29.', True,
     '0.9³ × 10 = 7.29.'),
    ('R3 · 3.2 · U', 'A discount factor γ close to 0 makes the agent far-sighted.', False,
     'Correct: γ close to 0 makes the agent short-sighted (only the next reward counts); γ close to 1 is far-sighted.'),
    ('R3 · 3.3 · U', 'Reward hacking means the agent earns a high reward without doing the intended task.', True,
     'Example: the boat-racing agent circling to hit respawning targets instead of finishing the race.'),
    ('R3 · 3.4 · U', 'A purely greedy agent can get stuck on a poor option forever.', True,
     'It never tries other options, so it never learns that a better one exists (0% best arm in the bandit simulation).'),
    ('R3 · Lab 3 · U', f'In {LAB} 3 (GridWorld), setting epsilon = 0 guarantees that the agent eventually finds the best route.', False,
     'Correct: with epsilon = 0 the agent never explores; it repeats the first route that worked, some squares are '
     'never tried, and it can get stuck on a poor route.'),
    # Recipe 04
    ('R4 · 4.1 · R', 'A single camera directly measures the distance to every pixel.', False,
     'Correct: a camera measures light and colour; depth must be inferred (stereo, motion, learning). LiDAR, radar and ultrasonic measure distance directly.'),
    ('R4 · 4.1 · A', 'A LiDAR pulse returning after 400 ns means the object is about 60 m away.', True,
     'd = c × t ÷ 2 = 3 × 10⁸ × 4 × 10⁻⁷ ÷ 2 = 60 m.'),
    ('R4 · 4.1 · U', 'Radar measures the speed of an object directly using the Doppler effect.', True,
     'The frequency shift of the echo gives the relative speed in one measurement.'),
    ('R4 · 4.4 · U', 'Two sensors that fail for the same reason (for example fog) give the same benefit as two independent sensors.', False,
     'Correct: correlated failures happen together; in the guide\'s example both-fail is 5% instead of 0.25% for independent sensors.'),
    ('R4 · Lab 4 · U', f'In {LAB} 4 (phyphox), a phone lying still on a table shows an acceleration (without g) of exactly 0 m/s².', False,
     'Correct: the reading jitters around 0 (about ±0.05 m/s²) and is never exactly zero; that jitter is sensor noise σ.'),
    # Recipe 05
    ('R5 · 5.1 · A', 'Fusing three element-wise operations on a 2 MB tensor cuts memory traffic from 12 MB to 4 MB.', True,
     'Unfused: 3 × (read 2 MB + write 2 MB) = 12 MB. Fused: one read + one write = 4 MB.'),
    ('R5 · 5.2 · U', 'In data parallelism, each chip stores only a slice of the model\'s weights.', False,
     'Correct: in data parallelism every chip holds a FULL copy of the model and processes a different slice of the batch; '
     'splitting the weights is model parallelism.'),
    ('R5 · 5.4 · U', 'A PUE of 1.0 means the facility uses no energy beyond the computing equipment.', True,
     'PUE = total facility energy ÷ IT energy; 1.0 means zero overhead for cooling and power conversion.'),
    ('R5 · 5.4 · A', 'If each of 4,096 chips has an MTBF of 43,800 hours, the whole system fails about every 10.7 hours.', True,
     '43,800 ÷ 4,096 ≈ 10.7 h, which is why large jobs need checkpoints.'),
    ('R5 · Lab 5 · U', f'In {LAB} 5, the FIRST call of a jax.jit function is much slower than later calls mainly because XLA compiles it.', True,
     'The first call traces and compiles the function (about 0.1-1 s); later calls reuse the compiled code and take milliseconds.'),
]

# ---------------------------------------------------------------- Multiple choice
# (tag, question, options, correct, reasoning)  correct = int (choose ONE) or list (select ALL)
MC = [
    # Recipe 01
    ('R1 · 1.1 · A', 'Weights [2, −1, 0.5], inputs [1, 4, 6]. What is the neuron output (no bias)?',
     ['−1', '1.0', '3', '5'], 1, '2 × 1 + (−1) × 4 + 0.5 × 6 = 2 − 4 + 3 = 1.0.'),
    ('R1 · 1.2 · U', 'Which chip best fits a voice assistant in a small battery-powered speaker?',
     ['GPU', 'Server CPU', 'NPU', 'TPU pod'], 2,
     'An NPU gives the most work per watt within a few watts; GPUs and TPU pods need far more power.'),
    ('R1 · 1.2 · A', 'In a 2 × 2 systolic array with W = [[1, 2], [3, 4]] and x = [5, 6], the output y = W·x is:',
     ['[11, 23]', '[17, 39]', '[12, 30]', '[23, 34]'], 1, '1·5 + 2·6 = 17; 3·5 + 4·6 = 39.'),
    ('R1 · 1.4 · A', 'A chip has 10 TFLOP/s peak and 500 GB/s memory bandwidth. Its ridge point is:',
     ['2 FLOP/byte', '20 FLOP/byte', '50 FLOP/byte', '5,000 FLOP/byte'], 1,
     '10 × 10¹² ÷ (500 × 10⁹) = 20 FLOP/byte; below 20 the chip is memory-bound.'),
    ('R1 · Lab 1 · A', f'{LAB} 1: ten 4,096 × 4,096 matrix multiplications (about 687 billion MACs in total) take 0.25 s on a T4 GPU. '
     'What throughput did the GPU reach?',
     ['0.07 TFLOP/s', '2.7 TFLOP/s', '5.5 TFLOP/s', '11 TFLOP/s'], 2,
     'TFLOP/s = 2 × MACs ÷ time = 2 × 6.87 × 10¹¹ ÷ 0.25 ≈ 5.5 × 10¹². 2.7 forgets the factor 2; 11 doubles it again.'),
    # Recipe 02
    ('R2 · 2.2 · U', 'Which of these does TCP provide that UDP does NOT?',
     ['Acknowledgement of delivered data', 'In-order delivery', 'Port numbers to reach the right program',
      'Retransmission of lost packets'], [0, 1, 3],
     'TCP adds acknowledgements, ordering and retransmission. Both TCP and UDP use port numbers, so C is wrong.'),
    ('R2 · 2.2 · U', 'Which protocol is built on UDP to speed up connection set-up for the web?',
     ['FTP', 'QUIC', 'ARP', 'SMTP'], 1, 'QUIC (RFC 9000) runs over UDP and combines transport and TLS handshakes.'),
    ('R2 · 2.3 · U', 'A factory camera must stop a machine within 15 ms. The best place to run the model is:',
     ['A distant cloud region', 'On or near the machine (edge)', "A user's laptop at home", 'Any server; latency does not matter'], 1,
     'Only the edge fits a 15 ms budget; a far cloud round trip alone can exceed it.'),
    ('R2 · 2.4 · U', 'Tail latency refers to:',
     ['The average delay', 'The slowest small fraction of requests', 'The delay of the last packet only', 'Upload speed'], 1,
     'For example the 99th-percentile delay; one slow server can delay a whole request.'),
    ('R2 · 2.5 · U', 'Multi-access edge computing (MEC) in 5G mainly helps AI services by:',
     ['Increasing screen resolution', 'Putting computing close to the base station', 'Replacing TCP', 'Removing the need for antennas'], 1,
     'Operator servers near the base station shorten the round trip.'),
    # Recipe 03
    ('R3 · 3.2 · A', 'A corridor of squares 1-2-3 leads to the goal, one step after square 3 (reward 1). With γ = 0.9, V(1) equals:',
     ['0.729', '0.81', '0.9', '1.0'], 1, 'V(3) = 1, V(2) = 0.9 × 1 = 0.9, V(1) = 0.9 × 0.9 = 0.81.'),
    ('R3 · 3.3 · A', 'Q = 0.2, r = 1, max Q(s′) = 0.5, α = 0.1, γ = 0.9. After one Q-learning update the new Q is:',
     ['0.245', '0.325', '0.45', '1.45'], 1, 'Target = 1 + 0.9 × 0.5 = 1.45; TD error = 1.25; new Q = 0.2 + 0.1 × 1.25 = 0.325.'),
    ('R3 · 3.3 · U', 'Which of these rewards are DENSE (given at almost every step)?',
     ['+speed every step', '+1 only on reaching the charging station', '−energy used every step', '+1 only for winning the game'], [0, 2],
     'A and C give feedback every step. B and D are sparse: they arrive only at the end.'),
    ('R3 · 3.4 · A', 'A bandit has arms that pay 0.2, 0.5 and 0.8 on average. The agent knows the best arm and uses ε = 0.1 '
     '(when exploring, it picks one of the three arms at random). Expected reward per pull:',
     ['0.50', '0.77', '0.80', '0.74'], 1,
     '0.9 × 0.8 + 0.1 × (0.2 + 0.5 + 0.8) ÷ 3 = 0.72 + 0.05 = 0.77. Careful: if exploration chose only the OTHER two arms the answer would be 0.9 × 0.8 + 0.1 × 0.35 = 0.755.'),
    ('R3 · 3.5 · U', 'Domain randomisation helps sim-to-real transfer by:',
     ['Making the simulator perfect', 'Training on many varied simulated conditions', 'Removing the reward', 'Using only real data'], 1,
     'If the policy works across many random variations, the real world looks like one more variation.'),
    # Recipe 04
    ('R4 · 4.1 · U', 'Which sensors measure distance DIRECTLY?',
     ['LiDAR', 'A single camera', 'Radar', 'Ultrasonic'], [0, 2, 3],
     'LiDAR (light time of flight), radar (radio echo) and ultrasonic (sound echo) measure range; a single camera must infer depth.'),
    ('R4 · 4.2 · A', 'At 100 km/h, how far does a car travel during 0.3 s of processing latency?',
     ['2.78 m', '8.33 m', '30 m', '64.3 m'], 1, '100 km/h = 27.78 m/s; 27.78 × 0.3 ≈ 8.33 m before the car even reacts.'),
    ('R4 · 4.3 · A', 'Radar says 50.0 m (σ = 0.5 m) and a camera says 52.0 m (σ = 2.0 m). The inverse-variance fused estimate is about:',
     ['50.12 m', '51.0 m', '51.5 m', '52.0 m'], 0,
     'Weights 1/σ²: 4 and 0.25. (4 × 50 + 0.25 × 52) ÷ 4.25 = 213 ÷ 4.25 ≈ 50.12 m: the precise radar dominates.'),
    ('R4 · 4.3 · A', 'A stereo camera has f = 1,000 px and baseline B = 0.3 m. A disparity of 6 px means a depth of:',
     ['42.9 m', '50 m', '60 m', '300 m'], 1, 'Z = f × B ÷ d = 1,000 × 0.3 ÷ 6 = 50 m.'),
    ('R4 · 4.4 · U', 'Under SAE J3016, a driver-assist that steers and brakes but needs constant human supervision is:',
     ['Level 0', 'Level 2', 'Level 4', 'Level 5'], 1, 'Level 2 = partial automation; the human remains the driver.'),
    # Recipe 05
    ('R5 · 5.2 · A', 'Training a 7B-parameter model with mixed-precision Adam needs about 16 bytes per parameter. Total training state:',
     ['14 GB', '56 GB', '112 GB', '224 GB'], 2, '7 × 10⁹ × 16 B = 112 GB: more than one chip\'s memory, so the state must be split.'),
    ('R5 · 5.3 · A', 'A 7 GB INT8 model is served on a chip with 1 TB/s memory bandwidth. The memory-bound limit for one user is about:',
     ['14 tokens/s', '143 tokens/s', '1,000 tokens/s', '4,571 tokens/s'], 1,
     'Every token reads all 7 GB: 7 GB ÷ 1 TB/s = 7 ms per token → about 143 tokens/s.'),
    ('R5 · 5.5 · A', 'The diameter (longest shortest path) of a 16 × 16 × 16 torus is:',
     ['16 hops', '24 hops', '45 hops', '48 hops'], 1, 'Torus: 3 × (16 ÷ 2) = 24 hops. 45 is the mesh without wraparound.'),
    ('R5 · 5.4 · U', 'Which changes usually RAISE the MFU (model FLOPs utilisation) of a training job?',
     ['Fusing operations to cut memory traffic', 'Overlapping communication with computation',
      'Making every step wait for a slow data loader', 'Using a larger batch per chip'], [0, 1, 3],
     'A, B and D keep the compute units busier. C leaves the chips idle, so MFU falls.'),
    ('R5 · Lab 5 · A', f'{LAB} 5: you change the matrix size in the JAX program from 4,096 to 8,192. By what factor do the FLOPs of a @ a grow?',
     ['2×', '4×', '8×', '16×'], 2,
     'Matrix multiply = 2n³ FLOPs; doubling n gives 2³ = 8×. For large sizes the measured time grows close to 8× too.'),
]

# ---------------------------------------------------------------- Open questions
# (tag, question, model answer / key points)
OPEN = [
    ('R1 · 1.4 · A', 'A layer has 512 inputs and 2,048 outputs, with FP16 weights, on a chip with 10 TFLOP/s and 500 GB/s. '
     'Compute the arithmetic intensity at batch 1 and at batch 32, decide whether each is memory-bound or compute-bound, '
     'estimate the achieved TFLOP/s, and explain in your own words why batching helps.',
     'Weights 512 × 2,048 = 1,048,576 → 2.1 MB; FLOPs per input 2,097,152. Batch 1 ≈ 1.0 FLOP/byte < 20 → memory-bound, '
     '≈ 0.5 TFLOP/s (5%). Batch 32 ≈ 29.7 FLOP/byte > 20 → compute-bound, ≈ 10 TFLOP/s. Batching reuses each fetched weight '
     'for many inputs, raising FLOPs per byte. Marks: intensities (2), classification and TFLOP/s (1), explanation (2). '
     'Accept small differences from rounding or from ignoring activation bytes.'),
    ('R1 · 1.5 · E', 'Evaluate running Face ID on the phone\'s Neural Engine instead of in the cloud. Give two advantages and two '
     'limitations of on-device AI, and decide whether the same choice would suit a large chatbot model. Justify your decision.',
     'Advantages: privacy (face data stays in the Secure Enclave), low latency, works offline, energy-efficient NPU. '
     'Limitations: limited compute and memory, unsupported operations fall back to the CPU, models must be small or quantised. '
     'Large chatbot: usually cloud or hybrid, because its weights exceed phone memory (or a small on-device model + cloud). '
     'Marks: 1 per valid point (4), justified decision (1). Accept other valid points, e.g. updates or battery life.'),
    ('R2 · 2.3 · E', 'A hospital wants an AI that reads X-rays in under 2 seconds and keeps patient data private, but the model needs '
     'a large GPU. Recommend cloud, edge (on-premises) or hybrid, and justify the design with latency, bandwidth, privacy and cost.',
     'Strong answers choose an on-premises edge GPU server or hybrid: data stays inside the hospital (privacy and law), LAN latency '
     'is low, X-ray images are large (bandwidth), the cloud can still be used for training or overflow. Cost: buying a server vs '
     'renting. Marks: recommendation (1), four justified factors (4). Accept a well-argued cloud answer that addresses '
     'privacy (encryption, anonymisation, contracts) and the 2-second budget.'),
    ('R2 · 2.4 · A', '14 GB of gradients are all-reduced across 256 chips with 100 GB/s per chip, and one compute step takes 8.2 s. '
     'Calculate the traffic per chip, the communication time and the scaling efficiency, then explain one way to improve it '
     'and why it works.',
     'Traffic = 2 × 255/256 × 14 ≈ 27.9 GB; time ≈ 0.28 s; efficiency = 8.2 ÷ (8.2 + 0.28) ≈ 96.7%. Improvements: overlap '
     'communication with compute, larger batch per chip (more compute per byte), gradient compression, hierarchical collectives. '
     'Marks: 3 calculations, 2 for one improvement with a correct reason.'),
    ('R3 · 3.3 · C', 'Design a reward for a home cleaning robot. Show how a naive reward could be hacked, propose a better reward, '
     'say whether it is sparse or dense, and compare the two rewards: which behaviour would each one produce?',
     'Naive: +1 per item put in the bin → the robot tips the bin and re-collects. Better: reward the reduction of rubbish on '
     'the floor (or clean area), penalties for knocking things over and for energy. Dense if given every step, sparse if only '
     'when the room is clean. Comparison: the naive one rewards activity, the better one rewards the real goal. '
     'Marks: hack (2), improved reward (2), sparse/dense with comparison (1). Accept other plausible hacks.'),
    ('R3 · Lab 3 · An', f'A student runs {LAB} 3 (GridWorld TD learning) with epsilon = 0. After 2,000 steps the agent still walks the '
     'same long route; the arrows on squares it never visited still show their starting values. Explain why this plan failed, '
     'what the student should change, and what they should expect with epsilon = 0.5. Connect the result to the greedy agent in '
     'the bandit simulation and compare ε-greedy with UCB as a fix.',
     'Failure: epsilon = 0 means no exploration; the agent exploits the first route that reached the goal, so unvisited squares '
     'are never updated and a shorter route is never found. Fix: epsilon about 0.1-0.2 (or decaying epsilon). Epsilon = 0.5: '
     'values are learned everywhere but the path looks messy and hits the −1 squares more often. Bandit link: the greedy '
     'agent found the best arm 0% of the time. ε-greedy explores at random at a fixed rate; UCB adds a bonus √(2 ln t / n) to '
     'little-tried actions and explores where it is uncertain. Marks: diagnosis (2), fix (1), ε = 0.5 prediction (1), '
     'comparison (1). Accept other reasonable fixes such as optimistic initial values.'),
    ('R4 · 4.4 · A', 'A car at 50 km/h has 0.15 s sensing-and-processing latency and brakes at 7 m/s². Compute the total stopping '
     'distance and the impact speed if a ball is detected 15 m ahead. What does this show about sensor range and latency '
     'when you design a safety system?',
     '13.89 m/s; blind distance 2.08 m; braking 13.89² ÷ 14 ≈ 13.78 m; total ≈ 15.9 m. Detected at 15 m: 12.92 m left to brake, '
     'v = √(13.89² − 2 × 7 × 12.92) ≈ 3.5 m/s at impact. Lesson: early, reliable detection (range) and low latency decide '
     'whether braking works; both must be in the design budget. Marks: 2 + 2 + 1.'),
    ('R4 · Lab 4 · An', f'In the offline version of {LAB} 4, two students stand 20 m apart. One claps; the other starts a stopwatch on '
     'SEEING the clap and stops it on HEARING it. Five trials give t = 0.21, 0.08, 0.15, 0.30 and 0.11 s. Compute the speed of '
     'sound from each trial and from the mean time, compare with 343 m/s, and explain why single results scatter so much. '
     'Why does averaging help, what can averaging NOT fix, and how would you improve the experiment?',
     'Speeds: 95, 250, 133, 67, 182 m/s. Mean t = 0.17 s → v ≈ 118 m/s (average the times, not the speeds). True delay is only '
     '20 ÷ 343 ≈ 0.058 s, shorter than human reaction time (about 0.2 s), so reaction errors are as big as the signal. '
     'Averaging n trials shrinks random noise by about √n (sensor fusion idea, 4.3) but cannot remove a systematic bias, '
     'e.g. reacting faster to sound than to sight. Improvements: a longer distance (100 m → 0.29 s), many more trials, or '
     'phyphox Acoustic Stopwatch on two phones (no human reaction). Marks: calculations (2), noise explanation (1), '
     'averaging vs bias (1), improvement (1).'),
    ('R5 · 5.4 · A', 'Estimate the training time, cost and energy for a 7B model on 1.4 trillion tokens with 256 chips '
     '(200 TFLOP/s each, 40% MFU, $2 per chip-hour, 400 W per chip, PUE 1.1). Use FLOPs ≈ 6 × parameters × tokens. '
     'Which single change would cut all three numbers, and why?',
     '6 × 7 × 10⁹ × 1.4 × 10¹² = 5.88 × 10²² FLOPs; 256 × 200 × 10¹² × 0.4 = 2.05 × 10¹⁶ FLOP/s → ≈ 2.87 × 10⁶ s ≈ 33.2 days; '
     '≈ 204,167 chip-hours ≈ $408,333; 204,167 h × 0.4 kW × 1.1 ≈ 89.8 MWh. Raising MFU (fusion, better batching, overlap) cuts '
     'time, cost and energy in proportion. Marks: 4 values, 1 for the change with reason. Accept "more chips" only if the '
     'answer notes that cost and energy stay about the same.'),
    ('R5 · Lab 5 · P', f'{LAB} 5 runs this JAX program on a Colab T4 GPU (FP32 peak about 8.1 TFLOP/s):\n'
     '    f = jax.jit(lambda a: (a @ a).sum())\n'
     'Measured times: size 1,024: first call 0.62 s, second call 0.0021 s; size 4,096: 0.034 s; size 8,192: 0.27 s. '
     'Explain the slow first call, compute the TFLOP/s and MFU for each size, and explain why the time ratio from 1,024 to '
     '4,096 is far from the FLOP ratio while 4,096 → 8,192 is close to 8×. Predict the time for size 16,384.',
     'First call: tracing + XLA compilation. FLOPs = 2n³: 1,024 → 2.15 GFLOP ÷ 0.0021 s ≈ 1.0 TFLOP/s (≈ 13% MFU); '
     '4,096 → 137 GFLOP ÷ 0.034 ≈ 4.0 TFLOP/s (≈ 50%); 8,192 → 1.10 TFLOP ÷ 0.27 ≈ 4.1 TFLOP/s (≈ 50%). 1,024 → 4,096 is 64× the '
     'FLOPs but only ≈ 16× the time, because at small sizes fixed overhead (launch, transfer) dominates and the GPU is not full. '
     '4,096 → 8,192: 0.27 ÷ 0.034 ≈ 7.9 ≈ 8×, both sizes are compute-bound. Prediction for 16,384: about 8 × 0.27 ≈ 2.2 s, if '
     'memory allows. Marks: compilation (1), TFLOP/s and MFU (2), ratio explanation (1), prediction (1). Accept answers '
     'that also mention timing noise or GPU clock changes.'),
]


def rotate_answers(mc):
    """Spread single-answer keys over A-D: rotate each option list so the answer lands on a target letter."""
    out, k = [], 0
    targets = [0, 2, 3, 1, 3, 0, 1, 2]
    for tag, text, opts, ans, why in mc:
        if isinstance(ans, int):
            t = targets[k % len(targets)]
            k += 1
            shift = (ans - t) % len(opts)
            opts = opts[shift:] + opts[:shift]
            ans = t
        out.append((tag, text, opts, ans, why))
    return out


MC = rotate_answers(MC)


def heading(doc, text, size=16, color=NAVY):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = color
    return p


def q(doc, n, tag, text, note=''):
    p = doc.add_paragraph()
    r = p.add_run(f'{n}. ')
    r.bold = True
    if note:
        b = p.add_run(note + ' ')
        b.bold = True
    p.add_run(text)
    t = p.add_run(f'   [{tag}]')
    t.italic = True
    t.font.size = Pt(9)
    t.font.color.rgb = RED
    return p


def letters(ans):
    return ', '.join('ABCD'[i] for i in ans) if isinstance(ans, list) else 'ABCD'[ans]


def coverage_table(doc):
    rows = []
    for r in range(1, 6):
        key = f'R{r} '
        rows.append((r, sum(t[0].startswith(key) for t in TF), sum(t[0].startswith(key) for t in MC),
                     sum(t[0].startswith(key) for t in OPEN)))
    names = {1: 'Recipe 01 · AI Hardware', 2: 'Recipe 02 · Networks & AI Infrastructure', 3: 'Recipe 03 · Reinforcement Learning',
             4: 'Recipe 04 · AI Sensors', 5: 'Recipe 05 · System Integration'}
    tb = doc.add_table(rows=1, cols=4)
    tb.style = 'Light Grid Accent 1'
    for c, h in enumerate(['Topic (guide recipe)', 'True/False', 'Multiple choice', 'Open']):
        tb.rows[0].cells[c].text = h
    for r, a, b, c in rows:
        cells = tb.add_row().cells
        cells[0].text, cells[1].text, cells[2].text, cells[3].text = names[r], str(a), str(b), str(c)
    cells = tb.add_row().cells
    cells[0].text, cells[1].text, cells[2].text, cells[3].text = 'Total', str(len(TF)), str(len(MC)), str(len(OPEN))


TITLE = 'Fengwei Fried Rice · Stage 2 Test Bank'
SUB = 'Team Fengwei Fried Rice · Stage 2 Study Guide · Hardware, Networks & Behaviourist AI · November 2026'
MULTI = sum(isinstance(m[3], list) for m in MC)
TOTAL = len(TF) + len(MC) + len(OPEN)
MC_NOTE = ('Each question says how many answers to choose: "(Choose ONE)" means exactly one correct option; '
           '"(Select ALL that apply)" means one or more options are correct and you must choose all of them.')
OPEN_NOTE = ('Write a developed answer of more than 100 words, or show full working (calculations, a diagram or '
             'worked steps) with a short explanation of what the result means.')
KEY_NOTE = ('Each open question is worth 5 marks. The key points below are a guide, not the only correct answer: '
            'give credit to any valid alternative that is correct and well reasoned.')


def mc_note(ans):
    return '(Select ALL that apply)' if isinstance(ans, list) else '(Choose ONE)'


def new_doc(title, sub):
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Arial'
    st.font.size = Pt(11)
    doc.core_properties.title = title
    doc.core_properties.author = 'Team Fengwei Fried Rice'
    t = heading(doc, title, 22)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s = doc.add_paragraph(sub)
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return doc


def write_key(doc, head=True):
    if head:
        heading(doc, 'Answer Key', 20, RED)
    n = 1
    heading(doc, 'Part A · True or False (false statements are corrected)', 13)
    for tag, _, ans, why in TF:
        doc.add_paragraph(f'{n}. {"True" if ans else "False"}. {why}  [{tag}]')
        n += 1
    heading(doc, 'Part B · Multiple choice', 13)
    for tag, _, opts, ans, why in MC:
        sel = ans if isinstance(ans, list) else [ans]
        doc.add_paragraph(f'{n}. {letters(ans)}) {" / ".join(opts[i] for i in sel)}. {why}  [{tag}]')
        n += 1
    heading(doc, 'Part C · Open questions (sample answers, key points and marking guide)', 13)
    doc.add_paragraph(KEY_NOTE)
    for tag, _, ans in OPEN:
        doc.add_paragraph(f'{n}. {ans}  [{tag}]')
        n += 1


def write_markdown(path):
    """All questions written out one by one, readable directly on GitHub (no answers)."""
    L = [f'# {TITLE}', '', f'*{SUB}*', '',
         f'{len(TF)} True/False · {len(MC)} multiple-choice ({len(MC) - MULTI} choose-one, {MULTI} select-all) · '
         f'{len(OPEN)} open · **{TOTAL} questions**. Each question is tagged [Recipe · guide section · level]. '
         'The answers are in the separate Answer Key PDF and at the end of the Word file.', '',
         '## Part A · True or False', '', 'Write T (true) or F (false) for each statement.', '']
    n = 1
    for tag, text, _, _ in TF:
        L += [f'**{n}.** {text} `[{tag}]`', '']
        n += 1
    L += ['## Part B · Multiple choice', '', MC_NOTE, '']
    for tag, text, opts, ans, _ in MC:
        L += [f'**{n}. {mc_note(ans)}** {text} `[{tag}]`', '']
        L += [f'- {"ABCD"[k]}) {o}' for k, o in enumerate(opts)] + ['']
        n += 1
    L += ['## Part C · Open questions', '', OPEN_NOTE, '']
    for tag, text, _ in OPEN:
        body = '\n\n'.join(text.split('\n'))
        L += [f'**{n}.** {body} `[{tag}]`', '']
        n += 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    doc = new_doc(TITLE, SUB)
    multi = MULTI
    doc.add_paragraph(
        f'Contents: {len(TF)} True/False, {len(MC)} multiple-choice ({len(MC) - multi} choose-one, {multi} select-all) and '
        f'{len(OPEN)} open questions, {len(TF) + len(MC) + len(OPEN)} in total, numbered 1-{len(TF) + len(MC) + len(OPEN)}. '
        'Each question is tagged [Recipe · guide section · level]. Levels: R = remember, U = understand, A = apply, '
        'An = analyse, E = evaluate, C = create, P = predict. Questions tagged "Lab" use a Kitchen Lab from the guide; '
        'all the data you need is in the question, so no website or app is required. '
        'The answer key starts on a new page after all questions.')
    heading(doc, 'Coverage', 13)
    coverage_table(doc)

    heading(doc, 'Part A · True or False')
    doc.add_paragraph('Write T (true) or F (false) for each statement.')
    n = 1
    for tag, text, _, _ in TF:
        q(doc, n, tag, text)
        n += 1
    heading(doc, 'Part B · Multiple choice')
    doc.add_paragraph(MC_NOTE)
    for tag, text, opts, ans, _ in MC:
        q(doc, n, tag, text, mc_note(ans))
        for k, o in enumerate(opts):
            doc.add_paragraph(f'{"ABCD"[k]}) {o}', style='List Bullet')
        n += 1
    heading(doc, 'Part C · Open questions')
    doc.add_paragraph('Write a developed answer of more than 100 words, or show full working (calculations, a diagram or '
                      'worked steps) with a short explanation of what the result means.')
    for tag, text, _ in OPEN:
        q(doc, n, tag, text)
        n += 1

    doc.add_page_break()
    write_key(doc)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    key = new_doc('Fengwei Fried Rice · Stage 2 Test Bank: Answer Key', SUB)
    write_key(key, head=False)
    key.save(KEY_OUT)
    write_markdown(MD_OUT)
    print(os.path.abspath(OUT), TOTAL, 'questions')
    print(os.path.abspath(KEY_OUT))
    print(os.path.abspath(MD_OUT))
    print('TF true/false:', Counter(t[2] for t in TF))
    print('MC keys:', Counter(letters(m[3]) for m in MC if isinstance(m[3], int)), 'multi:', multi)


if __name__ == '__main__':
    main()
