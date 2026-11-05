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


def write_markdown(path):
    """All questions written out one by one, readable directly on GitHub (no answers)."""
    L = ['<p align="center"><img src="../Study-Guide/assets/sprites/mama_cheer.png" height="110"> '
         '<img src="../Study-Guide/assets/sprites/fried_rice.png" height="70"> '
         '<img src="../Study-Guide/assets/sprites/boy_grin.png" height="100"></p>', '',
         f'# 🍳 {TITLE}', '', f'*{SUB}*', '',
         '> 👩‍🍳 **Mama says:** Welcome back to the kitchen, apprentice! Time to taste-test what you learned.', '',
         f'{len(TF)} True/False · {len(MC)} multiple-choice ({len(MC) - MULTI} choose-one, {MULTI} select-all) · '
         f'{len(OPEN)} open · **{TOTAL} questions**. Each question is tagged [Recipe · guide section · level]. '
         'The answers are in the separate Answer Key PDF and at the end of the Word file.', '',
         '## 🥚 Part A · True or False', '', 'Write T (true) or F (false) for each statement.', '']
    n = 1
    for tag, text, _, _ in TF:
        L += [f'**{n}.** {text} `[{tag}]`', '']
        n += 1
    L += ['---', '', '## 🍙 Part B · Multiple choice', '', MC_NOTE, '']
    for tag, text, opts, ans, _ in MC:
        L += [f'**{n}. {mc_note(ans)}** {text} `[{tag}]`', '']
        L += [f'- {"ABCD"[k]}) {o}' for k, o in enumerate(opts)] + ['']
        n += 1
    L += ['---', '', '## 🍲 Part C · Open questions', '', OPEN_NOTE, '']
    for tag, text, _ in OPEN:
        body = '\n\n'.join(text.split('\n'))
        L += [f'**{n}.** {body} `[{tag}]`', '']
        n += 1
    L += ['---', '', f'<sub>{CREDIT}</sub>', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ======================================================================= Cooking Mama theme (same look as the guide)
import tempfile
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(HERE)
FONTS = os.path.join(ROOT, 'assets', 'fonts')
SPR = os.path.join(ROOT, 'assets', 'sprites')
P_NAVY, P_RED, P_CREAM = (23, 50, 77), (239, 63, 79), (255, 244, 220)
P_YEL, P_MINT, P_PINK = (255, 201, 74), (143, 213, 192), (255, 214, 222)
REC = {1: ('THE KITCHEN CREW: AI HARDWARE', (231, 76, 60), 'fried_rice.png'),
       2: ('ORDER UP! NETWORKS & AI INFRA', (52, 152, 219), 'bento.png'),
       3: ('TRAINING THE KITCHEN PUP: RL', (46, 174, 96), 'pup.png'),
       4: ("CHEF'S SENSES: AI SENSORS", (155, 89, 182), 'peppers.png'),
       5: ('RUNNING THE CHAIN: INTEGRATION', (230, 126, 34), 'nabe.png')}
HEAD_FONT = 'Arial Rounded MT Bold'
CREDIT = 'Cooking Mama characters and pictures © Office Create. Used for class only, not for sale.'
IMG = tempfile.mkdtemp(prefix='tb_img_')


def hexc(c):
    return '%02X%02X%02X' % c


def pix(n):
    return ImageFont.truetype(os.path.join(FONTS, 'PressStart2P.ttf'), n)


def fred(n, var='SemiBold'):
    f = ImageFont.truetype(os.path.join(FONTS, 'Fredoka.ttf'), n)
    f.set_variation_by_name(var)
    return f


def sprite(name, h):
    im = Image.open(os.path.join(SPR, name)).convert('RGBA')
    return im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS)


def paste(base, name, x, y, h):
    s = sprite(name, h)
    base.alpha_composite(s, (int(x), int(y)))
    return s.width


def gingham(d, w, y0, y1, s=50):
    for y in range(y0, y1, s):
        for x in range(0, w, s):
            i, j = (x // s) % 2, (y // s) % 2
            d.rectangle([x, y, x + s, y + s], fill=(255, 180, 196) if i and j else (P_PINK if i or j else (255, 255, 255)))


def pill(d, box, text, f, fill=P_NAVY, fg=(255, 255, 255), r=None, shadow=True):
    x0, y0, x1, y1 = box
    r = r or (y1 - y0) // 2
    if shadow:
        d.rounded_rectangle([x0 + 8, y0 + 8, x1 + 8, y1 + 8], r, fill=(0, 0, 0, 60))
    d.rounded_rectangle(box, r, fill=fill, outline=P_NAVY, width=6)
    d.text(((x0 + x1) // 2, (y0 + y1) // 2), text, font=f, fill=fg, anchor='mm')


def save(im, name):
    p = os.path.join(IMG, name)
    bg = Image.new('RGB', im.size, P_CREAM)
    bg.paste(im, mask=im.split()[3])
    bg.save(p)
    return p


def img_cover():
    W, H = 1600, 1250
    im = Image.new('RGBA', (W, H), P_CREAM + (255,))
    d = ImageDraw.Draw(im)
    gingham(d, W, 0, 280)
    for x in range(-40, W + 80, 80):
        d.ellipse([x, 250, x + 80, 330], fill=P_CREAM)
    d.rounded_rectangle([60, 60, 940, 200], 26, fill=P_NAVY)
    d.rounded_rectangle([76, 76, 186, 184], 18, fill=P_YEL)
    d.text((131, 131), '?', font=pix(52), fill=P_NAVY, anchor='mm')
    d.polygon([(206, 106), (206, 156), (236, 131)], fill=P_RED)
    d.text((262, 131), 'FENGWEI FRIED RICE', font=pix(36), fill='white', anchor='lm')
    d.rounded_rectangle([160, 380, W - 152, 700], 40, fill=(0, 0, 0, 60))
    d.rounded_rectangle([150, 370, W - 160, 690], 40, fill='white', outline=P_NAVY, width=8)
    d.text((W // 2, 470), 'STAGE 2', font=pix(60), fill=P_RED, anchor='mm')
    d.text((W // 2, 590), 'TEST BANK', font=pix(92), fill=P_NAVY, anchor='mm')
    d.text((W // 2, 770), 'Hardware, Networks & Behaviourist AI', font=fred(54), fill=P_NAVY, anchor='mm')
    d.text((W // 2, 840), "Let's cook up some answers!  ·  November 2026", font=fred(42, 'Medium'), fill=P_RED, anchor='mm')
    d.rectangle([0, 1140, W, H], fill=(205, 140, 90))
    d.rectangle([0, 1140, W, 1156], fill=(170, 105, 60))
    items = (('mama_cheer.png', 300), ('boy_grin.png', 280), ('fried_rice.png', 150), ('girl_pigtails.png', 270),
             ('pup.png', 200), ('mama_book.png', 290))
    ws = [sprite(n, h).width for n, h in items]
    gap = (W - sum(ws)) // (len(items) + 1)
    x = gap
    for (n, h), w in zip(items, ws):
        paste(im, n, x, 1150 - h, h)
        x += w + gap
    return save(im, 'cover.png')


def img_banner(key, text, color, spr):
    W, H = 1600, 230
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([18, 48, W - 222, 198], 30, fill=(0, 0, 0, 60))
    d.rounded_rectangle([10, 40, W - 230, 190], 30, fill=P_NAVY)
    d.rounded_rectangle([28, 58, 140, 172], 20, fill=color)
    d.text((84, 115), key, font=pix(46), fill=P_NAVY if color in (P_YEL, P_MINT) else (255, 255, 255), anchor='mm')
    d.polygon([(160, 90), (160, 140), (190, 115)], fill=P_RED)
    d.text((214, 115), text, font=pix(34), fill='white', anchor='lm')
    s = sprite(spr, 220)
    im.alpha_composite(s, (W - 30 - s.width, 5))
    return save(im, f'banner_{key}.png')


def img_header():
    W, H = 1600, 120
    im = Image.new('RGBA', (W, H), P_CREAM + (255,))
    d = ImageDraw.Draw(im)
    gingham(d, W, 0, 70, 35)
    d.rounded_rectangle([20, 14, 760, 98], 20, fill=P_NAVY)
    d.rounded_rectangle([30, 24, 94, 88], 12, fill=P_YEL)
    d.text((62, 56), '?', font=pix(30), fill=P_NAVY, anchor='mm')
    d.text((112, 56), 'FENGWEI FRIED RICE', font=pix(22), fill='white', anchor='lm')
    pill(d, (1040, 22, 1580, 92), 'TEST BANK', pix(24), fill='white', fg=P_NAVY, shadow=False)
    return save(im, 'header.png')


def img_counter():
    W, H = 1600, 150
    im = Image.new('RGBA', (W, H), P_CREAM + (255,))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 110, W, H], fill=(205, 140, 90))
    d.rectangle([0, 110, W, 120], fill=(170, 105, 60))
    x = 60
    for n in ('tomato.png', 'carrot.png', 'onion.png', 'mama_icon_egg.png', 'cabbage.png', 'apple.png', 'eggplant.png',
              'mushroom.png' if os.path.exists(os.path.join(SPR, 'mushroom.png')) else 'peach.png', 'cucumber.png', 'pumpkin.png'):
        x += paste(im, n, x, 112 - 90, 90) + 70
    return save(im, 'counter.png')


# ---------------------------------------------------------------- Word helpers
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear')
    sh.set(qn('w:color'), 'auto')
    sh.set(qn('w:fill'), hexc(color))
    tcPr.append(sh)


def borders(table, color=P_NAVY, size=18):
    tblPr = table._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{e}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), str(size))
        el.set(qn('w:color'), hexc(color))
        b.append(el)
    tblPr.append(b)


def keep_together(table):
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit')
        trPr.append(cs)
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.keep_with_next = True


def run(p, text, bold=False, size=None, color=P_NAVY, font=None, italic=False):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(*color)
    if font:
        r.font.name = font
        r._element.rPr.rFonts.set(qn('w:eastAsia'), font)
    return r


def spacer(doc, pt=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run('')
    r.font.size = Pt(pt)
    return p


def window(doc, title, title_color, body_lines, title_fg=(255, 255, 255), icon=None):
    """A guide-style window: coloured title bar + white body, navy border. body_lines = list of (text, bold) or callables."""
    t = doc.add_table(rows=2, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t)
    top, body = t.rows[0].cells[0], t.rows[1].cells[0]
    shade(top, title_color)
    shade(body, (255, 255, 255))
    p = top.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    if icon:
        p.add_run().add_picture(os.path.join(SPR, icon), height=Inches(0.22))
        p.add_run('  ')
    run(p, title, bold=True, size=10.5, color=title_fg, font=HEAD_FONT)
    first = True
    for item in body_lines:
        bp = body.paragraphs[0] if first else body.add_paragraph()
        first = False
        bp.paragraph_format.space_after = Pt(2)
        item(bp) if callable(item) else run(bp, item[0], bold=item[1], size=10.5)
    keep_together(t)
    spacer(doc, 6)
    return t


def banner(doc, path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Inches(7.0))


def recipe_head(doc, r):
    name, col, icon = REC[r]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(os.path.join(SPR, icon), height=Inches(0.36))
    run(p, f'  RECIPE 0{r} · {name}', bold=True, size=13, color=col, font=HEAD_FONT)


def mama_says(doc, text, spr='mama_ok.png'):
    """Speech bubble next to Mama, like the guide's intro pages."""
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    a, b = t.rows[0].cells
    a.width, b.width = Inches(0.9), Inches(6.1)
    a.paragraphs[0].add_run().add_picture(os.path.join(SPR, spr), height=Inches(1.0))
    shade(b, (255, 255, 255))
    tcPr = b._tc.get_or_add_tcPr()
    tb = OxmlElement('w:tcBorders')
    for e in ('top', 'left', 'bottom', 'right'):
        el = OxmlElement(f'w:{e}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '18')
        el.set(qn('w:color'), hexc(P_NAVY))
        tb.append(el)
    tcPr.append(tb)
    first = True
    for para in text.split('\n'):
        p = b.paragraphs[0] if first else b.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(3)
        run(p, para, size=10.5)
    spacer(doc, 6)


def page_setup(doc, header_img, counter_img):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(0.6)
    sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(1.05)
    sec.header_distance, sec.footer_distance = Inches(0.25), Inches(0.2)
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.add_run().add_picture(header_img, width=Inches(7.07))
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.add_run().add_picture(counter_img, width=Inches(7.07))
    fp2 = sec.footer.add_paragraph()
    fp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(fp2, 'Fengwei Fried Rice · Stage 2 Test Bank · page ', size=8.5, color=P_NAVY, font=HEAD_FONT)
    r = fp2.add_run()
    for kind, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
        if kind:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), kind)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = txt
        r._r.append(el)
    r.font.size = Pt(8.5)
    # cream page colour like the guide
    bg = OxmlElement('w:background')
    bg.set(qn('w:color'), hexc(P_CREAM))
    doc.element.insert(0, bg)
    st = doc.settings.element
    st.insert(0, OxmlElement('w:displayBackgroundShape'))


def new_doc():
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Arial'
    st.font.size = Pt(10.5)
    st.font.color.rgb = RGBColor(*P_NAVY)
    doc.core_properties.title = TITLE
    doc.core_properties.author = 'Team Fengwei Fried Rice'
    page_setup(doc, IMGS['header'], IMGS['counter'])
    return doc


def cover(doc, key_only=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(IMGS['cover'], width=Inches(6.0))
    if key_only:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run(p, "MAMA'S ANSWER KEY", bold=True, size=24, color=P_RED, font=HEAD_FONT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(p, 'Team Fengwei Fried Rice · Stage 2 Study Guide companion', bold=True, size=12, font=HEAD_FONT)
    p.paragraph_format.space_after = Pt(0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run(p, CREDIT, size=8, color=(120, 120, 120), italic=True)
    mama_says(doc, (
        "Welcome back to the kitchen, apprentice! Time to taste-test what you learned. "
        f"This Test Bank has {TOTAL} questions from our five recipes: {len(TF)} True/False, {len(MC)} multiple-choice "
        f"({len(MC) - MULTI} choose-one, {MULTI} select-all) and {len(OPEN)} open questions, numbered 1-{TOTAL}.\n"
        'Each question is tagged [Recipe · guide section · level]. Levels: R = remember, U = understand, A = apply, '
        'An = analyse, E = evaluate, C = create, P = predict. Questions marked "Kitchen Lab" use a lab from the guide; '
        'all the data you need is in the question, so no website or app is needed.\n'
        + ('This file has only the answers. The questions are in the Word Test Bank and on GitHub (Test-Bank/Questions.md).'
           if key_only else "The answer key (Mama's Feedback) starts on a new page after all the questions. No peeking!")),
        'mama_cheer.png')
    if not key_only:
        coverage(doc)
    doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)


def coverage(doc):
    t = doc.add_table(rows=1, cols=4)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders(t, size=12)
    for c, h in enumerate(['Recipe (guide topic)', 'True/False', 'Multiple choice', 'Open']):
        cell = t.rows[0].cells[c]
        shade(cell, P_NAVY)
        run(cell.paragraphs[0], h, bold=True, size=10, color=(255, 255, 255), font=HEAD_FONT)
    for r in range(1, 6):
        key = f'R{r} '
        nice = {1: 'The Kitchen Crew: AI Hardware', 2: 'Order Up! Networks & AI Infrastructure',
                3: 'Training the Kitchen Pup: Reinforcement Learning', 4: "Chef's Senses: AI Sensors",
                5: 'Running the Chain: System Integration'}
        vals = [f'Recipe 0{r} · {nice[r]}', sum(x[0].startswith(key) for x in TF),
                sum(x[0].startswith(key) for x in MC), sum(x[0].startswith(key) for x in OPEN)]
        cells = t.add_row().cells
        for c, v in enumerate(vals):
            shade(cells[c], (255, 255, 255))
            run(cells[c].paragraphs[0], str(v), bold=(c == 0), size=10, color=REC[r][1] if c == 0 else P_NAVY)
    cells = t.add_row().cells
    for c, v in enumerate(['Total', len(TF), len(MC), len(OPEN)]):
        shade(cells[c], P_YEL)
        run(cells[c].paragraphs[0], str(v), bold=True, size=10)
    t.autofit = False
    for row in t.rows:
        for c, w in enumerate((3.6, 1.15, 1.35, 0.9)):
            row.cells[c].width = Inches(w)
            for p in row.cells[c].paragraphs:
                p.paragraph_format.space_after = Pt(1)
                if c:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    spacer(doc, 6)


def rnum(tag):
    return int(tag[1])


def tag_title(n, tag, extra=''):
    lab = '  ·  KITCHEN LAB' if 'Lab' in tag else ''
    return f'Q{n}  ·  {tag}{lab}{extra}'


def questions(doc):
    n = 1
    banner(doc, IMGS['A'])
    mama_says(doc, 'Part A · True or False. Read each statement and write T (true) or F (false).', 'mama_happy.png')
    cur = None
    for tag, text, _, _ in TF:
        if rnum(tag) != cur:
            cur = rnum(tag)
            recipe_head(doc, cur)
        window(doc, tag_title(n, tag), REC[cur][1], [(text, False), ('Answer:  T  /  F', True)])
        n += 1
    doc.add_page_break()
    banner(doc, IMGS['B'])
    mama_says(doc, 'Part B · Multiple choice. ' + MC_NOTE, 'mama_ok.png')
    cur = None
    for tag, text, opts, ans, _ in MC:
        if rnum(tag) != cur:
            cur = rnum(tag)
            recipe_head(doc, cur)
        lines = [lambda p, t=text, a=ans: (run(p, mc_note(a) + ' ', bold=True, size=10.5, color=P_RED), run(p, t, size=10.5))]
        lines += [(f'{"ABCD"[k]})  {o}', False) for k, o in enumerate(opts)]
        window(doc, tag_title(n, tag), REC[cur][1], lines)
        n += 1
    doc.add_page_break()
    banner(doc, IMGS['C'])
    mama_says(doc, 'Part C · Open questions. ' + OPEN_NOTE, 'mama_book.png')
    cur = None
    for tag, text, _ in OPEN:
        if rnum(tag) != cur:
            cur = rnum(tag)
            recipe_head(doc, cur)
        body = [(para, False) for para in text.split('\n')]
        body += [('Your answer:', True)] + [('_' * 86, False)] * 6
        window(doc, tag_title(n, tag), REC[cur][1], body)
        n += 1


def answer_key(doc):
    banner(doc, IMGS['K'])
    mama_says(doc, "Mama's Feedback! Check your answers here. False statements are corrected, every multiple-choice "
                   'answer has its reasoning, and open questions have key points with marks.', 'mama_cheer.png')
    n = 1
    recipe = None
    run(doc.add_paragraph(), 'PART A · TRUE OR FALSE', bold=True, size=13, color=P_RED, font=HEAD_FONT)
    for tag, _, ans, why in TF:
        window(doc, f'⭐ MAMA\'S FEEDBACK  ·  Q{n}  ·  {tag}', P_MINT,
               [lambda p, a=ans, w=why: (run(p, ('True. ' if a else 'False. '), bold=True, size=10.5,
                                             color=(46, 140, 80) if a else P_RED), run(p, w, size=10.5))], title_fg=P_NAVY)
        n += 1
    run(doc.add_paragraph(), 'PART B · MULTIPLE CHOICE', bold=True, size=13, color=P_RED, font=HEAD_FONT)
    for tag, _, opts, ans, why in MC:
        sel = ans if isinstance(ans, list) else [ans]
        window(doc, f'⭐ MAMA\'S FEEDBACK  ·  Q{n}  ·  {tag}', P_MINT,
               [lambda p, a=ans, s=sel, o=opts, w=why: (
                   run(p, f'{letters(a)})  {" / ".join(o[i] for i in s)}. ', bold=True, size=10.5, color=(46, 140, 80)),
                   run(p, w, size=10.5))], title_fg=P_NAVY)
        n += 1
    run(doc.add_paragraph(), 'PART C · OPEN QUESTIONS (5 marks each)', bold=True, size=13, color=P_RED, font=HEAD_FONT)
    mama_says(doc, KEY_NOTE, 'mama_happy.png')
    for tag, _, ans in OPEN:
        window(doc, f'⭐ MAMA\'S FEEDBACK  ·  Q{n}  ·  {tag}', P_MINT, [(ans, False)], title_fg=P_NAVY)
        n += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(SPR, 'sticker_tasty.png'), height=Inches(1.1))
    run(doc.add_paragraph(), CREDIT, size=8, color=(120, 120, 120), italic=True).font.size = Pt(8)


IMGS = {}


def render_images():
    IMGS['cover'] = img_cover()
    IMGS['header'] = img_header()
    IMGS['counter'] = img_counter()
    IMGS['A'] = img_banner('A', 'PART A · TRUE OR FALSE', P_YEL, 'mama_ok.png')
    IMGS['B'] = img_banner('B', 'PART B · MULTIPLE CHOICE', P_MINT, 'boy_grin.png')
    IMGS['C'] = img_banner('C', 'PART C · OPEN QUESTIONS', (255, 141, 161), 'girl_pigtails.png')
    IMGS['K'] = img_banner('★', "MAMA'S ANSWER KEY", P_RED, 'mama_cheer.png')


def main():
    render_images()
    doc = new_doc()
    cover(doc)
    questions(doc)
    doc.add_page_break()
    answer_key(doc)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    key = new_doc()
    cover(key, key_only=True)
    answer_key(key)
    key.save(KEY_OUT)
    write_markdown(MD_OUT)
    print(os.path.abspath(OUT), TOTAL, 'questions')
    print('TF true/false:', Counter(t[2] for t in TF))
    print('MC keys:', Counter(letters(m[3]) for m in MC if isinstance(m[3], int)), 'multi:', MULTI)


if __name__ == '__main__':
    main()
