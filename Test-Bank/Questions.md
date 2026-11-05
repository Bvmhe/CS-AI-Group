# Fengwei Fried Rice · Stage 2 Test Bank

*Team Fengwei Fried Rice · Stage 2 Study Guide · Hardware, Networks & Behaviourist AI · November 2026*

25 True/False · 25 multiple-choice (21 choose-one, 4 select-all) · 10 open · **60 questions**. Each question is tagged [Recipe · guide section · level]. The answers are in the separate Answer Key PDF and at the end of the Word file.

## Part A · True or False

Write T (true) or F (false) for each statement.

**1.** One MAC (multiply-accumulate) counts as two FLOPs. `[R1 · 1.1 · R]`

**2.** A TPU is more flexible than a CPU for general programs. `[R1 · 1.2 · U]`

**3.** If 90% of a job can be parallelised, 100 cooks give a speed-up of 100×. `[R1 · 1.3 · A]`

**4.** At batch size 1, a large layer on a typical accelerator is usually memory-bound. `[R1 · 1.4 · U]`

**5.** In Kitchen Lab 1 (Race the Crew), the GPU's speed-up over the CPU is LARGER for 256 × 256 matrices than for 4,096 × 4,096 matrices. `[R1 · Lab 1 · A]`

**6.** Encapsulation means each layer adds its own header to the data from the layer above. `[R2 · 2.1 · U]`

**7.** UDP guarantees that every packet arrives in order. `[R2 · 2.2 · U]`

**8.** Bandwidth and latency are two names for the same quantity. `[R2 · 2.4 · U]`

**9.** In a ring all-reduce, the traffic per chip approaches 2 × S as the number of chips grows. `[R2 · 2.4 · A]`

**10.** In Kitchen Lab 2, a server answers ping with an average round-trip time of 12 ms. Using the speed-of-light floor of 5 µs per km (one way), the server can be at most about 1,200 km away. `[R2 · Lab 2 · A]`

**11.** With rewards [0, 0, 0, 10] and γ = 0.9, the discounted return is 7.29. `[R3 · 3.2 · A]`

**12.** A discount factor γ close to 0 makes the agent far-sighted. `[R3 · 3.2 · U]`

**13.** Reward hacking means the agent earns a high reward without doing the intended task. `[R3 · 3.3 · U]`

**14.** A purely greedy agent can get stuck on a poor option forever. `[R3 · 3.4 · U]`

**15.** In Kitchen Lab 3 (GridWorld), setting epsilon = 0 guarantees that the agent eventually finds the best route. `[R3 · Lab 3 · U]`

**16.** A single camera directly measures the distance to every pixel. `[R4 · 4.1 · R]`

**17.** A LiDAR pulse returning after 400 ns means the object is about 60 m away. `[R4 · 4.1 · A]`

**18.** Radar measures the speed of an object directly using the Doppler effect. `[R4 · 4.1 · U]`

**19.** Two sensors that fail for the same reason (for example fog) give the same benefit as two independent sensors. `[R4 · 4.4 · U]`

**20.** In Kitchen Lab 4 (phyphox), a phone lying still on a table shows an acceleration (without g) of exactly 0 m/s². `[R4 · Lab 4 · U]`

**21.** Fusing three element-wise operations on a 2 MB tensor cuts memory traffic from 12 MB to 4 MB. `[R5 · 5.1 · A]`

**22.** In data parallelism, each chip stores only a slice of the model's weights. `[R5 · 5.2 · U]`

**23.** A PUE of 1.0 means the facility uses no energy beyond the computing equipment. `[R5 · 5.4 · U]`

**24.** If each of 4,096 chips has an MTBF of 43,800 hours, the whole system fails about every 10.7 hours. `[R5 · 5.4 · A]`

**25.** In Kitchen Lab 5, the FIRST call of a jax.jit function is much slower than later calls mainly because XLA compiles it. `[R5 · Lab 5 · U]`

## Part B · Multiple choice

Each question says how many answers to choose: "(Choose ONE)" means exactly one correct option; "(Select ALL that apply)" means one or more options are correct and you must choose all of them.

**26. (Choose ONE)** Weights [2, −1, 0.5], inputs [1, 4, 6]. What is the neuron output (no bias)? `[R1 · 1.1 · A]`

- A) 1.0
- B) 3
- C) 5
- D) −1

**27. (Choose ONE)** Which chip best fits a voice assistant in a small battery-powered speaker? `[R1 · 1.2 · U]`

- A) GPU
- B) Server CPU
- C) NPU
- D) TPU pod

**28. (Choose ONE)** In a 2 × 2 systolic array with W = [[1, 2], [3, 4]] and x = [5, 6], the output y = W·x is: `[R1 · 1.2 · A]`

- A) [12, 30]
- B) [23, 34]
- C) [11, 23]
- D) [17, 39]

**29. (Choose ONE)** A chip has 10 TFLOP/s peak and 500 GB/s memory bandwidth. Its ridge point is: `[R1 · 1.4 · A]`

- A) 2 FLOP/byte
- B) 20 FLOP/byte
- C) 50 FLOP/byte
- D) 5,000 FLOP/byte

**30. (Choose ONE)** Kitchen Lab 1: ten 4,096 × 4,096 matrix multiplications (about 687 billion MACs in total) take 0.25 s on a T4 GPU. What throughput did the GPU reach? `[R1 · Lab 1 · A]`

- A) 11 TFLOP/s
- B) 0.07 TFLOP/s
- C) 2.7 TFLOP/s
- D) 5.5 TFLOP/s

**31. (Select ALL that apply)** Which of these does TCP provide that UDP does NOT? `[R2 · 2.2 · U]`

- A) Acknowledgement of delivered data
- B) In-order delivery
- C) Port numbers to reach the right program
- D) Retransmission of lost packets

**32. (Choose ONE)** Which protocol is built on UDP to speed up connection set-up for the web? `[R2 · 2.2 · U]`

- A) QUIC
- B) ARP
- C) SMTP
- D) FTP

**33. (Choose ONE)** A factory camera must stop a machine within 15 ms. The best place to run the model is: `[R2 · 2.3 · U]`

- A) A distant cloud region
- B) On or near the machine (edge)
- C) A user's laptop at home
- D) Any server; latency does not matter

**34. (Choose ONE)** Tail latency refers to: `[R2 · 2.4 · U]`

- A) Upload speed
- B) The average delay
- C) The slowest small fraction of requests
- D) The delay of the last packet only

**35. (Choose ONE)** Multi-access edge computing (MEC) in 5G mainly helps AI services by: `[R2 · 2.5 · U]`

- A) Putting computing close to the base station
- B) Replacing TCP
- C) Removing the need for antennas
- D) Increasing screen resolution

**36. (Choose ONE)** A corridor of squares 1-2-3 leads to the goal, one step after square 3 (reward 1). With γ = 0.9, V(1) equals: `[R3 · 3.2 · A]`

- A) 1.0
- B) 0.729
- C) 0.81
- D) 0.9

**37. (Choose ONE)** Q = 0.2, r = 1, max Q(s′) = 0.5, α = 0.1, γ = 0.9. After one Q-learning update the new Q is: `[R3 · 3.3 · A]`

- A) 0.45
- B) 1.45
- C) 0.245
- D) 0.325

**38. (Select ALL that apply)** Which of these rewards are DENSE (given at almost every step)? `[R3 · 3.3 · U]`

- A) +speed every step
- B) +1 only on reaching the charging station
- C) −energy used every step
- D) +1 only for winning the game

**39. (Choose ONE)** A bandit has arms that pay 0.2, 0.5 and 0.8 on average. The agent knows the best arm and uses ε = 0.1 (when exploring, it picks one of the three arms at random). Expected reward per pull: `[R3 · 3.4 · A]`

- A) 0.50
- B) 0.77
- C) 0.80
- D) 0.74

**40. (Choose ONE)** Domain randomisation helps sim-to-real transfer by: `[R3 · 3.5 · U]`

- A) Removing the reward
- B) Using only real data
- C) Making the simulator perfect
- D) Training on many varied simulated conditions

**41. (Select ALL that apply)** Which sensors measure distance DIRECTLY? `[R4 · 4.1 · U]`

- A) LiDAR
- B) A single camera
- C) Radar
- D) Ultrasonic

**42. (Choose ONE)** At 100 km/h, how far does a car travel during 0.3 s of processing latency? `[R4 · 4.2 · A]`

- A) 8.33 m
- B) 30 m
- C) 64.3 m
- D) 2.78 m

**43. (Choose ONE)** Radar says 50.0 m (σ = 0.5 m) and a camera says 52.0 m (σ = 2.0 m). The inverse-variance fused estimate is about: `[R4 · 4.3 · A]`

- A) 52.0 m
- B) 50.12 m
- C) 51.0 m
- D) 51.5 m

**44. (Choose ONE)** A stereo camera has f = 1,000 px and baseline B = 0.3 m. A disparity of 6 px means a depth of: `[R4 · 4.3 · A]`

- A) 300 m
- B) 42.9 m
- C) 50 m
- D) 60 m

**45. (Choose ONE)** Under SAE J3016, a driver-assist that steers and brakes but needs constant human supervision is: `[R4 · 4.4 · U]`

- A) Level 2
- B) Level 4
- C) Level 5
- D) Level 0

**46. (Choose ONE)** Training a 7B-parameter model with mixed-precision Adam needs about 16 bytes per parameter. Total training state: `[R5 · 5.2 · A]`

- A) 14 GB
- B) 56 GB
- C) 112 GB
- D) 224 GB

**47. (Choose ONE)** A 7 GB INT8 model is served on a chip with 1 TB/s memory bandwidth. The memory-bound limit for one user is about: `[R5 · 5.3 · A]`

- A) 1,000 tokens/s
- B) 4,571 tokens/s
- C) 14 tokens/s
- D) 143 tokens/s

**48. (Choose ONE)** The diameter (longest shortest path) of a 16 × 16 × 16 torus is: `[R5 · 5.5 · A]`

- A) 16 hops
- B) 24 hops
- C) 45 hops
- D) 48 hops

**49. (Select ALL that apply)** Which changes usually RAISE the MFU (model FLOPs utilisation) of a training job? `[R5 · 5.4 · U]`

- A) Fusing operations to cut memory traffic
- B) Overlapping communication with computation
- C) Making every step wait for a slow data loader
- D) Using a larger batch per chip

**50. (Choose ONE)** Kitchen Lab 5: you change the matrix size in the JAX program from 4,096 to 8,192. By what factor do the FLOPs of a @ a grow? `[R5 · Lab 5 · A]`

- A) 16×
- B) 2×
- C) 4×
- D) 8×

## Part C · Open questions

Write a developed answer of more than 100 words, or show full working (calculations, a diagram or worked steps) with a short explanation of what the result means.

**51.** A layer has 512 inputs and 2,048 outputs, with FP16 weights, on a chip with 10 TFLOP/s and 500 GB/s. Compute the arithmetic intensity at batch 1 and at batch 32, decide whether each is memory-bound or compute-bound, estimate the achieved TFLOP/s, and explain in your own words why batching helps. `[R1 · 1.4 · A]`

**52.** Evaluate running Face ID on the phone's Neural Engine instead of in the cloud. Give two advantages and two limitations of on-device AI, and decide whether the same choice would suit a large chatbot model. Justify your decision. `[R1 · 1.5 · E]`

**53.** A hospital wants an AI that reads X-rays in under 2 seconds and keeps patient data private, but the model needs a large GPU. Recommend cloud, edge (on-premises) or hybrid, and justify the design with latency, bandwidth, privacy and cost. `[R2 · 2.3 · E]`

**54.** 14 GB of gradients are all-reduced across 256 chips with 100 GB/s per chip, and one compute step takes 8.2 s. Calculate the traffic per chip, the communication time and the scaling efficiency, then explain one way to improve it and why it works. `[R2 · 2.4 · A]`

**55.** Design a reward for a home cleaning robot. Show how a naive reward could be hacked, propose a better reward, say whether it is sparse or dense, and compare the two rewards: which behaviour would each one produce? `[R3 · 3.3 · C]`

**56.** A student runs Kitchen Lab 3 (GridWorld TD learning) with epsilon = 0. After 2,000 steps the agent still walks the same long route; the arrows on squares it never visited still show their starting values. Explain why this plan failed, what the student should change, and what they should expect with epsilon = 0.5. Connect the result to the greedy agent in the bandit simulation and compare ε-greedy with UCB as a fix. `[R3 · Lab 3 · An]`

**57.** A car at 50 km/h has 0.15 s sensing-and-processing latency and brakes at 7 m/s². Compute the total stopping distance and the impact speed if a ball is detected 15 m ahead. What does this show about sensor range and latency when you design a safety system? `[R4 · 4.4 · A]`

**58.** In the offline version of Kitchen Lab 4, two students stand 20 m apart. One claps; the other starts a stopwatch on SEEING the clap and stops it on HEARING it. Five trials give t = 0.21, 0.08, 0.15, 0.30 and 0.11 s. Compute the speed of sound from each trial and from the mean time, compare with 343 m/s, and explain why single results scatter so much. Why does averaging help, what can averaging NOT fix, and how would you improve the experiment? `[R4 · Lab 4 · An]`

**59.** Estimate the training time, cost and energy for a 7B model on 1.4 trillion tokens with 256 chips (200 TFLOP/s each, 40% MFU, $2 per chip-hour, 400 W per chip, PUE 1.1). Use FLOPs ≈ 6 × parameters × tokens. Which single change would cut all three numbers, and why? `[R5 · 5.4 · A]`

**60.** Kitchen Lab 5 runs this JAX program on a Colab T4 GPU (FP32 peak about 8.1 TFLOP/s):

    f = jax.jit(lambda a: (a @ a).sum())

Measured times: size 1,024: first call 0.62 s, second call 0.0021 s; size 4,096: 0.034 s; size 8,192: 0.27 s. Explain the slow first call, compute the TFLOP/s and MFU for each size, and explain why the time ratio from 1,024 to 4,096 is far from the FLOP ratio while 4,096 → 8,192 is close to 8×. Predict the time for size 16,384. `[R5 · Lab 5 · P]`
