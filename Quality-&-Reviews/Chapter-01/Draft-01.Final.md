# Module 01 · Section 1: Definition and Fundamental Concepts

*Hardware, Networks & Behaviourist AI · Module 01: AI Hardware Architecture*

> [!NOTE]
> **Where this module is going.** Section 1 defines AI hardware and the arithmetic it exists to speed up. Section 2 compares CPUs, GPUs, TPUs and NPUs. Section 3 shows how parallelism speeds up MACs, and where it stops helping. Section 4 explains why memory, not arithmetic, is often the real limit. Section 5 applies everything to a real device, the Apple Neural Engine.

> [!IMPORTANT]
> **Big idea.** AI hardware is not magic: it is ordinary computing hardware that has been arranged so that the one kind of arithmetic that AI does all day is cheap in time, energy, and money.

## What you will be able to do after this section

- Define AI hardware and name its five categories (1.1).
- Count the MACs and FLOPs in a simple layer (1.2).
- Explain why simply waiting for a faster CPU is not the answer (1.4).
- Tell training from inference and say how their hardware needs differ (1.5).
- Name the metrics used to compare chips and say what each one tells you (1.6).

> [!NOTE]
> **Running example.** Throughout the module we return to one task: unlocking a phone with your face (Face ID). A camera takes a picture, a neural network turns it into a list of numbers, and the phone compares those numbers with the ones saved at set-up. It must take a fraction of a second, use almost no battery, and never send your face to anyone else’s computer.

## Table of contents

- [1.1 What is AI hardware?](#11-what-is-ai-hardware)
- [1.2 What do AI models actually compute?](#12-what-do-ai-models-actually-compute)
- [1.3 Key terms](#13-key-terms)
- [1.4 Why not just use a faster CPU?](#14-why-not-just-use-a-faster-cpu)
- [1.5 Training versus inference](#15-training-versus-inference)
- [1.6 Choosing hardware: metrics to compare](#16-choosing-hardware-metrics-to-compare)
- [Section 1 summary](#section-1-summary)
- [Answer key](#answer-key)
- [References](#references)

---

## 1.1 What is AI hardware?

AI hardware is any physical computing device (a processor, a memory system or the wiring between them) that is designed, or is well suited, to run machine-learning workloads efficiently.

The word *efficiently* matters. A laptop CPU can run a neural network, and so could a person with a calculator and a lot of patience. The real questions are:

- How fast? (time)
- How much energy? (battery life, electricity bill, heat)
- At what cost? (price of the chip, size of the data centre)

AI hardware therefore includes the five categories in Table 1.1. Do not worry about the unfamiliar names yet: Section 2 explains the chips and Section 4 explains the memory. For now, just recognise them.

**Table 1.1.** Categories of AI hardware

| Category | What it is | Examples |
|---|---|---|
| General-purpose processors used for AI | Chips designed to run any program, but able to run AI | CPUs (central processing units) from Intel, AMD, Arm |
| Massively parallel processors | Thousands of simple cores doing the same thing to different data (a core is a unit that can run instructions on its own) | GPUs (graphics processing units) from NVIDIA, AMD |
| Domain-specific accelerators | Chips designed around the arithmetic of neural networks | TPUs (tensor processing units) from Google; NPUs (neural processing units) such as Apple’s Neural Engine |
| Memory systems | Fast memory placed near the compute units | On-chip SRAM (static random-access memory), HBM (High Bandwidth Memory) |
| Interconnects | Links between chips and between machines | NVLink, TPU inter-chip links, Ethernet/InfiniBand (Module 02) |

> [!TIP]
> **Going further.** A chip designed for one domain is often called an ASIC (application-specific integrated circuit). A TPU is an ASIC for neural-network arithmetic. The opposite extreme is the CPU, a general-purpose processor.

---

## 1.2 What do AI models actually compute?

Before comparing chips, we need to know what the chips are asked to do.

A neural network is built from layers. Most layers do one dominant operation: multiply a block of numbers (the *weights*, which the model learned during training) by another block of numbers (the *inputs*), then add the results together.

For one output value, the calculation looks like this:

```text
Output = (w1 × x1) + (w2 × x2) + (w3 × x3) + ... + (wn × xn)
```

Each “multiply, then add to a running total” step is called a **MAC** (multiply-accumulate).

### Worked example 1.A: a single neuron by hand

*Illustrative numbers.* A tiny neuron has three weights, w = [0.5, −1.0, 2.0], and receives three inputs, x = [4, 3, 1].

- **Step 1.** Start the running total at 0.
- **Step 2.** MAC #1: total = 0 + (0.5 × 4) = 0 + 2.0 = 2.0
- **Step 3.** MAC #2: total = 2.0 + (−1.0 × 3) = 2.0 − 3.0 = −1.0
- **Step 4.** MAC #3: total = −1.0 + (2.0 × 1) = −1.0 + 2.0 = 1.0

**Result.** The output is 1.0, computed with 3 MACs. (Real networks then pass this number through a simple function such as ReLU, which keeps positive numbers and turns negative ones into 0. That step is cheap compared with the MACs.)

You can check it yourself:

```python
weights = [0.5, -1.0, 2.0]
inputs = [4, 3, 1]

total = 0
for w, x in zip(weights, inputs):
    total += w * x          # one MAC per loop

print(total)                # 1.0
print(max(0, total))        # ReLU -> 1.0
```

**Exercise 1-1.** A neuron has weights w = [1.0, −2.0, 0.5, 3.0] and inputs x = [2, 4, 6, −1]. Work out the running total after each MAC, the final output, and the output after ReLU. How many MACs did it take? Then confirm your answer with the code above (change the two lists).

Now scale it up. One layer might have 4,096 neurons, each with 1,024 inputs. That is 1,024 × 4,096 = 4,194,304 MACs, about 4.2 million, for one layer and one input. A modern large language model needs billions to trillions of MACs to produce a single answer.

> [!IMPORTANT]
> **Key idea.** If a chip can do MACs very fast and very efficiently, it speeds up almost the whole model. This is why the MAC is the central operation in AI hardware design.

### Matrices: the same thing, written compactly

Writing out every MAC is tedious, so we group the weights into a table called a **matrix**, and the inputs into a list called a **vector**. “Multiply the weight matrix by the input vector” means “do all the neuron calculations at once”. When we process several inputs at the same time, they form a second matrix, and we speak of matrix multiplication. Almost all of AI hardware is, at heart, hardware for multiplying matrices. A matrix extended to more than two dimensions is called a **tensor**, which is where the “T” in TPU comes from.

**Figure 1.1.** A 3 × 3 weight matrix multiplied by a 3-value input vector. Notice that every output is a row of weights combined with the whole input vector.

```text
Weight matrix W          Input vector x     Output vector y
(rows = neurons)

[ w11 w12 w13 ]          [ x1 ]             [ w11·x1 + w12·x2 + w13·x3 ]
[ w21 w22 w23 ]    ×     [ x2 ]      =      [ w21·x1 + w22·x2 + w23·x3 ]
[ w31 w32 w33 ]          [ x3 ]             [ w31·x1 + w32·x2 + w33·x3 ]
```

Each output value = 3 MACs. Whole layer = 3 × 3 = 9 MACs.

The same idea in NumPy (the first row of W is the neuron from Worked example 1.A, so the first output is the 1.0 you already found):

```python
import numpy as np

W = np.array([[0.5, -1.0, 2.0],
              [1.0,  0.0, 1.0],
              [0.0,  2.0, -1.0]])
x = np.array([4, 3, 1])

y = W @ x                   # all 9 MACs in one matrix-vector product
print(y)                    # [1. 5. 5.]
```

**Exercise 1-2.** A layer has 512 inputs and 256 outputs. (a) How many MACs does it need for one input? (b) How many FLOPs? (c) Predict before calculating: if the number of outputs doubles to 512, what happens to the MAC count?

---

## 1.3 Key terms

These terms are used throughout the module. A complete glossary is at the end of the module. Some terms (bandwidth, utilisation) are only fully explained later, in Sections 3 and 4, so treat Table 1.2 as a first look.

**Table 1.2.** Key terms used in this module

| Term | Definition |
|---|---|
| Training | Adjusting a model’s weights using lots of data. Very compute-heavy; usually done in data centres. |
| Inference | Using a trained model to produce an answer (for example, unlocking your phone). Happens every time a user uses the AI. |
| Forward and backward pass | Forward pass: sending an input through the network to get an output. Backward pass: working out how to adjust the weights from the error. Training needs both; inference needs only the forward pass. |
| Batch size | How many inputs are processed together in one go. Batch size 1 means one input at a time. |
| MAC | Multiply-accumulate: multiply two numbers and add the result to a running total. |
| FLOP / FLOPs / FLOP/s | A FLOP is one floating-point operation (one add or multiply on decimal-style numbers); FLOPs is the plural. FLOP/s (also written FLOPS) means FLOPs per second. One MAC counts as 2 FLOPs. |
| TOPS | Tera (10¹²) operations per second. Often used for integer operations on AI chips. |
| Throughput | How much work is finished per second. |
| Latency | How long one task takes from start to finish. |
| Precision | How many bits are used to store each number: FP32 = 32-bit floating-point, FP16 = 16-bit float, INT8 = 8-bit integer. (BF16, a different 16-bit float format, and FP8, an 8-bit float, appear in Section 3.) Lower precision means less memory, less energy and usually faster hardware, at some cost in accuracy. |
| Accelerator | A chip that works alongside the CPU to speed up specific kinds of work. |
| Edge device | A device that runs AI close to the user (a phone, a car, a camera) instead of in a data centre. |
| Bandwidth | How many bytes per second can be moved between two parts of a system. |
| Utilisation | The share of a chip’s peak performance that a program actually achieves. |

> [!WARNING]
> **Caution: bits and bytes.** 1 byte = 8 bits. An FP32 number takes 4 bytes, an FP16 number 2 bytes, and an INT8 number 1 byte. Many mistakes in hardware calculations come from forgetting this conversion.

**Exercise 1-3.** A layer has 1,000 weights. How many bytes do the weights take in FP32, in FP16 and in INT8? Predict first: which format will be smallest, and by how much?

### Prefixes you will meet

**Table 1.3.** Metric prefixes

| Prefix | Symbol | Power of ten | Example |
|---|---|---|---|
| kilo | k | 10³ | 1 kB = 1,000 bytes |
| mega | M | 10⁶ | 8 MB of cache |
| giga | G | 10⁹ | 500 GB/s bandwidth |
| tera | T | 10¹² | 92 TOPS |
| peta | P | 10¹⁵ | data-centre clusters quote PFLOPS |
| milli / micro / nano / pico | m / µ / n / p | 10⁻³ / 10⁻⁶ / 10⁻⁹ / 10⁻¹² | 1 ms, 1 µs, 1 ns, 1 pJ |

Memory sizes are sometimes quoted in powers of two: 1 MiB = 2²⁰ = 1,048,576 bytes. The difference is small and does not change any conclusion in this module.

---

## 1.4 Why not just use a faster CPU?

For decades, the answer to “make computers faster” was simply “wait for the next CPU”. That stopped working for two reasons.

**Reason 1: the workload is highly repetitive and independent.** Millions of MACs can be done at the same time because they do not depend on each other: the result of neuron 17 does not change the calculation for neuron 18. A chip built for doing one complicated thing quickly wastes its effort on this. A chip built for doing a huge number of simple things at once does not.

**Reason 2: power and data-movement limits.** Chips can no longer simply be clocked faster without overheating. (The clock sets how many steps a chip takes each second; 1 GHz is a billion steps per second.) Historically, as transistors shrank, they also used less power, so each generation could run faster without getting hotter, a trend often called Dennard scaling. That trend ended in the mid-2000s, and clock speeds flattened at a few gigahertz. On top of that, fetching data from memory can cost far more energy than doing arithmetic on it (Section 4 puts numbers on this). Hennessy and Patterson, in their well-known textbook and Turing Award lecture, argue that because general-purpose processors have stopped improving as quickly as before, big gains in cost, energy, and performance must now come from domain-specific hardware (Hennessy & Patterson, 2019).

Gains now come from two directions, which this module keeps returning to:

- doing more things at once (parallelism), and
- moving less data (memory efficiency).

**Exercise 1-4.** In two or three sentences, explain to a friend why a CPU that is simply twice as fast would not solve the problem of running large neural networks. Use the words *independent* and *energy*.

---

## 1.5 Training versus inference

The same neural network is used in two very different ways, and the hardware requirements differ.

**Table 1.4.** Training compared with inference

| | Training | Inference |
|---|---|---|
| Goal | Learn weights from data | Use learned weights to answer a request |
| Frequency | Rare, long jobs (days to months) | Constant: millions of small requests |
| Main limit | Raw compute and chip-to-chip communication | Latency, energy and cost per request |
| Typical place | Large data-centre clusters | Data centres and edge devices (phones, cars) |
| Precision | Usually higher (FP32, BF16, FP16) | Often lower (INT8, FP8, even 4-bit) |
| Data movement | Forward pass and backward pass; many intermediate values stored | Forward pass only |
| Batch size | Large (hundreds or more) | Often 1 (one user at a time) |

This distinction explains much of the hardware landscape. The first Google TPU was built for inference only (Jouppi et al., 2017), and later generations added training. In April 2025 Google announced Ironwood, its seventh-generation TPU, which it describes as designed specifically for inference at scale (Vahdat, 2025).

> [!NOTE]
> **Why does the batch size matter?** Section 4 shows that a single-user request (batch size 1) tends to be limited by memory, whereas a large training batch tends to be limited by compute. That one fact shapes whole product lines. Face ID is a good example: each unlock is one inference on one face, so the batch size is 1.

**Exercise 1-5.** For each case, say whether it is mainly training or inference, and why. (a) A lab adjusts the weights of a new image model over three weeks on thousands of chips. (b) Your phone checks your face to unlock. (c) A video app suggests the next video to millions of viewers every minute. Then: which of these would you expect to use lower-precision numbers, and why?

---

## 1.6 Choosing hardware: metrics to compare

Section 1.3 defined the vocabulary. Table 1.5 shows how those ideas are used to choose between chips.

**Table 1.5.** Metrics for comparing hardware

| Metric | Question it answers | Why it matters |
|---|---|---|
| Throughput (TOPS, FLOP/s) | How much work per second? | Training speed |
| Latency | How long for one request? | User experience; real-time control |
| Performance per watt | How much work per joule? | Battery life; data-centre power bill |
| Utilisation | What share of peak is actually achieved? | Money paid for idle silicon |
| Performance per dollar | How much work per unit of cost? | Budget decisions |
| Memory capacity | How big a model fits? | Which models can run at all |

> [!WARNING]
> **Caution: peak is not typical.** A chip’s quoted “peak” number is what it could do if every arithmetic unit were busy every cycle. Real programs are usually far below that. Sections 3 and 4 explain why.

**Exercise 1-6.** Choose the two most important metrics from Table 1.5 for (a) Face ID on a phone and (b) a data-centre cluster training a large model. Give one reason for each choice.

---

## Section 1 summary

- AI hardware is any device (processor, memory, or the links between them) that runs machine-learning workloads efficiently, judged by time, energy and cost.
- Most of the work in a neural network is MACs. One MAC is 2 FLOPs, and a layer with m inputs and n outputs needs m × n MACs per input.
- A matrix multiplication is many MACs done together; a tensor is a matrix with more dimensions.
- Waiting for faster CPUs stopped working because the work is independent and repetitive, and because power and data movement limit single chips. Gains now come from parallelism and from moving less data.
- Training and inference put different pressures on hardware: batch size, precision and latency all differ.
- Peak numbers are not typical numbers. Utilisation shows what a chip really delivers.

---

## Answer key

<details>
<summary><strong>Show answers</strong></summary>

**Exercise 1-1.** Running total: 1.0 × 2 = 2.0; then 2.0 + (−2.0 × 4) = −6.0; then −6.0 + (0.5 × 6) = −3.0; then −3.0 + (3.0 × −1) = −6.0. The output is −6.0 after 4 MACs. ReLU turns a negative number into 0, so the output after ReLU is 0.

**Exercise 1-2.** (a) 512 × 256 = 131,072 MACs. (b) 2 × 131,072 = 262,144 FLOPs. (c) The MAC count doubles to 262,144, because each of the 512 outputs needs 512 MACs and there are twice as many outputs.

**Exercise 1-3.** FP32: 1,000 × 4 = 4,000 bytes. FP16: 1,000 × 2 = 2,000 bytes. INT8: 1,000 × 1 = 1,000 bytes. INT8 is smallest, one quarter of FP32, because 8 bits is 1 byte and 32 bits is 4 bytes.

**Exercise 1-4.** Model answer: Neural networks are made of millions of MACs that do not depend on each other, so a chip that does many of them at once beats a chip that does one thing twice as fast. Also, a faster clock means more heat and more energy use, and moving data to and from memory costs a lot of energy, so a single faster core quickly hits power limits.

**Exercise 1-5.** (a) Training: it adjusts weights using lots of data and runs for weeks. (b) Inference: one input, a trained model, an answer in a fraction of a second (batch size 1). (c) Inference: a trained model answering a constant stream of requests. Cases (b) and (c) are more likely to use lower precision such as INT8, because inference often does (Table 1.4) and smaller numbers save memory and energy.

**Exercise 1-6.** (a) Face ID: latency (the unlock must feel instant) and performance per watt (it must barely touch the battery). Memory capacity is also a reasonable answer, since the model has to fit on the phone. (b) Training cluster: throughput (how fast the training finishes) and performance per dollar or per watt (the cost and the power bill of running thousands of chips for weeks).

</details>

---

## References

Hennessy, J. L., & Patterson, D. A. (2019). A new golden age for computer architecture. *Communications of the ACM, 62*(2), 48–60.

Jouppi, N. P., et al. (2017). In-datacenter performance analysis of a Tensor Processing Unit. *Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA)*, 1–12.

Vahdat, A. (2025, April 9). Ironwood: The first Google TPU for the age of inference. *Google Cloud Blog*. https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/
