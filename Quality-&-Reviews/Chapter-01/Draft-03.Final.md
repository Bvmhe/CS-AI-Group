# Hardware, Networks & Behaviourist AI

## Module 01 · Section 3: Parallel Processing and AI Acceleration

> [!IMPORTANT]
> **Big idea.** Neural networks consist of huge numbers of independent MACs, so the way to go fast is to do many of them at the same time. But parallel hardware has limits, and the limits are as important as the speed-ups.

## Learning objectives

By the end of this section you should be able to:

1. Name and distinguish the four levels of parallelism (instruction-level, data-level, matrix-level, and across chips).
2. Count the MACs and FLOPs in a dense layer, and estimate the time for one pass on an idealised CPU core and an idealised parallel accelerator.
3. State Amdahl's law, apply it to a worked example, and explain its ceiling.
4. Explain why parallel hardware has limits: Amdahl's law, idle units, feeding the units, and communication.
5. Describe three acceleration techniques (quantisation, specialised units, batching) and calculate the memory footprint of a model in different precisions.

## Table of contents

- [3.1 Kinds of parallelism](#31-kinds-of-parallelism)
- [3.2 Worked example 3.A: how many MACs, and how long?](#32-worked-example-3a-how-many-macs-and-how-long)
- [3.3 Why parallel hardware has limits](#33-why-parallel-hardware-has-limits)
- [3.4 Other acceleration techniques](#34-other-acceleration-techniques)
- [3.5 Activities for Section 3](#35-activities-for-section-3)
- [3.6 Summary](#36-summary)
- [3.7 Glossary](#37-glossary)
- [3.8 Answer key](#38-answer-key)
- [References](#references)

---

## 3.1 Kinds of parallelism

"Parallel" simply means "at the same time". Hardware does this at several different levels:

| Level | Idea | Example |
|---|---|---|
| **Instruction-level** | A single core overlaps several instructions (like an assembly line) | Modern CPU pipelines |
| **Data-level (SIMD / SIMT)** | One instruction acts on many values at once | GPU warps; CPU vector units |
| **Matrix-level (systolic)** | A grid of MAC units passes data neighbour to neighbour | TPU matrix unit |
| **Across chips** (data, pipeline and model splitting) | Split the model, or the batch of data, over many chips | Training clusters (Module 02) |

**Table 3.1.** Four levels of parallelism in AI hardware.

### Terms

- **SIMD (Single Instruction, Multiple Data):** the same instruction is applied to a block of values in lock-step. On GPUs the equivalent idea is often called **SIMT** (Single Instruction, Multiple Threads): groups of threads (a *warp* on NVIDIA GPUs) execute together.
- **Thread:** one stream of instructions. GPUs run tens of thousands at once.
- **Data parallelism:** the same operation on different data at the same time (for example, 1,000 pixels each multiplied by the same filter weights).
- **Pipeline parallelism:** one big model is split into stages across several chips (for example, one chip handles layers 1–10 while another handles layers 11–20), and data flows from stage to stage.

GPUs are built for data parallelism. A very useful trick they use is **latency hiding**: when one group of threads has to wait for memory, the GPU switches instantly to another group that is ready. With thousands of threads available, the arithmetic units rarely sit idle. CPUs take the opposite approach: they spend silicon on big caches and prediction logic so that one thread rarely waits at all.

> [!NOTE]
> **Bridge to Networks.** The last row of the table is where hardware meets networking. Once a model is split across chips, the links between them (NVLink, InfiniBand; Module 02) decide how much of the extra compute you can actually use.

---

## 3.2 Worked example 3.A: how many MACs, and how long?

**Problem.** A dense (fully connected) layer takes an input vector of 1,024 numbers and produces 4,096 outputs. The weight matrix is 1,024 × 4,096. Estimate the time for one pass on (a) a single idealised CPU core and (b) an idealised parallel accelerator.

**Step 1: Count the MACs.** Each of the 4,096 outputs needs 1,024 multiplications and additions.

```text
MACs = 1,024 × 4,096 = 4,194,304   (about 4.2 million)
```

**Step 2: Convert to FLOPs.** One MAC = 2 FLOPs (one multiply, one add).

```text
FLOPs = 2 × 4,194,304 = 8,388,608   (about 8.4 million)
```

**Step 3: Estimate time on two idealised chips** (illustrative assumptions).

| Chip (idealised) | Assumption | Calculation | Time for one pass |
|---|---|---|---|
| Single CPU core, one MAC per cycle, 3 GHz | 3 × 10⁹ MACs/s | 4,194,304 ÷ 3 × 10⁹ | ≈ **1.4 ms** |
| Accelerator with 4,096 MAC units working in parallel, 1 GHz | 4,096 × 10⁹ = 4.096 × 10¹² MACs/s | 4,194,304 ÷ 4.096 × 10¹² | ≈ **1.0 µs** |

**Table 3.2.** Estimated time for one pass of the layer on two idealised chips.

**Step 4: Interpret.** The idealised parallel accelerator is about 1,365× faster (roughly 1,400×) on this layer, even though its clock is slower, because it does 4,096 MACs per cycle instead of 1.

The following Python program reproduces the numbers. Its complete expected output is shown below the code.

```python
macs = 1024 * 4096
cpu_time = macs / 3e9            # 1 MAC per cycle at 3 GHz
acc_time = macs / (4096 * 1e9)   # 4,096 MACs per cycle at 1 GHz

print(macs)                      # 4194304
print(round(cpu_time * 1e3, 2), "ms")   # ~1.40 ms
print(round(acc_time * 1e6, 2), "us")   # ~1.02 us
print(round(cpu_time / acc_time))       # ~1365
```

Expected output:

```text
4194304
1.4 ms
1.02 us
1365
```

> [!WARNING]
> **Important caveat.** These are ideal numbers. They assume the data is always ready the instant it is needed. Section 3.3 and Section 4 explain why real chips often cannot reach them.

---

## 3.3 Why parallel hardware has limits

Adding more units does not give unlimited speed-up. Four things get in the way.

### 1. Amdahl's law

If a fraction *p* of a job can be parallelised, the best possible speed-up on *N* units is

```text
speed-up = 1 ÷ ( (1 − p) + p ÷ N )
```

*Worked example.* With p = 0.9 and N = 100: speed-up = 1 ÷ (0.1 + 0.009) ≈ **9.2×**. Even with unlimited units it can never exceed 1 ÷ 0.1 = **10×**, because the 10% serial part remains. In an AI pipeline the serial part might be loading data, pre-processing, or a step that cannot be spread out.

The following Python program computes the worked example and the ceiling. Its complete expected output is shown below the code.

```python
def amdahl(p, n):
    return 1 / ((1 - p) + p / n)

print(round(amdahl(0.9, 100), 1))     # 9.2
print(round(amdahl(0.9, 10**9), 1))   # 10.0  (the ceiling)
```

Expected output:

```text
9.2
10.0
```

### 2. Idle units

A 4,096-unit accelerator can only be fully used if there are at least 4,096 independent MACs to do at once. A small layer leaves units idle, so **utilisation** (Section 1.6) falls.

### 3. Feeding the units

Units cannot compute without data. If memory cannot deliver numbers fast enough, the MACs wait. This is the subject of Section 4.

### 4. Communication

Across many chips, the time spent exchanging results can outweigh the gain from adding chips (Module 02).

---

## 3.4 Other acceleration techniques

Parallel MAC units are only one trick. Designers combine several. The three most important for this module are:

| Technique | What it does | Why it helps |
|---|---|---|
| **Lower precision (quantisation)** | Use INT8 or FP16 instead of FP32 | Smaller numbers: more fit in the same memory and more can be processed per cycle |
| **Specialised units** | NVIDIA Tensor Cores and Apple's Neural Engine are circuits that do small matrix multiplications as a single hardware step | Fewer instructions and less control overhead per MAC |
| **Batching** | Process several inputs together so the same weights are reused | Fetch weights once, use them many times (Section 4) |

**Table 3.3.** Three acceleration techniques and why they help.

> [!TIP]
> **Going further.** Three more techniques you may meet:
> - **Sparsity:** skip multiplications by zero, so there is no work for values that cannot change the answer.
> - **Operator fusion:** combine several steps (for example, multiply, add bias and apply ReLU) into one pass to avoid writing and re-reading intermediate results.
> - **Weight-stationary data reuse:** keep weights in place while data moves past them, as in systolic arrays.

### Reduced precision (quantisation) in more detail

Quantisation converts a model's numbers from a high-precision format (such as FP32) to a lower-precision one (such as INT8). Using fewer bits per number lets a chip fit more MAC units in the same area and move fewer bytes. The first TPU used 8-bit integers for inference (Jouppi et al., 2017); later accelerators support formats such as bfloat16 (a 16-bit float with the same range as FP32) and 8-bit floating point.

| Format | Bits | Bytes | Typical use |
|---|---:|---:|---|
| FP32 | 32 | 4 | Classic training; "full" precision |
| BF16 / FP16 | 16 | 2 | Training and inference on modern accelerators |
| FP8 | 8 | 1 | Newer training/inference hardware |
| INT8 | 8 | 1 | Inference on phones, edge devices, TPUs |
| INT4 and below | 4 | 0.5 | Aggressively compressed models |

**Table 3.4.** Common number formats in AI hardware.

The following Python program quantises four weights to INT8 and back, and reports the maximum error. Its complete expected output is shown below the code.

```python
import numpy as np

w = np.array([0.5, -0.31, 0.02, 0.97], dtype=np.float32)
scale = 1.0 / 127
q = np.round(w / scale).astype(np.int8)    # quantise
w_back = q.astype(np.float32) * scale      # dequantise

print(q)                      # [ 64 -39   3 123]
print(np.round(w_back, 4))    # [ 0.5039 -0.3071  0.0236  0.9685]
print("max error:", np.abs(w - w_back).max().round(4))
```

Expected output:

```text
[ 64 -39   3 123]
[ 0.5039 -0.3071  0.0236  0.9685]
max error: 0.0039
```

---

## 3.5 Activities for Section 3

- [ ] **Activity 3.A.** The layer in Worked example 3.A is changed so that the output has 8,192 values instead of 4,096. What happens to the MAC count?
- [ ] **Activity 3.B.** With the same 4,096-unit accelerator, roughly how long would the 8,192-output layer take?
- [ ] **Activity 3.C.** A job is 80% parallelisable. What is the maximum speed-up on 8 units, and what is the limit with unlimited units?
- [ ] **Activity 3.D.** A 7-billion-parameter model is stored in FP16. How much memory do its weights need? How much if converted to INT8? To 4-bit?

---

## 3.6 Summary

- Hardware exploits parallelism at four levels: instruction-level, data-level (SIMD/SIMT), matrix-level (systolic), and across chips (data, pipeline and model splitting).
- A dense layer from 1,024 inputs to 4,096 outputs contains 4,194,304 MACs (about 8.4 million FLOPs); on idealised hardware this takes about 1.4 ms on one CPU core and about 1.0 µs on a 4,096-unit accelerator, a speed-up of about 1,365×.
- Amdahl's law caps the speed-up: with 90% parallelisable work the ceiling is 10×, and on 100 units you get about 9.2×.
- Parallel hardware also loses to idle units, memory that cannot feed the units fast enough, and communication between chips.
- Quantisation, specialised units and batching are the three main acceleration techniques covered here; quantising four sample weights to INT8 gives a maximum error of 0.0039.

---

## 3.7 Glossary

| Term | Meaning |
|---|---|
| **Amdahl's law** | A formula giving the maximum speed-up from parallelising a fraction *p* of a job on *N* units: 1 ÷ ((1 − p) + p ÷ N). |
| **Batch / batching** | Processing several inputs together so the same weights are reused. |
| **BF16 (bfloat16)** | A 16-bit floating-point format with the same range as FP32. |
| **Data parallelism** | Doing the same operation on different data at the same time. |
| **FLOP** | Floating-point operation; one multiply or one add. One MAC equals two FLOPs. |
| **FP32 / FP16 / FP8** | Floating-point formats using 32, 16 or 8 bits per number. |
| **INT8 / INT4** | Integer formats using 8 or 4 bits per number. |
| **Latency hiding** | Switching to another ready group of threads while one group waits, so arithmetic units stay busy. |
| **MAC** | Multiply-accumulate: the operation `a × b + c`. |
| **Operator fusion** | Combining several operations into one pass to avoid writing and re-reading intermediate results. |
| **Pipeline parallelism** | Splitting a model into stages across several chips, with data flowing from stage to stage. |
| **Quantisation** | Converting numbers from a high-precision to a lower-precision format. |
| **SIMD / SIMT** | Single Instruction, Multiple Data / Threads: one instruction applied to many values or threads at once. |
| **Sparsity** | Skipping multiplications by zero. |
| **Thread** | One stream of instructions. |
| **Utilisation** | The fraction of a chip's arithmetic units that are actually doing useful work. |
| **Warp** | A group of GPU threads that execute together. |

---

## 3.8 Answer key

**Activity 3.A.**
The MAC count doubles, to 1,024 × 8,192 = **8,388,608 MACs** (about 16.8 million FLOPs), because there are twice as many outputs and each still needs 1,024 MACs.

**Activity 3.B.**
About **2 µs**, ideally: twice the MACs at the same throughput. In practice the accelerator may do slightly better than a simple doubling if the larger layer keeps more units busy, or slightly worse if memory becomes the bottleneck.

**Activity 3.C.**
On 8 units: 1 ÷ (0.2 + 0.8 ÷ 8) = 1 ÷ 0.3 ≈ **3.3×**.
With unlimited units: 1 ÷ 0.2 = **5×**, because the 20% serial part remains.

**Activity 3.D.**
FP16: 7 × 10⁹ × 2 bytes = **14 GB**.
INT8: 7 × 10⁹ × 1 byte = **7 GB**.
4-bit: 7 × 10⁹ × 0.5 bytes = **3.5 GB**.

---

## References

- Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Conference Proceedings*.
- Jouppi, N. P., et al. (2017). In-datacenter performance analysis of a Tensor Processing Unit. *Proceedings of ISCA 2017*.

[← Previous: Section 2](Draft-02.2.md) · [Back to module overview](Draft-03.1.md) · [Next: Section 4 →](Draft-04.2.md)
