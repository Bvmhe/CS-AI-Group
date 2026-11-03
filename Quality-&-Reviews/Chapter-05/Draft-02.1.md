# Section 2: Distributed AI Systems

> [!IMPORTANT]
> **Big idea.** One chip is not enough: either the model does not fit, or training would take years. Splitting the work across many chips solves both problems, but every split creates communication, and communication is the price of going distributed.

## Learning objectives

By the end of this section you should be able to:

- Explain two reasons for distributing a model: memory and time.
- Calculate the memory needed to train a model and decide how many chips are needed to hold it.
- Distinguish data, tensor and pipeline parallelism, and say when each is used.
- Calculate the per-chip traffic of a ring all-reduce and the resulting communication overhead.
- Calculate scaling efficiency from measured times.

## 2.1 Why distribute? Two reasons

**Reason 1: the model does not fit.** Training needs far more memory than the weights alone. With the common *mixed-precision Adam* recipe, each parameter needs roughly:

| Item | Bytes per parameter |
|---|---|
| Weights (BF16) | 2 |
| Gradients (BF16) | 2 |
| Master copy of weights (FP32) | 4 |
| Optimiser state: two moments (FP32 each) | 8 |
| **Total** | **16** |

*Table 2.1. Approximate training memory per parameter (a widely used rule of thumb; exact recipes vary). Activations (intermediate values) come on top.*

### Worked example 2.A: does our 7B model fit on one chip?

*Illustrative numbers.* Suppose each chip has 96 GB of memory.

- Training state = 7 × 10⁹ parameters × 16 bytes = **112 GB**.
- 112 GB > 96 GB, so the training state alone does not fit on one chip, before counting any activations.
- If the state is *sharded* (each chip holds only a slice) across 256 chips, each chip holds 112 ÷ 256 ≈ **0.44 GB** of state, leaving the rest of its memory for activations.

```python
import math

params = 7e9
bytes_per_param = 16
state_gb = params * bytes_per_param / 1e9
chip_gb = 96

print(state_gb)                       # 112.0
print(math.ceil(state_gb / chip_gb))  # 2  (minimum chips, state only)
print(round(state_gb / 256, 2))       # 0.44 GB per chip when sharded 256 ways
```

Expected output:

```
112.0
2
0.44
```

**Reason 2: time.** Training our model needs about 6 × N × D FLOPs, where N is the parameter count and D is the number of training tokens (a standard estimate; see Section 4.2). For N = 7 × 10⁹ and D = 1.4 × 10¹², that is about 5.9 × 10²² FLOPs. One chip delivering 100 TFLOP/s effective would need about 5.9 × 10²² ÷ 10¹⁴ ≈ 5.9 × 10⁸ seconds, which is almost 19 years, if run alone. Parallel chips are not a luxury but a necessity.

## 2.2 Three ways to split the work

| Strategy | What is split | Each chip holds | Communication | Best when |
|---|---|---|---|---|
| **Data parallelism** | The batch of training examples | A full copy of the model (or a shard of its state) | Gradients are summed across chips once per step (all-reduce) | The model (or its shards) fits on one chip; you want more throughput |
| **Tensor (model) parallelism** | Individual matrices inside a layer | A slice of every layer | Partial results exchanged inside every layer; needs very fast links | A single layer is too big, or you want lower latency |
| **Pipeline parallelism** | Groups of layers (stages) | A few consecutive layers | Activations passed from stage to stage | The model is deep; links between stages are slower |

*Table 2.2. Three parallelism strategies.*

Real large-model training combines all three ("3D parallelism"): tensor parallelism inside a group of tightly linked chips, pipeline parallelism across groups, and data parallelism across replicas. The rule of thumb is to put the most communication-hungry strategy (tensor) on the fastest links.

```
Data parallel:     chip A [full model] <-- batch part 1
                   chip B [full model] <-- batch part 2     then: sum gradients

Tensor parallel:   chip A [left half of each layer]  <-> chip B [right half]

Pipeline parallel: chip A [layers 1-10] --> chip B [layers 11-20] --> chip C [layers 21-30]
```

*Figure 2.1. The three strategies in schematic form. Source: author's own diagram.*

> [!NOTE]
> **Bridge to Networks (Module 02).** Each strategy loads the network differently. Data parallelism sends large, regular messages once per step. Tensor parallelism sends smaller messages very often, so it needs the lowest latency and highest bandwidth. Pipeline parallelism sends modest messages between neighbours. This is why the network design decides which strategy is practical.

## 2.3 The cost of keeping chips in sync: all-reduce

In data parallelism, each chip computes gradients on its own part of the batch. Before the weights are updated, all gradients must be summed and shared with every chip. This operation is called an **all-reduce**.

In the common **ring all-reduce**, chips are arranged in a ring and pass chunks of the data around. For a gradient of S bytes on N chips, each chip sends and receives about

```
traffic per chip = 2 × (N − 1) ÷ N × S   bytes
```

Notice that for large N this approaches 2 × S, almost independent of N. That property is why data parallelism scales well in principle.

### Worked example 2.B: communication overhead for the 7B model

*Illustrative numbers.*

- Gradients: 7 × 10⁹ values in BF16 = **14 GB**.
- N = 256 chips. Traffic per chip = 2 × (255 ÷ 256) × 14 GB ≈ **27.9 GB**.
- Link bandwidth per chip: 100 GB/s. Time = 27.9 ÷ 100 ≈ **0.28 s**.
- Compute per step: the batch is 4 × 10⁶ tokens, so FLOPs = 6 × 7 × 10⁹ × 4 × 10⁶ = 1.68 × 10¹⁷. The cluster delivers 256 × 200 TFLOP/s × 40% MFU = 2.048 × 10¹⁶ FLOP/s, so compute takes about **8.2 s**.
- If communication is not overlapped with compute, one step takes 8.2 + 0.28 = 8.48 s, so **efficiency ≈ 96.7%**.

```python
grad_bytes = 14e9
N = 256
link_bw = 100e9                           # bytes/s per chip

traffic = 2 * (N - 1) / N * grad_bytes
comm = traffic / link_bw

flops_step = 6 * 7e9 * 4e6
cluster_rate = 256 * 200e12 * 0.40
compute = flops_step / cluster_rate

eff = compute / (compute + comm)
print(round(traffic / 1e9, 2))   # 27.89
print(round(comm, 2))            # 0.28
print(round(compute, 1))         # 8.2
print(f"{eff:.1%}")              # 96.7%
```

Expected output:

```
27.89
0.28
8.2
96.7%
```

The overhead is small here because each step has a lot of compute (a large batch). With a smaller batch per chip, compute per step shrinks while the all-reduce stays the same size, and the overhead grows. Real systems also *overlap* communication with the backward pass, which hides part of it.

> [!WARNING]
> **Simplification.** This example treats pure data parallelism with a full gradient on every chip. In practice the 112 GB training state of Worked example 2.A would be sharded, which changes the communication pattern (more, smaller collectives). The order of magnitude is still instructive.

## 2.4 Scaling efficiency

How well does adding chips pay off? Measure the **speed-up** and the **scaling efficiency**:

```
speed-up = T(1 chip) ÷ T(N chips)
scaling efficiency = speed-up ÷ N
```

**Example.** A job takes 100 hours on 1 chip and 2.0 hours on 64 chips. Speed-up = 50. Scaling efficiency = 50 ÷ 64 ≈ **78%**. The missing 22% went to communication, idle units, and the serial parts predicted by Amdahl's law (Module 01, Section 3.3).

*Figure 2.2. Ideal versus achieved scaling. The ideal line is a straight diagonal; real curves bend away as communication and serial work grow. Source: author's own diagram, based on Amdahl (1967).*

Two terms are used for the two ways to measure scaling:

- **Strong scaling:** fixed problem size, more chips. Efficiency falls as each chip gets less work.
- **Weak scaling:** problem size grows with the chip count. Easier to keep efficient, and what large training runs usually rely on.

## 2.5 Activities for Section 2

- **Activity 2.A.** A 13-billion-parameter model is trained with the 16-bytes-per-parameter recipe. Each chip has 80 GB. How much memory does the training state need, and what is the minimum number of chips needed just to hold it?
- **Activity 2.B.** A model has 1 billion parameters, gradients stored in BF16. It is trained with data parallelism on 8 chips connected at 50 GB/s each. Calculate the traffic per chip and the time for one ring all-reduce.
- **Activity 2.C.** A job takes 120 hours on 1 chip and 4.5 hours on 32 chips. Calculate the speed-up and scaling efficiency.
- **Activity 2.D.** Name the best-fitting parallelism strategy for each case: (a) the model fits on one chip, and you want to process more data per second; (b) a single matrix is too large for one chip's memory; (c) a very deep model whose layers can be grouped into stages on separate chips.

## 2.6 Summary

- Models are distributed for two reasons: memory (training needs about 16 bytes per parameter, so a 7B model needs about 112 GB of state) and time (the 7B model needs about 5.9 × 10²² FLOPs).
- Data, tensor and pipeline parallelism split the batch, the matrices and the layers respectively; large systems combine all three and put tensor parallelism on the fastest links.
- A ring all-reduce moves about 2 × (N − 1) ÷ N × S bytes per chip; for 14 GB of gradients on 256 chips that is about 27.9 GB, or about 0.28 s at 100 GB/s, a 3% overhead on an 8.2 s step.
- Scaling efficiency (speed-up ÷ N) shows how much of the added hardware is really used; a drop from 100% is caused by communication, idle units and serial work.

---

