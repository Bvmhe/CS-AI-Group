# Section 1: Hardware–Software Interaction

> **Big idea.** A model is not run on hardware directly. It passes through layers of software that decide which chip does what, in which order, and how data moves. Much of the real-world speed of AI comes from these layers, not from the chip's peak numbers.

---

## Learning Objectives

By the end of this section, you should be able to:

1. Describe the layers of the AI software stack, from framework to hardware.
2. Explain what a compiler does for an AI model and why operator fusion saves memory traffic.
3. Calculate the memory traffic of a chain of operations with and without fusion.
4. Define MFU (model FLOPs utilisation) and explain why it is almost always well below 100%.
5. Explain why unsupported operations and device switches slow a model down.

---

## 1.1 The Story in One Picture

Imagine you write a single line of Python: `y = relu(W @ x + b)`. To you, it looks like one step. To the chip, it is a long journey. Your code becomes a graph of operations, the graph is rewritten by a compiler, the rewritten graph is split into work for different chips, the work is scheduled, and only then does a transistor switch. Every layer in that journey is a place where speed can be won or lost. This is the whole story of this section.

**Figure 1.1.** The AI software stack. Each layer can hide detail, but each is also a place where work can be wasted. *Source: author's own diagram, based on the stack described in Module 01, Section 2.7.*

```text
  Application / model code      "y = relu(W @ x + b)"
          |
  Framework                     PyTorch, JAX, TensorFlow  (builds a graph of operations)
          |
  Compiler / graph optimiser    XLA, TensorRT, Core ML  (fuses, reorders, assigns to devices)
          |
  Runtime and libraries         CUDA libraries, TPU runtime  (launches work, manages memory)
          |
  Driver and firmware           talks to the chip
          |
  Hardware                      MAC units, on-chip memory, HBM, interconnect
```

Everything else in this section is a detail of this picture.

---

## 1.2 The Software Stack

A programmer writes a model in a **framework** such as PyTorch, JAX or TensorFlow. Between that code and the transistors there are several layers. Each layer hides detail from the one above, and each is a place where performance can be won or lost.

The key point: **the same model can run at very different speeds on the same chip**, depending on how well these layers cooperate. This is called *hardware–software co-design* when the chip and the software are designed together, as with TPUs and the XLA compiler.

---

## 1.3 What a Compiler Does for an AI Model

A model is first turned into a **computation graph**: a diagram where each node is an operation (matrix multiply, add, ReLU) and each arrow is a tensor flowing between them. The compiler then improves this graph. The four most important jobs are:

**Table 1.1.** Four jobs of an AI compiler.

| Job | What it does | Why it helps |
|---|---|---|
| **Operator fusion** | Merges several operations into one pass over the data | Intermediate results never travel to and from off-chip memory |
| **Layout and tiling** | Cuts large tensors into tiles that fit in on-chip memory | Data stays in fast memory while it is reused |
| **Precision assignment** | Chooses FP16, BF16 or INT8 for each operation | Fewer bytes moved, more MACs per cycle |
| **Placement and scheduling** | Decides which chip or core runs each part, and in what order | Keeps units busy; overlaps compute with data transfer |

### Worked Example 1.A: Why Fusion Saves Memory Traffic

*Illustrative numbers.* A tensor has 1,000,000 FP16 values (2 bytes each), so it is 2 MB. We apply three element-wise operations in a row: multiply by a scale, add a bias, apply ReLU.

**Unfused.** Each operation reads the whole tensor from memory and writes a whole result back.

- 3 operations × (1 read + 1 write) × 2 MB = **12 MB** of traffic.

**Fused.** One combined operation reads the tensor once, does all three steps on each value while it is in a register, and writes the final result once.

- 1 read + 1 write × 2 MB = **4 MB** of traffic.

Fusion cuts memory traffic by a factor of 3. Because these operations do almost no arithmetic per byte, they are memory-bound (Module 01, Section 4.3), so a 3× cut in traffic gives close to a 3× cut in time.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
n = 1_000_000
bytes_per_value = 2                      # FP16
tensor_mb = n * bytes_per_value / 1e6    # 2.0 MB

unfused = 3 * 2 * tensor_mb              # 3 ops, each reads + writes
fused = 1 * 2 * tensor_mb                # one read + one write

print(unfused, fused, unfused / fused)
```

Expected output:

```text
12.0 4.0 3.0
```

> **Going further.** The compiler can only fuse operations it understands. A custom operation it does not recognise acts as a wall: the data must be written out before the wall and read back after it.

---

## 1.4 Utilisation at System Level: MFU

Module 01 defined *utilisation* as the share of peak that a chip actually achieves. For training large models, the standard measure is **MFU (model FLOPs utilisation)**:

```text
MFU = (FLOPs the model mathematically needs per second) ÷ (peak FLOP/s of the hardware)
```

It counts only the useful arithmetic of the model, not wasted or repeated work. Well-tuned large training runs commonly report MFU somewhere in the range of 30–60%; the rest is lost to waiting for memory, waiting for other chips, and gaps between operations. A low MFU means you are paying for idle silicon.

*Example.* A chip has a peak of 275 TFLOP/s. A training job achieves 110 TFLOP/s of useful model arithmetic. MFU = 110 ÷ 275 = **40%**.

> **Caution: MFU is not the same as hardware utilisation.** A chip can look "100% busy" in a monitoring tool while doing little useful arithmetic (for example, spending all its time copying data). MFU measures *useful* work, which is why it is preferred.

---

## 1.5 Device Switches and Fallbacks

In Module 01 you saw that an unsupported operation "falls back" to the CPU on a phone. The same problem appears in data centres. Whenever execution moves between devices (CPU to accelerator, or one chip to another), data must be copied, and the accelerator waits.

Consider a model with 10 operations where operation 5 is unsupported on the accelerator. The graph splits into three segments: operations 1–4 on the accelerator, operation 5 on the CPU, operations 6–10 on the accelerator. That is **two device switches**, each with a copy. If the copies are large, they can cost more time than all the accelerated arithmetic saved.

---

## 1.6 Activities for Section 1

- [ ] **Activity 1.A.** A tensor has 4,000,000 FP16 values. Four element-wise operations are applied in a row. Calculate the memory traffic (in MB) unfused and fused, and the percentage saved.
- [ ] **Activity 1.B.** A chip has a peak of 400 TFLOP/s. A training job achieves 140 TFLOP/s of useful model arithmetic. What is the MFU?
- [ ] **Activity 1.C.** A model has 12 operations in a row. Operations 3 and 8 are unsupported on the accelerator. How many device switches occur?
- [ ] **Activity 1.D.** In two or three sentences, explain why a compiler needs to know the size of a chip's on-chip memory when it decides how to tile a large matrix.

### Answer Key for Activities

**Activity 1.A.**
- Tensor size = 4,000,000 × 2 bytes = 8,000,000 bytes = **8.0 MB**.
- Unfused traffic = 4 operations × (1 read + 1 write) × 8 MB = **64 MB**.
- Fused traffic = 1 read + 1 write × 8 MB = **16 MB**.
- Percentage saved = (64 − 16) ÷ 64 = **75%**.

**Activity 1.B.**
- MFU = 140 ÷ 400 = **35%**.

**Activity 1.C.**
- Segments: operations 1–2 on the accelerator, operation 3 on the CPU, operations 4–7 on the accelerator, operation 8 on the CPU, operations 9–12 on the accelerator.
- There are **4 device switches** (accelerator→CPU, CPU→accelerator, accelerator→CPU, CPU→accelerator).

**Activity 1.D.** On-chip memory is fast but small, and data that does not fit must be written to slow off-chip memory and read back. If the compiler knows the size, it can choose a tile size that keeps the working set entirely on-chip, so each value is loaded once and reused many times. If the tile is too large, it spills and the reuse is lost; if it is too small, the arithmetic units are under-used.

---

## 1.7 Summary

- The AI software stack (framework, compiler, runtime, driver, hardware) decides how well a model uses a chip; the same model can run at very different speeds on the same hardware.
- A compiler turns a model into a computation graph and then fuses operations, tiles tensors, assigns precision and schedules work.
- Fusing three element-wise operations on a 2 MB tensor cut memory traffic from 12 MB to 4 MB, a 3× saving on a memory-bound step.
- MFU measures useful model arithmetic as a share of peak; values of 30–60% are typical for well-tuned large runs.
- Every switch between devices adds data copies; unsupported operations can therefore slow a whole model.

---

## References

Kurose, J. F., & Ross, K. W. (2020). *Computer networking: A top-down approach* (8th ed.). Pearson.
