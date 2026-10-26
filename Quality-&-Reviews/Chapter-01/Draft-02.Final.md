# Hardware, Networks & Behaviourist AI
## Module 01 — Section 2: CPU vs GPU vs TPU

**Version:** 1.1  
**Topic:** Computer architecture for artificial intelligence  
**Estimated study time:** 20–25 minutes

---

## Learning Objectives

By the end of this section, learners should be able to:

1. Explain the main differences between CPUs, GPUs, and TPUs.
2. Compare their flexibility, parallel-processing capabilities, energy efficiency, and typical AI applications.
3. Explain how a systolic array performs matrix multiplication and reduces data movement.
4. Interpret the findings of the first Google TPU research paper in its historical context.
5. Explain the role of NPUs in small devices and evaluate how software affects hardware performance.

## 2.1 The Three Chips: The Big Idea

### Big Idea

CPUs, GPUs, and TPUs are designed for different computing needs. CPUs prioritise flexibility, GPUs provide parallel processing for many different workloads, and TPUs specialise in accelerating tensor operations used in AI.

The main trade-off is between **flexibility and specialisation**. A specialised processor can be highly efficient for supported tasks, but its advantages depend on the workload, software, memory system, and power requirements.

### 2.1.1 Central Processing Unit (CPU)

A CPU is a general-purpose processor designed to execute a wide variety of instructions.

Its cores can handle different tasks, including operating-system operations, program control, data processing, and tasks that cannot easily be parallelised.

**Typical strengths:**
- Flexible and suitable for general-purpose computing.
- Effective at handling branching, decision-making, and sequential tasks.
- Useful for coordinating other processors and managing data.

**Typical limitation:** A CPU may be less efficient than a specialised accelerator when performing very large numbers of similar mathematical operations.

### 2.1.2 Graphics Processing Unit (GPU)

A GPU is designed to perform many operations in parallel. Although GPUs were originally developed for graphics, their parallel-processing capabilities also make them useful for AI workloads.

Neural networks frequently perform matrix multiplication and other operations across large collections of data. These operations can often be divided into smaller calculations that run concurrently.

**Typical strengths:**
- High throughput for parallel workloads.
- Suitable for training and inference with many AI models.
- Programmable for graphics, scientific computing, and other parallel tasks.

**Typical limitation:** Performance depends on the workload, memory bandwidth, software optimisation, and power consumption.

### 2.1.3 Tensor Processing Unit (TPU)

A TPU is a Google-designed application-specific integrated circuit (ASIC) that accelerates tensor operations, particularly those used in machine learning.

TPU architectures use specialised arithmetic hardware, including matrix-multiplication units. Many designs use systolic arrays, in which data moves through interconnected processing elements.

**Typical strengths:**
- Specialised hardware for supported AI operations.
- High throughput for suitable matrix and tensor workloads.
- Potentially high energy efficiency when the workload and software are well matched to the hardware.

**Typical limitation:** Flexibility, available operations, performance, and ease of programming depend on the particular TPU generation and its software ecosystem.

### An Everyday Analogy

Imagine three kitchens:

- **CPU:** A small team of experienced chefs who can prepare many different dishes.
- **GPU:** A large team that can perform the same preparation task on many ingredients simultaneously.
- **TPU:** A specialised production line designed to perform particular operations efficiently at scale.

No kitchen is automatically the best choice for every meal. The appropriate choice depends on what must be prepared, how much is needed, and the available resources.

---

## 2.2 Comparing CPU, GPU, and TPU

**Table 2.1.** General comparison of processor architectures.

| Feature | CPU | GPU | TPU |
|---|---|---|---|
| Primary design goal | General-purpose computing | Parallel processing | Specialised AI computation |
| Processing organisation | A relatively small number of complex cores | Many parallel processing units | Specialised tensor and matrix-processing hardware |
| Flexibility | High | High for supported parallel workloads | More specialised |
| Sequential tasks | Generally well suited | Can be less efficient for highly sequential tasks | Depends on supported operations |
| Parallel AI calculations | Suitable, but may have lower throughput | Well suited | Well suited to supported tensor operations |
| Energy efficiency | Depends on the workload | Depends on the workload and GPU design | Can be high for suitable workloads |
| Typical applications | Program control, data preparation, operating systems | AI training, inference, graphics, scientific computing | Large-scale AI training and inference |
| Programming | General-purpose languages and compilers | CUDA, OpenCL, and supported frameworks | Supported frameworks and compiler/runtime tools |
| Main limitation | May have fewer parallel arithmetic resources | Memory bandwidth and power can limit performance | Specialisation and software compatibility can limit flexibility |

**Important:** The table describes general tendencies, not universal performance rankings. Modern CPUs and GPUs include specialised instructions and arithmetic units, while TPU capabilities vary by generation.

Sources: [Jouppi et al. (2017)](https://doi.org/10.1145/3079856.3080246), [Sze et al. (2017)](https://doi.org/10.1109/JPROC.2017.2761740), and [Hennessy and Patterson (2019)](https://doi.org/10.1145/3282307).

### 2.2.1 Important Terms

- **Core:** A processing unit capable of executing instructions.
- **Cache:** Small, fast memory that stores frequently or recently used data.
- **Parallel processing:** Performing multiple calculations at the same time.
- **Tensor:** A generalised multidimensional data structure.
- **ASIC:** An application-specific integrated circuit designed for particular tasks.
- **Compiler:** Software that translates program instructions into a form that can execute on a target processor.
- **Inference:** Using a trained AI model to produce predictions or other outputs.
- **Throughput:** The amount of work completed per unit of time.
- **Latency:** The time required to complete a particular operation or request.
- **Energy efficiency:** The amount of useful work completed for a given amount of energy.
- **MAC:** Multiply-accumulate operation, commonly expressed as \(a \times b + c\).
- **TOPS:** Trillions of operations per second. The precision and definition of an operation must be specified when comparing reported figures.

---

## 2.3 Where the Chip Area Goes

### Big Idea

Processor designs allocate hardware resources differently. General-purpose processors devote resources to flexible instruction execution, while specialised accelerators devote more resources to the calculations they are designed to perform.

**Figure 2.1. Conceptual comparison of processor designs.**

```text
+----------------------+  +----------------------+  +----------------------+
|         CPU          |  |         GPU          |  |         TPU          |
+----------------------+  +----------------------+  +----------------------+
| Complex cores        |  | Parallel processing  |  | Matrix-processing    |
|                      |  | units                |  | units                |
| Large caches         |  | Arithmetic units     |  | Systolic array       |
|                      |  |                      |  |                      |
| Control logic        |  | Parallel workloads   |  | On-chip buffers      |
+----------------------+  +----------------------+  +----------------------+
| General-purpose      |  | Parallel computation |  | Specialised tensor   |
| processing           |  |                      |  | computation          |
+----------------------+  +----------------------+  +----------------------+
