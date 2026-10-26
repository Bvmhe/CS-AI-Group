
# Hardware, Networks & Behaviourist AI

## Module 01 · Section 2: CPU vs GPU vs TPU

> **Big idea**
>
> CPUs, GPUs, and TPUs are designed around different trade-offs between flexibility, parallel processing, and efficiency. A specialised chip can be very efficient for supported workloads, but it may be less flexible than a general-purpose processor.

## Table of Contents

- [2.1 The Three Chips in One Sentence Each](#21-the-three-chips-in-one-sentence-each)
- [2.2 Comparison Table](#22-comparison-table)
- [2.3 Visual: Where the Chip Area Goes](#23-visual-where-the-chip-area-goes)
  - [Worked Example 2.A: A 2 × 2 Systolic Array](#worked-example-2a-a-2--2-systolic-array)
- [2.4 What the First TPU Paper Measured](#24-what-the-first-tpu-paper-measured)
- [2.5 The NPU: AI on Small Devices](#25-the-npu-ai-on-small-devices)
- [2.6 Software: Why a Chip Is Only as Useful as Its Tools](#26-software-why-a-chip-is-only-as-useful-as-its-tools)
- [2.7 Activities for Section 2](#27-activities-for-section-2)
- [References](#references)
- [Answer Key](#answer-key)

---

## 2.1 The Three Chips in One Sentence Each

### CPU: Central Processing Unit

A **CPU** is a general-purpose processor designed to handle many different types of instructions. It usually has a relatively small number of powerful cores, with features such as caches, branch prediction, and out-of-order execution.

CPUs are useful for tasks that require flexibility, decision-making, and low response times. In an AI system, a CPU may prepare data, run application logic, coordinate other processors, or execute a small model.

### GPU: Graphics Processing Unit

A **GPU** is designed to perform many similar calculations in parallel. It was originally developed for graphics, where many pixels and vertices must be processed using similar operations.

The same ability is useful for AI. Neural networks perform large numbers of mathematical operations, especially matrix multiplications, that can often be carried out in parallel.

GPUs are programmable and support many different workloads, although their performance depends on the model, memory access patterns, software, and available hardware features.

### TPU: Tensor Processing Unit

A **TPU** is Google's application-specific integrated circuit (ASIC) designed to accelerate machine-learning calculations. Its architecture includes specialised hardware for tensor operations, including matrix multiplication.

A TPU can provide high throughput and energy efficiency for supported workloads. However, it is less general-purpose than a CPU, and its usefulness depends on compatible software, operations, and access to the hardware.

### An Everyday Analogy

Imagine three different kitchens:

- **CPU:** A small team of experienced chefs who can prepare many different dishes and adapt to changing orders.
- **GPU:** A large kitchen team where many cooks perform similar tasks at the same time.
- **TPU:** A specialised production line designed to make a particular product very efficiently.

No single design is automatically the best for every task. The right choice depends on what must be calculated, how much data is involved, the required response time, the power budget, and the available software.

### Key Terms

- **Core:** A processing unit that executes instructions.
- **Cache:** Small, fast memory that stores copies of data likely to be used again.
- **Branch prediction:** A technique that predicts the direction of a program branch so the CPU can continue working.
- **Out-of-order execution:** A technique that allows a processor to execute independent instructions in a different order when it is safe to do so.
- **Tensor:** A general term for a multidimensional array of numbers.
- **MAC (multiply-accumulate):** An operation that multiplies two values and adds the product to an accumulated result.
- **ASIC:** An application-specific integrated circuit designed for a particular set of tasks.

---

## 2.2 Comparison Table

The following table compares typical design goals. Actual performance varies between products and workloads.

**Table 2.1. General comparison of CPU, GPU, and TPU designs.**

| Feature | CPU | GPU | TPU |
|---|---|---|---|
| Main design goal | Flexibility and low response time | High-throughput parallel processing | Accelerating supported machine-learning operations |
| Processing organisation | A relatively small number of powerful cores | Many parallel processing units | Specialised matrix and tensor-processing hardware |
| Control logic | Complex control, caching, and instruction scheduling | Parallel execution with hardware and software designed for many tasks | Hardware specialised for supported tensor operations |
| Flexibility | Very high | High for supported parallel workloads | More limited than a general-purpose CPU |
| AI applications | Data preparation, application logic, smaller models, and coordination | Training and inference for many neural networks | Supported machine-learning workloads, particularly in Google's infrastructure |
| Main limitation | May provide less parallel throughput for large matrix calculations | Performance can be limited by memory bandwidth, power, or workload suitability | Depends on supported operations, software compatibility, availability, and cost |
| Programming tools | General-purpose languages and compilers | CUDA, OpenCL, and machine-learning frameworks | Frameworks and compilers such as TensorFlow, JAX, and XLA |

*Sources: Hennessy and Patterson (2019); Jouppi et al. (2017); Sze et al. (2017); NVIDIA CUDA documentation.*

### How to Read the Comparison

The table describes broad design differences, not a universal ranking.

For example, a GPU may be suitable for training a large neural network because it can execute many calculations in parallel. A CPU may be a better choice for a web server that handles varied tasks. A TPU may be appropriate for a supported machine-learning workload that can benefit from its specialised hardware.

However, a chip's theoretical performance does not guarantee that it will be the fastest or cheapest choice. Memory movement, software support, power consumption, and the type of calculation also matter.

---

## 2.3 Visual: Where the Chip Area Goes

The following diagram is a conceptual illustration. It is not a measured representation of the physical area of particular commercial chips.

**Figure 2.1. Conceptual differences in CPU, GPU, and TPU organisation.**

```text
        CPU                       GPU                       TPU
+--------------------+  +--------------------+  +--------------------+
| [Core] [Core]      |  | [Unit] [Unit]      |  | [MAC] [MAC] [MAC] |
| [Core] [Core]      |  | [Unit] [Unit]      |  | [MAC] [MAC] [MAC] |
|                    |  | [Unit] [Unit]      |  | [MAC] [MAC] [MAC] |
| Cache and control  |  | Many parallel units|  | Matrix engine     |
+--------------------+  +--------------------+  +--------------------+
   Flexible tasks        Parallel operations      Tensor operations
```

**Alt text:** A conceptual comparison of three processor designs. The CPU contains a small number of powerful cores and supporting control and cache resources. The GPU contains many parallel processing units. The TPU contains specialised multiply-accumulate hardware for tensor calculations.

**Source:** Original conceptual illustration based on the architectural descriptions in Hennessy and Patterson (2019), Jouppi et al. (2017), and Sze et al. (2017). The diagram is not to scale.

A CPU devotes resources to features that help it handle varied instructions efficiently. A GPU is designed to execute many operations in parallel. A TPU devotes specialised resources to supported machine-learning operations.

The important distinction is not simply the number of units. It is how the hardware is organised and how efficiently data can reach those units.

### Worked Example 2.A: A 2 × 2 Systolic Array

A **systolic array** is a grid of processing elements that perform calculations while data moves between neighbouring elements. Such an arrangement can reduce repeated data transfers and support efficient matrix multiplication.

In this example, each processing element holds one weight. The example is simplified to illustrate the arithmetic and data reuse; it is not a cycle-accurate simulation of every commercial systolic array.


### Step 1: Define the Weights and Input

Let the weight matrix be

$$
W =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

and let the input vector be

$$
x =
\begin{bmatrix}
5 \\
6
\end{bmatrix}
$$

We want to calculate

$$
y = Wx
$$

The matrix has two rows and two columns, while the input vector has two elements. Therefore, the result contains two elements.

### Step 2: Calculate the First Output

The first output uses the first row of the matrix:

$$
y_1 = (1 \times 5) + (2 \times 6)
$$

Calculate each multiplication:

$$
1 \times 5 = 5
$$

$$
2 \times 6 = 12
$$

Add the two products:

$$
y_1 = 5 + 12 = 17
$$

Therefore,

$$
\boxed{y_1 = 17}
$$

### Step 3: Calculate the Second Output

The second output uses the second row of the matrix:

$$
y_2 = (3 \times 5) + (4 \times 6)
$$

Calculate each multiplication:

$$
3 \times 5 = 15
$$

$$
4 \times 6 = 24
$$

Add the two products:

$$
y_2 = 15 + 24 = 39
$$

Therefore,

$$
\boxed{y_2 = 39}
$$

### Step 4: Write the Complete Result

Combine both outputs:

$$
y =
\begin{bmatrix}
17 \\
39
\end{bmatrix}
$$

Therefore, the final result is:

$$
\boxed{y = [17, 39]}
$$


## 2.4 What the First TPU Paper Measured

Jouppi et al. (2017) analysed Google's first-generation Tensor Processing Unit using workloads from data-centre applications.

The paper reported several important design characteristics:

- The TPU's matrix multiply unit contained 65,536 8-bit multiply-accumulate units.
- Its peak arithmetic performance was reported as 92 trillion 8-bit operations per second (TOPS), using the paper's stated counting convention.
- It included 28 MiB of software-managed on-chip memory.
- The evaluation compared the TPU with contemporary CPU and GPU systems on selected inference workloads.
- Memory bandwidth limited performance for several of the evaluated applications.

These figures describe the first-generation TPU and the conditions of the original study. They must not be interpreted as a direct ranking of current processors.

### Why the Results Matter

A specialised processor may execute many arithmetic operations quickly, but those operations are useful only when the required data can reach the processing units.

For example, a processor might contain enough arithmetic units to perform a calculation very quickly in theory. If the units spend much of their time waiting for data from memory, actual performance will be lower.

This is one reason that memory bandwidth, data reuse, and data movement are important in accelerator design.

### Historical Context

The first TPU paper was published in 2017 and evaluated hardware from an earlier period. Modern GPUs also include specialised hardware for matrix operations, such as NVIDIA Tensor Cores. The performance difference between a GPU and a TPU therefore depends on the particular products, software, and workload being compared.

A more recent example is Google's Ironwood TPU, announced in April 2025. Google's announcement describes a newer accelerator generation and its high-bandwidth memory capabilities. Because hardware specifications and product availability can change, use the manufacturer's announcement for current claims rather than treating the first TPU paper as a description of today's hardware.

**Source:** Jouppi et al. (2017); Google Cloud (2025).

### Connection to the Roofline Model

The roofline model helps explain why peak arithmetic performance does not always determine real performance. It compares a processor's arithmetic capability with the amount of data that must be transferred from memory.

The model is introduced and calculated in **Section 4**. This section provides the motivation: a large number of MAC units cannot guarantee high performance if memory bandwidth becomes the bottleneck.

---

## 2.5 The NPU: AI on Small Devices

An **NPU (Neural Processing Unit)** is a processor designed to accelerate neural-network operations. NPUs are commonly integrated into phones, tablets, and laptops.

A phone has a limited battery and must manage heat. It therefore cannot use the same power budget as a large data-centre accelerator. An NPU is designed to perform supported AI calculations efficiently within a smaller power budget.

NPUs may use techniques such as:

- Specialised arithmetic units for neural-network operations.
- Low-precision numerical formats where appropriate.
- On-chip memory to reduce some data transfers.
- Hardware and software scheduling designed for efficient inference.

An NPU is often part of a **system-on-a-chip (SoC)**. An SoC integrates several components, such as the CPU, GPU, memory controllers, and other processing blocks, into one chip.

### Examples of Mobile and Personal-Computer AI Hardware

**Table 2.2. Examples of AI acceleration platforms. Names and capabilities depend on the product generation.**

| Platform or company | Example of AI hardware or software platform |
|---|---|
| Apple | Neural Engine, accessed through supported frameworks such as Core ML |
| Google Pixel | Google Tensor SoC with integrated AI-processing capabilities |
| Qualcomm | Hexagon processing technology and related AI acceleration |
| Laptop processors | Integrated NPUs in selected processor generations |

These products are not identical. Their performance depends on supported operations, numerical precision, memory, power limits, software frameworks, and the model being executed.

### When Is an NPU Useful?

Consider a smart speaker that must recognise a wake word and process simple voice commands while using a small amount of power.

An NPU may be a good choice for supported neural-network calculations because it is designed for efficient AI inference. The CPU may still handle application logic, and other processors may be used for tasks that the NPU does not support.

The system can therefore use multiple types of processing hardware rather than relying on a single chip for every task.

### Connection to the Apple Neural Engine Case Study

The Apple Neural Engine is one example of AI acceleration in a personal device. Its role, supported operations, and relationship with Core ML are examined in the Apple Neural Engine case study later in the module.

That case study should use the same learning structure as the other sections: a central idea, learning objectives, sourced technical information, worked reasoning, exercises, and a worked answer key.

---

## 2.6 Software: Why a Chip Is Only as Useful as Its Tools

Hardware performance alone does not determine whether a processor is practical to use. Programmers also need software tools that can run calculations efficiently on the hardware.

These tools include programming interfaces, libraries, compilers, machine-learning frameworks, and runtime systems.

### Common Software Routes

**Table 2.3. Examples of software used to access different processors.**

| Hardware | Examples of software tools | Role |
|---|---|---|
| CPU | General-purpose languages, compilers, and libraries | Executes application logic and general computations |
| GPU | CUDA, OpenCL, PyTorch, and TensorFlow | Supports parallel computation and AI workloads |
| TPU | TensorFlow, JAX, and XLA-based compilation | Compiles supported calculations for TPU execution |
| Apple Neural Engine | Core ML and supported Apple development tools | Deploys compatible models across available processing hardware |

The table lists examples, not an exhaustive list. Actual support depends on the device, software version, model, and operations used.

### What Does a Compiler Do?

A **compiler** translates a program or an intermediate representation into instructions that a processor can execute.

A machine-learning developer usually does not manually control every MAC unit. Instead, the developer creates a model using a framework. The software stack then prepares and executes the model using supported hardware.

The process can be summarised as follows:

```text
Machine-learning model
          |
          v
Framework and operations
          |
          v
Compiler or runtime
          |
          v
Supported processor
          |
          v
Model output
```

**Figure 2.3. Simplified route from a machine-learning model to hardware execution.**

**Alt text:** A flow diagram showing a machine-learning model passing through a framework and operations, then a compiler or runtime, before executing on a supported processor and producing an output.

**Source:** Original conceptual diagram based on the software stacks documented by NVIDIA, Google, and Apple.

### Why Software Compatibility Matters

Suppose a model uses an operation that is not supported efficiently by a particular accelerator. Depending on the software stack, the operation might be executed on another processor, transformed into a different implementation, or rejected.

If some operations run on a CPU while others run on an accelerator, extra data transfers and coordination may reduce performance.

Therefore, a faster chip on paper is not always faster for a particular model.

### The Meaning of a Software Ecosystem

A **software ecosystem** is the collection of tools, libraries, frameworks, documentation, and developer support associated with a platform.

An established ecosystem can make a processor easier to program and integrate into an existing project. Switching platforms may require code changes, testing, and new deployment procedures.

When choosing hardware for an AI project, developers should consider both the chip and the software ecosystem around it.

---

## 2.7 Activities for Section 2

Complete the following activities after reading Sections 2.1–2.6. Try to answer each question before consulting the answer key.

### Activity 2.1: Choose the Processor

For each scenario, choose the most suitable option from CPU, GPU, or TPU. Explain your reasoning in one or two sentences. More than one answer may be reasonable if you state your assumptions.

**A.** A web server runs mixed tasks, including handling requests, checking permissions, and executing application logic.

**B.** A research team wants to train a large image-recognition model and has access to a suitable GPU programming environment.

**C.** A data centre runs a high-volume inference service using a model that is well supported by its available TPU infrastructure.

### Activity 2.2: Systolic Array Worked Example

Use Worked Example 2.A.

Given

\[
W =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\qquad
x =
\begin{bmatrix}
5\\
6
\end{bmatrix},
\]

answer the following questions:

1. Calculate the first output, \(y_1\).
2. Calculate the second output, \(y_2\).
3. How many weight values are there?
4. Under the simplified reuse assumption, how many weight reads and input reads are required?
5. How many reads are counted if each of the four multiply operations fetches its own weight and input?
6. Calculate the percentage reduction in reads when reuse is used.

Show your calculations.

### Activity 2.3: Choose Hardware for a Smart Speaker

A company wants to add a voice assistant to a smart speaker. The device has a small battery and must recognise simple voice commands locally.

Arrange CPU, GPU, and NPU from most suitable to least suitable for the neural-network inference task. Give one reason for your ranking.

Explain why the CPU might still be needed even if an NPU is available.

### Activity 2.4: Why Might a Company Choose a GPU?

The first-generation TPU showed strong performance on the workloads evaluated in the original research. However, a start-up with a different AI workload might still choose a GPU.

Give **two reasons** why a GPU could be a better practical choice.

### Activity 2.5: Understand the Memory Bottleneck

A processor has many arithmetic units, but its memory system cannot deliver data quickly enough to keep them busy.

Answer the following:

1. What is a memory bottleneck?
2. Why might adding more arithmetic units fail to improve performance significantly?
3. Give one way that software or hardware design can reduce unnecessary memory transfers.

### Activity 2.6: Explain the Role of Software

A developer has selected an AI accelerator because its advertised peak performance is high. However, some operations in the developer's model are not efficiently supported by the accelerator.

Explain two ways that this software limitation could affect the real performance or practicality of the project.

### Activity 2.7: Short Reflection

In approximately 80–120 words, explain why selecting hardware for AI requires more than comparing advertised performance figures.

Your answer should mention at least three of the following:

- Type of workload.
- Parallel processing.
- Memory bandwidth and data movement.
- Power consumption.
- Software compatibility.
- Cost and hardware availability.

---

## References

1. Hennessy, J. L., & Patterson, D. A. (2019). A new golden age for computer architecture. *Communications of the ACM, 62*(2), 48–60. https://doi.org/10.1145/3282307

2. Jouppi, N. P., et al. (2017). In-datacenter performance analysis of a Tensor Processing Unit. *Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA)*. https://doi.org/10.1145/3079856.3080246

3. Sze, V., Chen, Y.-H., Yang, T.-J., & Emer, J. S. (2017). Efficient processing of deep neural networks: A tutorial and survey. *Proceedings of the IEEE, 105*(12), 2295–2329. https://doi.org/10.1109/JPROC.2017.2761740

4. NVIDIA. (n.d.). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

5. Google Cloud. (2025). *Introducing Ironwood, Google's seventh-generation TPU*. Google Cloud Blog. https://cloud.google.com/blog/

6. Apple. (n.d.). *Core ML documentation*. Apple Developer. https://developer.apple.com/documentation/coreml

> **Source note:** The references above support the general architectural and software concepts in this section. For product-specific performance, memory capacity, and availability claims, consult the relevant manufacturer's documentation for the exact product generation. Do not use a general company homepage as the sole source for a specific numerical specification.

---

## Answer Key

### Activity 2.1: Choose the Processor

**A. CPU**

A CPU is a reasonable choice because the server handles varied tasks, including decisions and application logic. A GPU or accelerator may still be used for particular workloads within the server.

**B. GPU**

A GPU is a reasonable choice because image-model training often involves large numbers of parallel mathematical operations. The best choice still depends on the model, available memory, software support, and budget.

**C. TPU**

A TPU is a reasonable choice because the scenario specifies a high-volume inference workload that is supported by the available TPU infrastructure. The conclusion depends on the workload being compatible and the overall deployment meeting the service's cost and performance requirements.

### Activity 2.2: Systolic Array Worked Example

**1. First output**

\[
y_1 = (1 \times 5) + (2 \times 6)
\]

\[
y_1 = 5 + 12 = \boxed{17}
\]

**2. Second output**

\[
y_2 = (3 \times 5) + (4 \times 6)
\]

\[
y_2 = 15 + 24 = \boxed{39}
\]

Therefore,

\[
\boxed{y = [17,39]}
\]

**3. Number of weight values**

The matrix has two rows and two columns:

\[
2 \times 2 = \boxed{4\text{ weights}}
\]

**4. Reads with reuse**

Under the simplified assumption, each weight is read once and each input is read once.

\[
4 + 2 = \boxed{6\text{ reads}}
\]

This count excludes partial sums and other transfers within the hardware.

**5. Reads without reuse**

Each of the four multiply operations fetches one weight and one input.

\[
4 \times 2 = \boxed{8\text{ reads}}
\]

**6. Percentage reduction**

First, find the difference:

\[
8 - 6 = 2
\]

Then divide the reduction by the original count:

\[
\frac{2}{8}\times100\% = \boxed{25\%}
\]

The simplified model reduces the number of counted reads from eight to six, or 25%.

### Activity 2.3: Choose Hardware for a Smart Speaker

A reasonable ranking for the neural-network inference task is:

1. **NPU:** If it supports the model, it may perform inference efficiently within a small power budget.
2. **CPU:** It can run some models and handle general application logic, but may be less efficient for supported neural-network operations.
3. **GPU:** It can execute parallel calculations, but its power and resource requirements may be less suitable for this particular device.

This ranking is not universal. Actual results depend on the hardware, model, software support, and power limits.

The CPU may still be needed to manage the application, handle user input, coordinate tasks, and execute operations that the NPU does not support.

### Activity 2.4: Why Might a Company Choose a GPU?

Possible answers include:

- **Software compatibility:** The company may already use a GPU-based framework or have code optimised for its GPU.
- **Workload flexibility:** The model may contain operations that are not well supported by the available TPU.
- **Availability:** The team may already have access to suitable GPUs, while TPU access may be limited.
- **Cost:** The GPU may be more economical for the team's scale or deployment requirements.

Any two well-explained reasons are acceptable.

### Activity 2.5: Understand the Memory Bottleneck

**1. Definition**

A memory bottleneck occurs when the processor cannot receive data quickly enough to keep its processing units busy.

**2. Why more arithmetic units may not help**

If the processor is waiting for data, adding more arithmetic units does not necessarily allow more calculations to finish. The new units may also remain idle while waiting for data.

**3. One possible solution**

Data reuse can reduce repeated memory accesses. For example, a processing element may reuse a weight instead of fetching the same value repeatedly. Other possible approaches include improving memory bandwidth or changing the data layout.

### Activity 2.6: Explain the Role of Software

Two possible effects are:

- **Reduced performance:** Unsupported or inefficiently supported operations may run on another processor or require extra data transfers, increasing execution time.
- **Reduced practicality:** The developer may need to change the model, rewrite parts of the software, or use a different framework. This can increase development time and maintenance costs.

The advertised peak performance does not guarantee that the full model will achieve that performance.

### Activity 2.7: Short Reflection

**Sample answer (102 words):**

Choosing AI hardware requires more than comparing advertised performance figures because different processors are designed for different workloads. A GPU can perform many calculations in parallel, while a CPU is flexible for general application tasks. A specialised accelerator may be efficient for supported neural-network operations, but its performance can be limited by memory bandwidth and data movement. Power consumption is also important for devices such as phones and smart speakers. In addition, software compatibility determines whether a model can use the accelerator efficiently. Cost and hardware availability affect whether a solution is practical. Therefore, hardware should be selected based on the complete workload and deployment requirements.

---

## Section 2 Quality Checklist

- [x] Sections are numbered continuously from 2.1 to 2.7.
- [x] Worked Example 2.A is included in the Markdown and contains the calculations, dataflow explanation, code, and expected output.
- [x] The systolic-array activity refers to Worked Example 2.A.
- [x] Figures have numbers, captions, alt text, and source notes.
- [x] Historical TPU results are distinguished from modern hardware.
- [x] Software tools and hardware compatibility are explained.
- [x] Activities 2.1–2.7 are included.
- [x] The answer key includes worked calculations and explanations.
- [x] References are provided for the main architectural and software claims.
- [ ] Before submission, verify every product-specific claim against the exact primary-source document and replace any broad or incomplete source link with the specific supporting page.
