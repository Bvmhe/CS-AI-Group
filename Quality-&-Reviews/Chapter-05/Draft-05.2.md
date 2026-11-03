# Section 5: Case Study — Google TPU Infrastructure

> **Big idea.** Google's TPU infrastructure is this module in one real system: a chip designed around MACs and on-chip memory (Module 01), joined by a purpose-built network into pods of thousands of chips (Section 2), programmed through a compiler (Section 1), scheduled as a shared resource (Section 3), and designed with energy, cost and failure in mind (Section 4).

---

## Learning Objectives

By the end of this section, you should be able to:

1. Describe the layers of a TPU system from chip to pod to multi-pod training job.
2. Explain what a torus network is, and calculate how wraparound links shorten the longest path.
3. Explain how optical circuit switching improves availability and flexibility.
4. Explain why hierarchical communication saves slow-network traffic, with a calculation.
5. Evaluate the strengths and limitations of TPU infrastructure for a given team.

> **A note on specifications.** TPU generations, chip counts and figures change quickly. This section teaches the *design ideas*, which are stable, and uses numbers from the cited papers where given. Before quoting any current specification, check Google Cloud's TPU documentation and the Google announcement for that generation. Calculations marked *illustrative* use round numbers chosen for teaching.

---

## 5.1 A Short History: From One Chip to a System

**Table 5.1.** Selected milestones in TPU design. Verify current generations before citing.

| Generation | Main idea | Source |
|---|---|---|
| First TPU (deployed 2015) | Inference-only chip with an 8-bit matrix unit, 92 TOPS peak, 28 MiB on-chip buffer | Jouppi et al. (2017); Module 01, Section 2.5 |
| Next generations (from 2017) | Added training: floating-point (bfloat16) arithmetic, HBM, and chips wired together into pods | Google Cloud TPU documentation |
| TPU v4 | Pods of 4,096 chips joined by optically reconfigurable links; specialised units for embedding lookups | Jouppi et al. (2023) |
| Later generations | Variants for cost-efficient serving and for the largest training runs; the newest announced generation (Ironwood, 2025) is aimed at inference at scale | Google (2025); Google Cloud TPU documentation |

The pattern is the one from Section 4: each generation moved attention from *the chip* to *the system around the chip*, because once a chip is fast, the limits become memory, communication, reliability and energy.

---

## 5.2 Building Blocks: From Chip to Pod

Using the TPU v4 description in Jouppi et al. (2023) as a concrete example.

**Table 5.2.** Levels of a TPU system (v4 example).

| Level | What it is | Why it exists |
|---|---|---|
| **Chip** | Matrix-multiply units, a vector unit, on-chip memory, HBM, and inter-chip network ports | Executes MACs efficiently (Module 01) |
| **Tray / host** | A small group of chips (four, in v4) plus a host computer that loads data and runs the program | Chips are accelerators and need a CPU host |
| **Rack / cube** | 64 chips arranged as a 4 × 4 × 4 cube | A unit of wiring and deployment |
| **Pod** | 64 cubes, 4,096 chips, linked into one system | The largest tightly coupled unit |
| **Slice** | The subset of a pod a job is given (for example 8 × 8 × 8 chips) | Lets many jobs share a pod |
| **Multi-slice job** | Several slices, possibly in different pods, linked by the data-centre network | Training beyond one pod |

**Figure 5.1.** The hierarchy from chip to multi-slice job. *Source: author's own diagram, based on Jouppi et al. (2023).*

```text
   chip --> tray (4 chips + host) --> cube (64 chips, 4x4x4) --> pod (64 cubes = 4,096 chips)
                                                                   |
                                      many pods linked by data-centre network --> multi-slice job
```

---

## 5.3 The Interconnect: A 3D Torus with Optical Switching

In Section 2 you saw that distributed training is limited by communication. TPU pods address this by connecting each chip **directly to its neighbours** over dedicated **inter-chip interconnect (ICI)** links, arranged as a three-dimensional grid. Each chip has six neighbours (up, down, left, right, front, back). In a **torus**, the links *wrap around*: the chip at one edge of the grid is linked to the chip at the opposite edge.

Why is this good for AI?

- Collective operations such as all-reduce can be built from neighbour-to-neighbour transfers (the ring idea from Section 2.3), which use short, fast cables.
- There is no single central switch that every message must cross, which would become the bottleneck.
- Wraparound links shorten the longest path between any two chips.

### Worked Example 5.A: How Much Do Wraparound Links Help?

In a grid with *side* chips along each dimension, the longest path (the **diameter**) is measured in hops.

- **Without wraparound (mesh):** the longest distance along one dimension is *side* − 1, so the diameter is 3 × (*side* − 1).
- **With wraparound (torus):** the longest distance along one dimension is about *side* ÷ 2, so the diameter is 3 × (*side* ÷ 2).

For a 4 × 4 × 4 cube (64 chips): mesh = 9 hops, torus = **6 hops**. For a 16 × 16 × 16 pod (4,096 chips): mesh = 45 hops, torus = **24 hops**.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
def diameter(side, wrap):
    per_dim = side // 2 if wrap else side - 1
    return 3 * per_dim

for side in (4, 16):
    print(side ** 3, diameter(side, True), diameter(side, False))
```

Expected output:

```text
64 6 9
4096 24 45
```

### Optical Circuit Switches (OCS)

TPU v4 does not wire cubes to each other with fixed cables. Instead, the links between cubes pass through **optical circuit switches**: devices that steer light with tiny mirrors, so that the connections between cubes can be *re-patched by software* without anyone touching a cable (Jouppi et al., 2023). This gives three benefits:

1. **Availability.** If a chip, tray or cube fails, the switches can route around it and assemble the requested slice from healthy cubes.
2. **Flexible slice shapes.** A job can ask for the shape that suits its parallelism (for example, a long thin slice or a cube), and the switches build it.
3. **Incremental deployment.** A pod can start working before every cube has arrived, and be expanded.

> **Bridge to Networks (Module 02).** This is the same trade-off studied in Module 02: dedicated, direct links give high bandwidth and low latency for predictable traffic, and switched or reconfigurable networks give flexibility. TPU pods choose the former for the training traffic and use the latter (OCS and the data-centre network) to manage shape, failures and scale.

---

## 5.4 Software: Compilers, Sharding and Multi-Slice Training

The hardware would be unusable without software that hides its structure.

- **XLA** (Section 1.2) compiles the model graph for the TPU, fusing operations and assigning layouts.
- **Frameworks** such as JAX, TensorFlow and PyTorch/XLA let programmers write models in a familiar way.
- **Sharding annotations.** The programmer states how each tensor should be split across the chips (for example, split this matrix by rows across a group of chips). The compiler then inserts the required communication (all-reduce, all-gather) automatically. This is the Section 2 parallelism strategies expressed as a few hints rather than hand-written messages.
- **Multi-slice training** connects slices over the data-centre network. Communication inside a slice uses the fast ICI; between slices it uses the slower network.

### Worked Example 5.B: Why Communicate Hierarchically?

*Illustrative numbers.* Two slices of 256 chips each train the 7B model. Gradients are 14 GB. The slow network gives each chip 12.5 GB/s between slices.

- **Naive:** every chip sends its full 14 GB across the slow network: 14 ÷ 12.5 ≈ **1.12 s**.
- **Hierarchical:** first, inside each slice (fast ICI), a *reduce-scatter* leaves each of the 256 chips holding just 14 ÷ 256 ≈ 0.055 GB of the summed gradient. Only these small pieces cross the slow network: 0.055 ÷ 12.5 ≈ **4.4 ms**.
- The slow-network traffic and time shrink by a factor of **256**, the number of chips in a slice.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
grad_gb = 14
slow_bw = 12.5                          # GB/s per chip between slices
naive = grad_gb / slow_bw
hier = grad_gb / 256 / slow_bw
print(round(naive, 2))
print(round(hier * 1e3, 1))
print(round(naive / hier))
```

Expected output:

```text
1.12
4.4
256
```

The general principle: *do as much reduction as possible on the fastest links, and send only the minimum across the slowest.* This is the Module 01 memory-hierarchy idea (keep data near the compute) applied to a network hierarchy.

---

## 5.5 Operating the Fleet: Scheduling, Reliability, Energy

- **Scheduling.** Pods are shared. A cluster scheduler (Section 3.4) allocates slices to jobs by quota and priority, places them so their chips are neighbours, and starts a training job only when its whole slice is available (gang scheduling). With OCS, the "shape" problem is easier: any set of healthy cubes can be wired into the requested slice, which reduces fragmentation and raises utilisation.
- **Reliability.** Section 4.3 showed that thousands of chips fail every few hours. TPU systems combine health monitoring, optical rerouting around bad hardware, and frequent checkpointing, so that a failure costs minutes rather than days.
- **Energy.** A custom chip spends silicon on MACs and on-chip memory instead of general-purpose control logic (Module 01, Section 2.3). At the facility level, Google reports a fleet-wide PUE close to 1.1 for its data centres, and recent TPU generations use liquid cooling to remove heat from densely packed chips (Google, n.d.; check the latest figures). The TPU v4 paper reports energy-efficiency comparisons with other accelerators; consult it for the exact numbers and conditions (Jouppi et al., 2023).

---

## 5.6 Evaluation: Strengths and Limitations

**Table 5.3.** Strengths and limitations of TPU infrastructure.

| Strengths | Limitations |
|---|---|
| High performance per watt and per dollar on large matrix workloads | Less flexible than a GPU for unusual operations; unsupported operations fall back to slower paths |
| Co-design of chip, network, compiler and scheduler lets the whole system be tuned together | Tied to a particular software stack (XLA/JAX/PyTorch-XLA); porting from a CUDA codebase takes effort |
| Direct-link torus network scales training to thousands of chips | Mostly available through Google's own cloud, which creates dependence on one provider |
| Optical switching raises availability and utilisation | Added complexity and hardware that other architectures do not need |
| Generations tailored for training or for serving | Peak numbers do not equal real performance; each team must measure its own workload |

> **Judge with your own numbers.** The way to decide between platforms is the way this module taught: measure MFU and cost per useful token on *your* workload, and include porting effort. A chip that is faster on paper may not be cheaper in practice.

---

## 5.7 Case-Study Questions

1. Why is a 3D torus a good match for all-reduce and tensor-parallel communication?
2. Explain in three sentences how optical circuit switches improve the availability of a pod.
3. Calculate the diameter, in hops, of an 8 × 8 × 8 mesh and an 8 × 8 × 8 torus. By what percentage does wraparound shorten the longest path?
4. In Worked Example 5.B, why does hierarchical communication cut slow-network time by exactly the number of chips per slice?
5. Name three ideas from Module 01 that appear in the TPU system, and say where.
6. A start-up has a PyTorch model that uses a custom GPU kernel. List two things it must check before moving to TPUs.

### Answer Key for Case-Study Questions

**Question 1.** A 3D torus gives each chip six direct, short, dedicated links to its neighbours. Collective operations such as all-reduce and tensor-parallel shuffles are built from neighbour-to-neighbour transfers (the ring all-reduce of Section 2.3), so the communication stays on fast, low-latency cables instead of passing through a central switch. Wraparound links reduce the longest path, so the diameter of the network shrinks and the collective completes in fewer hops.

**Question 2.** Optical circuit switches steer light with tiny mirrors, so the connections between cubes can be re-patched by software without touching a cable. If a chip, tray or cube fails, the switches route around it and assemble the requested slice from the remaining healthy cubes. Because the slice can be rebuilt from any set of healthy cubes, a single failure no longer takes down a large fraction of the pod; the job restarts on a fresh slice in minutes rather than waiting for a hardware repair.

**Question 3.**
- 8 × 8 × 8 mesh: diameter = 3 × (8 − 1) = **21 hops**.
- 8 × 8 × 8 torus: diameter = 3 × (8 ÷ 2) = **12 hops**.
- Percentage shortened = (21 − 12) ÷ 21 ≈ **42.9%**.

**Question 4.** In the hierarchical scheme, each slice first performs a *reduce-scatter* on its fast ICI links, so that after the local reduction each of the 256 chips holds only 1/256 of the summed gradient. Only these 1/256-sized pieces cross the slow network. Because both the naive traffic (the full 14 GB per chip) and the hierarchical traffic (14 ÷ 256 GB per chip) are compared on the same slow link at 12.5 GB/s, the time ratio is exactly 256, the number of chips per slice.

**Question 5.** Three ideas from Module 01 that appear in the TPU system:
- **Systolic MAC arrays and INT8/bfloat16 arithmetic**: the matrix-multiply unit inside each TPU chip.
- **On-chip memory keeps data close to compute**: the large on-chip buffer (28 MiB in the first TPU) and the HBM hierarchy on later chips.
- **Custom chip design spends silicon on the workload**: a TPU devotes silicon to MACs and on-chip memory instead of general-purpose control logic, improving energy efficiency.

**Question 6.** Two things the start-up must check:
1. **Whether the custom kernel is supported** by XLA/PyTorch-XLA on the TPU, or whether it will fall back to a slower path (Section 1.4). If it is unsupported, the model may run far slower than its peak numbers suggest.
2. **How much porting effort is required** to express the model with sharding annotations and TPU-compatible operations, and whether the team has (or can hire) the skills to operate a TPU stack. Porting cost and ecosystem lock-in (Section 5.6) often outweigh a raw speed advantage.

---

## 5.8 Conclusion

The lesson of the TPU case study is that **AI performance is a property of systems, not chips**. A chip's peak number matters only if the compiler keeps it busy, the network keeps it fed, the scheduler keeps it shared, the cooling keeps it running, and the budget keeps it paid for. Each of those layers is a place to win or lose, and measuring each is the only reliable way to know which.

---

## 5.9 Summary

- A TPU system is a hierarchy: chip, tray, cube, pod (4,096 chips in v4) and multi-slice jobs across pods.
- Chips are joined by direct links in a 3D torus; wraparound shortens the longest path in a 16 × 16 × 16 pod from 45 to 24 hops.
- Optical circuit switches let software re-patch connections, which improves availability, slice-shape flexibility and incremental deployment.
- XLA and sharding annotations turn the parallelism strategies of Section 2 into compiler-generated communication; hierarchical collectives reduce slow-network traffic by the slice size (256× in the example).
- Scheduling, health monitoring, checkpointing and efficient cooling are as important as the chip itself.
- TPUs offer efficiency and scale at the cost of flexibility and ecosystem dependence; measure your own workload before choosing.

---

## References

Google. (n.d.). *Google Cloud TPU documentation*. Google Cloud.

Google. (2025). *Ironwood: The next generation of TPUs for AI inference at scale*. Google Cloud Blog.

Jouppi, N. P., Young, C., Patil, N., Patterson, D., Agrawal, G., Bajwa, R., ... & Yoon, D. H. (2017). In-datacenter performance analysis of a tensor processing unit. *Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA)*, 1–12. https://doi.org/10.1145/3079856.3080246

Jouppi, N. P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., ... & Patterson, D. (2023). TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. *Proceedings of the 50th Annual International Symposium on Computer Architecture (ISCA)*, 1–14. https://doi.org/10.1145/3579371.3589350
