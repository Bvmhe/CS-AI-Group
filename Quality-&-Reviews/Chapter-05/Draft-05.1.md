# Section 5: Case Study — Google TPU Infrastructure

> [!IMPORTANT]
> **Big idea.** Google's TPU infrastructure is this module in one real system: a chip designed around MACs and on-chip memory (Module 01), joined by a purpose-built network into pods of thousands of chips (Section 2), programmed through a compiler (Section 1), scheduled as a shared resource (Section 3), and designed with energy, cost and failure in mind (Section 4).

## Learning objectives

By the end of this section you should be able to:

- Describe the layers of a TPU system from chip to pod to multi-pod training job.
- Explain what a torus network is, and calculate how wraparound links shorten the longest path.
- Explain how optical circuit switching improves availability and flexibility.
- Explain why hierarchical communication saves slow-network traffic, with a calculation.
- Evaluate the strengths and limitations of TPU infrastructure for a given team.

> [!WARNING]
> **A note on specifications.** TPU generations, chip counts and figures change quickly. This section teaches the *design ideas*, which are stable, and uses numbers from the cited papers where given. Before quoting any current specification, check Google Cloud's TPU documentation and the Google announcement for that generation. Calculations marked *illustrative* use round numbers chosen for teaching.

## 5.1 A short history: from one chip to a system

| Generation | Main idea | Source |
|---|---|---|
| First TPU (deployed 2015) | Inference-only chip with an 8-bit matrix unit, 92 TOPS peak, 28 MiB on-chip buffer | Jouppi et al. (2017); Module 01, Section 2.5 |
| Next generations (from 2017) | Added training: floating-point (bfloat16) arithmetic, HBM, and chips wired together into pods | Google Cloud TPU documentation |
| TPU v4 | Pods of 4,096 chips joined by optically reconfigurable links; specialised units for embedding lookups | Jouppi et al. (2023) |
| Later generations | Variants for cost-efficient serving and for the largest training runs; the newest announced generation (Ironwood, 2025) is aimed at inference at scale | Google (2025); Google Cloud TPU documentation |

*Table 5.1. Selected milestones in TPU design. Verify current generations before citing.*

The pattern is the one from Section 4: each generation moved attention from *the chip* to *the system around the chip*, because once a chip is fast, the limits become memory, communication, reliability and energy.

## 5.2 Building blocks: from chip to pod

Using the TPU v4 description in Jouppi et al. (2023) as a concrete example:

| Level | What it is | Why it exists |
|---|---|---|
| **Chip** | Matrix-multiply units, a vector unit, on-chip memory, HBM, and inter-chip network ports | Executes MACs efficiently (Module 01) |
| **Tray / host** | A small group of chips (four, in v4) plus a host computer that loads data and runs the program | Chips are accelerators and need a CPU host |
| **Rack / cube** | 64 chips arranged as a 4 × 4 × 4 cube | A unit of wiring and deployment |
| **Pod** | 64 cubes, 4,096 chips, linked into one system | The largest tightly coupled unit |
| **Slice** | The subset of a pod a job is given (for example 8 × 8 × 8 chips) | Lets many jobs share a pod |
| **Multi-slice job** | Several slices, possibly in different pods, linked by the data-centre network | Training beyond one pod |

*Table 5.2. Levels of a TPU system (v4 example).*

```
   chip --> tray (4 chips + host) --> cube (64 chips, 4x4x4) --> pod (64 cubes = 4,096 chips)
                                                                   |
                                      many pods linked by data-centre network --> multi-slice job
```

*Figure 5.1. The hierarchy from chip to multi-slice job. Source: author's own diagram, based on Jouppi et al. (2023).*

## 5.3 The interconnect: a 3D torus with optical switching

In Section 2 you saw that distributed training is limited by communication. TPU pods address this by connecting each chip **directly to its neighbours** over dedicated **inter-chip interconnect (ICI)** links, arranged as a three-dimensional grid. Each chip has six neighbours (up, down, left, right, front, back). In a **torus**, the links *wrap around*: the chip at one edge of the grid is linked to the chip at the opposite edge.

Why is this good for AI?

- Collective operations such as all-reduce can be built from neighbour-to-neighbour transfers (the ring idea from Section 2.3), which use short, fast cables.
- There is no single central switch that every message must cross, which would become the bottleneck.
- Wraparound links shorten the longest path between any two chips.

### Worked example 5.A: how much do wraparound links help?

In a grid with *side* chips along each dimension, the longest path (the **diameter**) is measured in hops.

- **Without wraparound (mesh):** the longest distance along one dimension is *side* − 1, so the diameter is 3 × (*side* − 1).
- **With wraparound (torus):** the longest distance along one dimension is about *side* ÷ 2, so the diameter is 3 × (*side* ÷ 2).

For a 4 × 4 × 4 cube (64 chips): mesh = 9 hops, torus = **6 hops**. For a 16 × 16 × 16 pod (4,096 chips): mesh = 45 hops, torus = **24 hops**.

```python
def diameter(side, wrap):
    per_dim = side // 2 if wrap else side - 1
    return 3 * per_dim

for side in (4, 16):
    print(side ** 3, diameter(side, True), diameter(side, False))
```

Expected output:

```
64 6 9
4096 24 45
```

### Optical circuit switches (OCS)

TPU v4 does not wire cubes to each other with fixed cables. Instead, the links between cubes pass through **optical circuit switches**: devices that steer light with tiny mirrors, so that the connections between cubes can be *re-patched by software* without anyone touching a cable (Jouppi et al., 2023). This gives three benefits:

1. **Availability.** If a chip, tray or cube fails, the switches can route around it and assemble the requested slice from healthy cubes.
2. **Flexible slice shapes.** A job can ask for the shape that suits its parallelism (for example, a long thin slice or a cube), and the switches build it.
3. **Incremental deployment.** A pod can start working before every cube has arrived, and be expanded.

> [!NOTE]
> **Bridge to Networks (Module 02).** This is the same trade-off studied in Module 02: dedicated, direct links give high bandwidth and low latency for predictable traffic, and switched or reconfigurable networks give flexibility. TPU pods choose the former for the training traffic and use the latter (OCS and the data-centre network) to manage shape, failures and scale.

## 5.4 Software: compilers, sharding and multi-slice training

The hardware would be unusable without software that hides its structure.

- **XLA** (Section 1.2) compiles the model graph for the TPU, fusing operations and assigning layouts.
- **Frameworks** such as JAX, TensorFlow and PyTorch/XLA let programmers write models in a familiar way.
- **Sharding annotations.** The programmer states how each tensor should be split across the chips (for example, split this matrix by rows across a group of chips). The compiler then inserts the required communication (all-reduce, all-gather) automatically. This is the Section 2 parallelism strategies expressed as a few hints rather than hand-written messages.
- **Multi-slice training** connects slices over the data-centre network. Communication inside a slice uses the fast ICI; between slices it uses the slower network.

### Worked example 5.B: why communicate hierarchically?

*Illustrative numbers.* Two slices of 256 chips each train the 7B model. Gradients are 14 GB. The slow network gives each chip 12.5 GB/s between slices.

- **Naive:** every chip sends its full 14 GB across the slow network: 14 ÷ 12.5 ≈ **1.12 s**.
- **Hierarchical:** first, inside each slice (fast ICI), a *reduce-scatter* leaves each of the 256 chips holding just 14 ÷ 256 ≈ 0.055 GB of the summed gradient. Only these small pieces cross the slow network: 0.055 ÷ 12.5 ≈ **4.4 ms**.
- The slow-network traffic and time shrink by a factor of **256**, the number of chips in a slice.

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

```
1.12
4.4
256
```

The general principle: *do as much reduction as possible on the fastest links, and send only the minimum across the slowest.* This is the Module 01 memory-hierarchy idea (keep data near the compute) applied to a network hierarchy.

## 5.5 Operating the fleet: scheduling, reliability, energy

- **Scheduling.** Pods are shared. A cluster scheduler (Section 3.4) allocates slices to jobs by quota and priority, places them so their chips are neighbours, and starts a training job only when its whole slice is available (gang scheduling). With OCS, the "shape" problem is easier: any set of healthy cubes can be wired into the requested slice, which reduces fragmentation and raises utilisation.
- **Reliability.** Section 4.3 showed that thousands of chips fail every few hours. TPU systems combine health monitoring, optical rerouting around bad hardware, and frequent checkpointing, so that a failure costs minutes rather than days.
- **Energy.** A custom chip spends silicon on MACs and on-chip memory instead of general-purpose control logic (Module 01, Section 2.3). At the facility level, Google reports a fleet-wide PUE close to 1.1 for its data centres, and recent TPU generations use liquid cooling to remove heat from densely packed chips (Google, n.d.; check the latest figures). The TPU v4 paper reports energy-efficiency comparisons with other accelerators; consult it for the exact numbers and conditions (Jouppi et al., 2023).

## 5.6 Evaluation: strengths and limitations

| Strengths | Limitations |
|---|---|
| High performance per watt and per dollar on large matrix workloads | Less flexible than a GPU for unusual operations; unsupported operations fall back to slower paths |
| Co-design of chip, network, compiler and scheduler lets the whole system be tuned together | Tied to a particular software stack (XLA/JAX/PyTorch-XLA); porting from a CUDA codebase takes effort |
| Direct-link torus network scales training to thousands of chips | Mostly available through Google's own cloud, which creates dependence on one provider |
| Optical switching raises availability and utilisation | Added complexity and hardware that other architectures do not need |
| Generations tailored for training or for serving | Peak numbers do not equal real performance; each team must measure its own workload |

*Table 5.3. Strengths and limitations of TPU infrastructure.*

> [!TIP]
> **Judge with your own numbers.** The way to decide between platforms is the way this module taught: measure MFU and cost per useful token on *your* workload, and include porting effort. A chip that is faster on paper may not be cheaper in practice.

## 5.7 Case-study questions

1. Why is a 3D torus a good match for all-reduce and tensor-parallel communication?
2. Explain in three sentences how optical circuit switches improve the availability of a pod.
3. Calculate the diameter, in hops, of an 8 × 8 × 8 mesh and an 8 × 8 × 8 torus. By what percentage does wraparound shorten the longest path?
4. In Worked example 5.B, why does hierarchical communication cut slow-network time by exactly the number of chips per slice?
5. Name three ideas from Module 01 that appear in the TPU system, and say where.
6. A start-up has a PyTorch model that uses a custom GPU kernel. List two things it must check before moving to TPUs.

## 5.8 Conclusion

The lesson of the TPU case study is that **AI performance is a property of systems, not chips**. A chip's peak number matters only if the compiler keeps it busy, the network keeps it fed, the scheduler keeps it shared, the cooling keeps it running, and the budget keeps it paid for. Each of those layers is a place to win or lose, and measuring each is the only reliable way to know which.

## 5.9 Summary

- A TPU system is a hierarchy: chip, tray, cube, pod (4,096 chips in v4) and multi-slice jobs across pods.
- Chips are joined by direct links in a 3D torus; wraparound shortens the longest path in a 16 × 16 × 16 pod from 45 to 24 hops.
- Optical circuit switches let software re-patch connections, which improves availability, slice-shape flexibility and incremental deployment.
- XLA and sharding annotations turn the parallelism strategies of Section 2 into compiler-generated communication; hierarchical collectives reduce slow-network traffic by the slice size (256× in the example).
- Scheduling, health monitoring, checkpointing and efficient cooling are as important as the chip itself.
- TPUs offer efficiency and scale at the cost of flexibility and ecosystem dependence; measure your own workload before choosing.

---
