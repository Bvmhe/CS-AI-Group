1. Fix the “Specialised units” row.
“Circuits that do small matrix multiplications as a single hardware step” is right for Tensor Cores, but Apple’s Neural Engine is a different kind of block: a dedicated accelerator for whole neural-network layers, not a small matrix unit inside a GPU. Naming both as the same thing is slightly misleading. Either give only Tensor Cores as the example or say “e.g. NVIDIA Tensor Cores; dedicated NPUs such as Apple’s Neural Engine”.

2. Sparsity needs a caveat.
“Skip multiplications by zero” sounds free, but dense hardware gains nothing from zeros unless it is designed to skip them. NVIDIA’s 2:4 structured sparsity is the usual example. Add a note like “needs hardware or a data format that supports it; unstructured zeros are hard to exploit”.

3. Batching is under-explained.
“Fetch weights once, use them many times” is the right idea. It also has a trade-off: larger batches raise throughput but also latency, which matters for real-time inference. One clause on that would help.

4. The “Why it helps” for quantisation is a bit vague.
“More can be processed per cycle” is true, but the main benefits are less memory and bandwidth, plus smaller and more energy-efficient MAC units. The paragraph below says this better, so just align the table with it. The paragraph also should mention the cost: lower precision can reduce accuracy, which is why techniques like calibration or quantisation-aware training exist.

5. Format table: be careful with the groupings.

“BF16 / FP16” on one row hides a real difference. BF16 keeps FP32’s range with fewer mantissa bits, while FP16 has a smaller range. Your paragraph describes bfloat16 as “the same range as FP32”, which is correct, but that is not true of FP16, so the row should not blur them. Split them or add a note.
The table cuts off in the screenshot, so I can’t check the last rows. If you list INT4 or others, check that each “typical use” is accurate.
“FP8 … newer training/inference hardware” is fine. Mentioning that FP8 has two variants (E4M3 and E5M2) would add precision if you want it.

6. Citation and wording.

“Jouppi et al., 2017” is the right reference for the first TPU, which used 8-bit integers for inference. Make sure it appears in your reference list.
“Quantisation” in the table versus “reduced precision” in the heading: pick one primary term and define the other once.
Several rows say “(Section 4)” or reference systolic arrays without a link. A clickable anchor would help a reader on GitHub.

7. Minor consistency points.

“Weight stationary / data reuse” lumps two ideas together. Weight-stationary is one specific dataflow, and data reuse is the general goal. A small rewording like “Dataflow design (e.g. weight-stationary)” would be more accurate.
“Operator fusion” is well explained. Adding “common in compilers and frameworks (e.g. kernel fusion)” would connect it to software, since fusion is as much a compiler technique as a hardware one.
