1. “Data-level” means two different things.
In the table, data-level is SIMD/SIMT inside one chip. The last row says “model- and data-level across chips”, which is a different idea: replicating the model and giving each chip a different batch. The “Data parallelism” definition (1,000 pixels × the same filter) only fits the first meaning. A reader will likely wonder whether these are the same thing. Consider renaming the last row “Distributed (multi-chip)” and mentioning data parallelism and model parallelism as its two flavors.

2. “Model (task) parallelism” blends two concepts.
Task parallelism means different operations running at the same time. Model parallelism means splitting one model across devices. Your example (layers 1-10 on one chip, 11-20 on another) is specifically pipeline-style model parallelism. Pick one term, or define them separately.

3. The latency-hiding and CPU-vs-GPU content is in the wrong bullet.
It’s stuck onto the end of the model parallelism bullet, though it has nothing to do with it. It’s one of the best parts of the passage, so give it its own bullet or a short paragraph titled “Why GPUs hide latency and CPUs don’t.”

4. Some terms are used but never defined here.

MAC (multiply-accumulate) and systolic
Lock-step, and what a warp is (a group of 32 threads on NVIDIA)
TPU

If they’re defined earlier, a “see Section X” pointer is enough. If not, add a one-line gloss.

5. A small accuracy point on SIMT.
“Lock-step” is a good first approximation, but it’s worth a sentence on divergence: if threads in a warp take different branches, the warp runs both paths one after another, which wastes speed.

6. Missing “why should I care?”
One sentence tying this to AI would help: neural networks are mostly big matrix multiplications, which are ideal for all these kinds of parallelism.

A cleaner version of the confusing bullet:

Model parallelism: one big model is split across several chips (for example, one chip handles layers 1-10 and another handles layers 11-20). Data parallelism across chips is the alternative: every chip holds a full copy of the model and processes a different slice of the batch.
