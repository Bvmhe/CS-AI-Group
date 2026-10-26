What I verified

1,024 × 4,096 = 4,194,304 MACs, and 2× that = 8,388,608 FLOPs.
4,194,304 ÷ 3×10⁹ ≈ 1.4 ms, and 4,194,304 ÷ 4.096×10¹² ≈ 1.0 µs.
The speedup is ≈ 1,365×, which is exactly 4,096 ÷ 3.

Suggested improvements

Step 4 gives two numbers for one result. “About 1,365× faster (roughly 1,400×)” looks like a hedge or a mistake. Keep one, and consider adding why it comes out that way: 4,096 units × 1 GHz vs 1 unit × 3 GHz = 4,096/3 ≈ 1,365.
Step 2 is never used. The FLOPs conversion doesn’t feed into Step 3, which works entirely in MACs. Either say why it’s there (“vendors quote FLOPS, so this lets you compare with spec sheets”) or compute the times in FLOPs.
The CPU assumption contradicts Section 3.1. The previous section teaches that CPUs have vector (SIMD) units, but here a CPU core does “one MAC per cycle”. Label it “scalar, no SIMD” so it doesn’t read as a strawman, and note that a vectorised core would be much closer.
Step 1 wording is slightly off. “1,024 multiplications and additions” is cleaner as “1,024 MACs”, since that’s the term you’re introducing. It would also help to mention that the bias is ignored.
Cycles may be more intuitive than seconds. The accelerator needs 4,194,304 ÷ 4,096 = 1,024 cycles. Showing that makes the idea of “4,096 units working at once” more concrete, and it makes the practice question easier (2,048 cycles ≈ 2.0 µs).
The caveat could be more concrete. “Data is always ready” is abstract. One line would help: feeding 4 million weights in 1 µs would need about 4 TB/s of memory bandwidth, which is why real chips fall short.
