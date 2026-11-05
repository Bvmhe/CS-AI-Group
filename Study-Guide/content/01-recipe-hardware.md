# Recipe 01 · The Kitchen Crew — AI Hardware Architecture
Source: Main/Chapter-01/01-AI-Hardware-Architecture.md (all numbers checked against its worked examples and answer key).

---

## P01-01 · RECIPE OPENER
TEMPLATE: opener
TEXT:
- RECIPE 01 · The Kitchen Crew
- AI Hardware Architecture
- Mama's Problem: "1,000 plates of fried rice in one minute! Who should cook — one master chef, a thousand helpers, or a machine?"
- Steps: 1 What does AI cook? · 2 Meet the crew (CPU, GPU, TPU, NPU) · 3 Cooking in parallel · 4 The pantry problem (memory) · 5 Case: the home rice cooker (Apple Neural Engine)
- You will: count MACs · pick the right chip · trace a systolic array · use Amdahl's law · use the roofline model
IMG: big fried-rice / wok dish, Mama surprised

## P01-02 · STEP 1 · WHAT DOES AI COOK?
TEMPLATE: explain
TEXT:
- Predict first: a neural network "thinks" by doing… (a) logic puzzles (b) one tiny maths move, billions of times (c) internet searches?
- Answer: (b). Almost every layer multiplies inputs by learned weights and adds them up.
- One "multiply, then add to the running total" = a MAC (multiply-accumulate). 🍳 Like adding one ingredient × its amount to the wok.
- Worked example: weights [0.5, −1.0, 2.0], inputs [4, 3, 1]
  total = 0 → +0.5×4 = 2.0 → +(−1.0×3) = −1.0 → +2.0×1 = 1.0 → output 1.0 (3 MACs)
- Scale it up: a layer with 1,024 inputs and 4,096 outputs = 1,024 × 4,096 = 4,194,304 MACs, for ONE input. Big language models need billions to trillions per answer.
- 🔬 Real tech: 1 MAC = 2 FLOPs (one multiply + one add). Grouping weights in a matrix lets hardware do many MACs at once; a tensor is a many-dimensional matrix (the "T" in TPU).
IMG: stir-frying minigame screenshot; Mama explaining

## P01-03 · MINI-GAME 1.1 · STIR-FRY MATHS
TEMPLATE: activity
TEXT:
- 🎯 Goal: compute a neuron output with MACs and count the work in a layer.
- 🧺 Prep: pen, paper, calculator. Re-read the worked example on the previous page.
- 🔪 Task: (1) weights w = [2, −1, 0.5], inputs x = [1, 4, 6]: write each MAC step and the output. (2) A layer has 512 inputs and 2,048 outputs: how many MACs per input? How many FLOPs?
- 👀 Observe: write your running total after every step.
- 💭 Reflect: why would a chip that does MACs very fast speed up almost the whole model?
- ⭐ Mama's Feedback: (1) 0 → 2 → −2 → 1.0 (3 MACs). (2) 512 × 2,048 = 1,048,576 MACs = 2,097,152 FLOPs. Reflect: MACs are nearly all of a network's work, so speeding them up speeds up nearly everything.
IMG: mixing/cooking minigame, Mama thumbs-up

## P01-04 · STEP 1b · WRITING THE RECIPE vs COOKING THE ORDER
TEMPLATE: explain
TEXT:
- Training = writing the recipe: adjust the model's weights using lots of data. Rare, huge jobs (days–months) in data centres; higher precision (FP32/BF16/FP16); big batches.
- Inference = cooking each order: use the finished weights to answer a request. Happens millions of times; latency, energy and cost per request matter; often batch size 1 and low precision (INT8, 4-bit).
- 🔬 Precision = bits per number: FP32 = 4 bytes, FP16 = 2 bytes, INT8 = 1 byte. Fewer bits → less memory and energy, a little less accuracy.
- Papa says: "Easy! The quoted peak speed is what I'll get." ❌ Peak means every unit busy every cycle — real programs reach much less (utilisation).
- Taste Test: (a) Face ID unlocking (b) fine-tuning a model on 100 GPUs for 3 weeks (c) a chatbot answering 10,000 requests/second — training or inference?
- Answers: (a) inference (b) training (c) inference.
IMG: recipe book / notebook image; Papa

## P01-05 · STEP 2 · MEET THE CREW
TEMPLATE: explain-4cards
TEXT:
- CPU = the Head Chef: a few very skilled cores, can cook ANY dish, one step at a time each. Flexible, low latency. Best for mixed tasks, data loading, small models.
- GPU = 1,000 Line Cooks: thousands of simple cores all doing "chop these onions" together. High parallel throughput. Best for training and serving large models.
- TPU = the Dumpling Conveyor: Google's chip built around a grid of MAC units. Highest efficiency for neural-network maths, least flexible.
- NPU = the Home Rice Cooker: a small, low-power AI engine inside a phone/laptop chip (Apple Neural Engine, Qualcomm Hexagon).
- The rule: the more specialised the cook, the more efficient at that dish — and the less useful for anything else.
- 🔬 CPUs spend chip area on caches and control logic; GPUs on many arithmetic units; TPUs almost entirely on the MAC grid and its memory.
IMG: 4 character/cooking images (one per chip)

## P01-06 · MINI-GAME 1.2 · HIRE THE CHEF
TEMPLATE: activity
TEXT:
- 🎯 Goal: match a chip to a workload and justify it.
- 🧺 Prep: the crew cards on the previous page.
- 🔪 Task: choose CPU, GPU, TPU or NPU: (a) a web server running mixed tasks (b) training a large image model (c) a fixed, high-volume inference service in a data centre (d) a voice assistant in a battery-powered smart speaker — rank NPU, CPU, GPU from best to worst.
- 👀 Observe: for each, write the ONE feature that decided it (flexibility, parallel throughput, efficiency per watt…).
- 💭 Reflect: why might a start-up still choose a GPU even though a TPU was faster in Google's tests?
- ⭐ Mama's Feedback: (a) CPU — mixed, latency-sensitive work needs flexibility. (b) GPU (or TPU on Google Cloud) — huge parallel matrix work. (c) TPU / fixed accelerator — fixed, high volume, best per watt. (d) NPU > CPU > GPU — NPU gives the most work per watt; a GPU drains a small battery. Reflect: unsupported operations fall back to the CPU; the team's software already targets GPUs; GPU availability/cost; memory-bound workloads don't benefit.
IMG: hiring/choosing scene, Mama with clipboard

## P01-07 · STEP 2b · THE DUMPLING CONVEYOR (SYSTOLIC ARRAY)
TEMPLATE: explain-diagram
TEXT:
- A systolic array is a grid of MAC cells. Each cell keeps ONE weight. Inputs and partial sums are passed hand-to-hand to the neighbour — like dumplings on a conveyor — instead of being fetched from memory again and again.
- Worked example: W = [[1, 2], [3, 4]], x = [5, 6]
  Cycle 1: x1 = 5 enters column 1 → 1×5 = 5 and 3×5 = 15
  Cycle 2: x2 = 6 reaches column 2 → 2×6 + 5 = 17 and 4×6 + 15 = 39
  Result y = [17, 39] with 4 MACs.
- Memory reads: systolic 4 weights + 2 inputs = 6; without passing = 8. The saving grows with array size.
- 🔬 First TPU (Jouppi et al., 2017): 65,536 8-bit MACs, 92 TOPS peak, ~15–30× faster and ~30–80× more TOPS/watt than 2015 CPUs/GPUs — but 4 of 6 workloads were limited by memory bandwidth. Historical comparison, not today's ranking.
IMG: 2×2 grid diagram (built in Canva shapes) + dumpling image

## P01-08 · MINI-GAME 1.3 · CONVEYOR BELT TRACE
TEMPLATE: activity
TEXT:
- 🎯 Goal: trace a weight-stationary systolic array cycle by cycle.
- 🧺 Prep: the worked example on the previous page.
- 🔪 Task: W = [[2, 1], [0, 3]], x = [4, 5]. Fill in: Cycle 1 row-1 sum = __, row-2 sum = __. Cycle 2 row-1 sum = __, row-2 sum = __. Count the memory reads with and without neighbour passing.
- 👀 Observe: which numbers were read from memory, and which were handed along?
- 💭 Reflect: why does reusing data matter more than adding more MAC cells?
- ⭐ Mama's Feedback: Cycle 1: 2×4 = 8 and 0×4 = 0. Cycle 2: 1×5 + 8 = 13 and 3×5 + 0 = 15. y = [13, 15]. Reads: 6 (systolic) vs 8 (no reuse). Reflect: fetching data costs far more energy and time than multiplying it (see Step 4).
IMG: dumpling-making minigame screenshot

## P01-09 · STEP 3 · COOKING IN PARALLEL
TEMPLATE: explain
TEXT:
- Mama's Problem: one chef chops 4,194,304 onions one by one. A kitchen with 4,096 helpers chops 4,096 at a time. How much faster?
- Idealised CPU core: 1 MAC per cycle at 3 GHz → 4,194,304 ÷ 3×10⁹ ≈ 1.4 ms
- Idealised accelerator: 4,096 MAC units at 1 GHz → 4,194,304 ÷ 4.096×10¹² ≈ 1.0 µs
- ≈ 1,365× faster, even with a slower clock — because it does 4,096 MACs per cycle.
- Four levels of parallelism: instruction-level (one core overlaps steps) · data-level SIMD/SIMT (GPU warps) · matrix-level (systolic arrays) · across chips (Recipe 02 & 05).
- But… 4 things stop the speed-up: (1) Amdahl's law (2) idle units (3) feeding the units (memory) (4) communication between chips.
- 🔬 Other boosters: quantisation (smaller numbers), specialised units (Tensor Cores, Neural Engine), batching (reuse weights).
IMG: chopping minigame screenshot, many helpers

## P01-10 · MINI-GAME 1.4 · CHOPPING RACE (AMDAHL'S LAW)
TEMPLATE: activity
TEXT:
- 🎯 Goal: discover why adding cooks stops helping.
- 🧺 Prep: speed-up = 1 ÷ ((1 − p) + p ÷ N). p = fraction that can be shared out; N = number of cooks.
- 🔪 Task: a fried-rice order is 90% chopping (shareable, p = 0.9) and 10% waiting for the one wok (not shareable). PREDICT the speed-up with 100 cooks before calculating. Then fill the table for N = 1, 2, 4, 8, 16, 100, "infinity".
- 👀 Observe: N=1 → 1.0× · N=2 → 1.8× · N=4 → 3.1× · N=8 → 4.7× · N=16 → 6.4× · N=100 → 9.2× · ∞ → 10×. Plot it — where does the curve flatten?
- 💭 Reflect: a job is 80% parallel. Speed-up on 8 units? Limit with unlimited units?
- ⭐ Mama's Feedback: 1 ÷ (0.2 + 0.1) ≈ 3.3× on 8 units; limit 1 ÷ 0.2 = 5×. The serial 20% (the single wok!) caps everything. In AI pipelines the "wok" is data loading, pre-processing, or steps that can't be split.
IMG: chopping race / timer images, Mama fire-eyes (for the wok bottleneck)

## P01-11 · STEP 4 · THE PANTRY PROBLEM (MEMORY HIERARCHY)
TEMPLATE: explain-diagram
TEXT:
- In your hands = registers (tiny, instant) → Chopping board = cache / on-chip SRAM → Fridge across the kitchen = DRAM / HBM → Supermarket = storage (huge, slow).
- Faster memory is smaller and more expensive; bigger memory is slower and cheaper.
- The surprising fact (Horowitz, 2014, 45 nm): 32-bit multiply ≈ 3.7 pJ · read from small SRAM ≈ 5 pJ · read from DRAM ≈ 640 pJ.
- A DRAM read costs ≈ 173× a multiply and ≈ 128× an SRAM read. Moving food costs more than cooking it!
- A good cook plans so that everything needed is already on the chopping board. Good AI hardware and software do the same.
- 🔬 HBM = DRAM stacked right next to the chip for very high bandwidth. Unified memory (Apple silicon) = CPU, GPU and NPU share one pool, so no copying.
IMG: fridge / pantry / supermarket photos in a row

## P01-12 · MINI-GAME 1.5 · PANTRY RUN
TEMPLATE: activity
TEXT:
- 🎯 Goal: feel why data movement and number size matter.
- 🧺 Prep: the energy table on the previous page; 1 byte = 8 bits.
- 🔪 Task: (1) Rank from cheapest to most expensive energy per access: DRAM read, register access, small SRAM read. (2) A 7-billion-parameter model: how much memory for its weights in FP16? INT8? 4-bit? (3) A chip doubles its MAC units but keeps the same memory bandwidth. What happens on a memory-bound job?
- 👀 Observe: compare your answers to the size of a phone's memory (≈ 8 GB, shared with other apps).
- 💭 Reflect: which change helps a memory-bound workload more: more MACs or smaller weights?
- ⭐ Mama's Feedback: (1) register < SRAM (~5 pJ) < DRAM (~640 pJ). (2) 14 GB · 7 GB · 3.5 GB. (3) Almost nothing — the units just wait for data. Reflect: smaller weights (fewer bytes to move).
IMG: running/shopping scene, groceries

## P01-13 · STEP 4b · COOK SPEED vs DELIVERY SPEED (ROOFLINE)
TEMPLATE: explain-diagram
TEXT:
- Arithmetic intensity = FLOPs ÷ bytes moved (how much cooking per bag of groceries).
- Attainable performance = min(peak compute, intensity × memory bandwidth). Plotted, it looks like a roof: a slope (memory limit) meeting a flat top (compute limit).
- Ridge point = peak ÷ bandwidth. Below it: memory-bound (cook waits for delivery). Above it: compute-bound (cook is busy).
- Example chip: 10 TFLOP/s, 500 GB/s → ridge = 20 FLOP/byte. Layer 1,024 × 4,096 in FP16:
  Batch 1: intensity ≈ 1.0 → 0.5 TFLOP/s = 5% of peak (memory-bound; ~95% of time waiting)
  Batch 32: intensity ≈ 30.8 → 10 TFLOP/s = 100% (compute-bound)
- Batching = cook 32 plates from one delivery of ingredients: weights fetched once, used 32 times.
- ⚠ Batching raises throughput but each order may wait longer (latency). Phones answer one user at a time → memory-bound.
IMG: roofline drawn with Canva shapes; batch-cooking image (many plates)

## P01-14 · MINI-GAME 1.6 · BATCH COOKING
TEMPLATE: activity
TEXT:
- 🎯 Goal: use the roofline to decide memory-bound vs compute-bound.
- 🧺 Prep: same chip (10 TFLOP/s, 500 GB/s, ridge 20). Layer: 512 inputs × 2,048 outputs, FP16. Weights fetched once per batch; inputs/outputs once per sample.
- 🔪 Task: (1) weights and weight bytes (2) FLOPs at batch 1 (3) intensity at batch 1, 8, 32 — memory- or compute-bound? (4) attainable TFLOP/s and % of peak (5) smallest compute-bound batch.
- 👀 Observe: record a table: batch | intensity | bound | attainable | % peak.
- 💭 Reflect: a voice assistant answers one user at a time within milliseconds. What hardware would you choose?
- ⭐ Mama's Feedback: (1) 1,048,576 weights = 2,097,152 bytes (2) 2,097,152 FLOPs (3–4) batch 1 ≈ 1.0, memory, 0.5 TF, 5% · batch 8 ≈ 7.9, memory, 3.9 TF, 39% · batch 32 ≈ 29.7, compute, 10 TF, 100% (5) batch 22. Reflect: batch 1 is memory-bound → choose high bandwidth, big on-chip memory, low-precision weights (e.g., an NPU).
IMG: many plates / buffet image

## P01-15 · STEP 5 · CASE STUDY: THE HOME RICE COOKER (APPLE NEURAL ENGINE)
TEMPLATE: case
TEXT:
- The Apple Neural Engine (ANE) is the NPU inside Apple's chips. First shipped 2017 in the A11 Bionic (iPhone X), described as 600 billion operations per second.
- It sits on one system-on-a-chip with the CPU, GPU and unified memory. Core ML decides whether each part of a model runs on the CPU, GPU or Neural Engine.
- Face ID = a sense → decide → act loop: 1 scan (TrueDepth camera) → 2 turn face into numbers → 3 AI inference on the Neural Engine → 4 compare with your saved face → 5 unlock. Face data stays on the device (Secure Enclave).
- ✅ Fast · energy-efficient · private · real-time · works offline
- ❌ Limited compute and memory · not every operation supported (falls back to CPU) · big models still need the cloud
- Training happens in the data centre; the finished weights are cooked (inferred) at home on the phone.
IMG: rice cooker / home kitchen image; phone

## P01-16 · TASTE TEST 01 (choose your answer)
TEMPLATE: quiz
TEXT:
- Q1. Face ID processes one face at a time. On the roofline it is most likely… A) compute-bound B) memory-bound C) network-bound
- Q2. A 3-billion-parameter model quantised to 4-bit needs about… A) 12 GB B) 6 GB C) 1.5 GB
- Q3. A model has one unsupported operation in the middle. Why can it run slowly on the NPU? A) the NPU overheats B) data must move to the CPU and back C) Core ML deletes the operation
- Tap/flip to the answers page →
IMG: Mama quiz pose, 3 dishes as answer buttons

## P01-17 · TASTE TEST 01 · ANSWERS
TEMPLATE: answers
TEXT:
- Q1 = B. Batch 1 means low arithmetic intensity: the weights are fetched for just one input, so memory is the limit.
- Q2 = C. 3 × 10⁹ × 0.5 bytes = 1.5 GB — fits on a phone with 8 GB shared memory, but it is still a big share.
- Q3 = B. Each switch to the CPU and back moves data between processors and loses the accelerator's efficiency; the data movement, not the operation, is the cost.
- All right? Mama is proud! ⭐ Stamp Gold. Missed one? Mama will fix it — re-read Steps 4–5 and try again for Silver.
IMG: Mama happy + Mama fire-eyes side by side

## P01-18 · RECIPE CARD 01 (summary)
TEMPLATE: recipecard
TEXT:
- Recipe Card · The Kitchen Crew
- Ingredients (key terms): MAC · FLOP · training / inference · CPU / GPU / TPU / NPU · systolic array · Amdahl's law · memory hierarchy · arithmetic intensity · roofline · batching · quantisation
- Method (big ideas):
  1. AI is mostly MACs — speed those up and you speed up the model.
  2. Specialised chips trade flexibility for efficiency.
  3. Parallelism helps until the serial part, idle units, memory or communication get in the way.
  4. Moving data costs more than computing on it — keep ingredients on the chopping board.
  5. On-device AI (NPU) wins on privacy, latency and battery, but is limited by memory and supported operations.
- 🔓 Unlocked: Recipe Book page 1!
IMG: finished dish, Mama celebrating
