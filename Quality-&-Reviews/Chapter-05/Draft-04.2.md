# Section 4: Challenges in AI Infrastructure: Energy, Cost and Scalability

> **Big idea.** Three ceilings limit how large an AI system can usefully become: the electricity it needs, the money it costs, and the failures it suffers. They are linked, and improving one often pushes pressure onto another.

---

## Learning Objectives

By the end of this section, you should be able to:

1. Distinguish power from energy, and use PUE to compute the energy of a facility.
2. Estimate the compute, time, cost and energy of a training run from its parameters and tokens.
3. Estimate the cost per million tokens of serving a model.
4. Explain why failures become routine at scale, and calculate a sensible checkpoint interval.
5. Describe the trade-offs between energy, cost and scalability.

---

## 4.1 Energy

**Power** is the rate of using energy, in watts (W). **Energy** is power used over time, in joules or, more commonly for electricity, kilowatt-hours (kWh). A 400 W chip running for 10 hours uses 4 kWh.

Chips are only part of the load. A data centre also needs cooling, power conversion and lighting. The ratio is called **PUE (power usage effectiveness)**:

```text
PUE = total facility energy ÷ energy used by the computing equipment
```

A PUE of 1.0 would mean zero overhead. Older facilities often have PUE of 1.5 or worse; the most efficient large operators report values around 1.1.

### Worked Example 4.A: What PUE Is Worth

A computing load of 1 MW, running all year (8,760 hours):

- PUE 1.1: 1 × 1.1 × 8,760 = **9,636 MWh**.
- PUE 1.6: 1 × 1.6 × 8,760 = **14,016 MWh**.
- Difference: **4,380 MWh** per year, which at an illustrative $80 per MWh is **$350,400**, for the same computing.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
it_mw, hours = 1.0, 8760
for pue in (1.1, 1.6):
    print(round(it_mw * pue * hours, 1), 'MWh')
print(round(it_mw * (1.6 - 1.1) * hours, 1), 'MWh')
print(f'${round(4380 * 80)}')
```

Expected output:

```text
9636.0 MWh
14016.0 MWh
4380.0 MWh
$350400
```

**From energy to carbon.** Emissions = energy × carbon intensity of the electricity. Intensity varies enormously by place and time. For an illustrative 89.8 MWh job (Worked Example 4.B): at 0.4 kg CO₂e per kWh the emissions are about 35.9 tonnes; on a grid at 0.05 kg per kWh they are about 4.5 tonnes. *Where and when* a job runs can matter as much as how efficiently it runs.

**Why energy is a design limit.** Module 01 showed that moving a 32-bit word from DRAM costs about 170× the energy of multiplying it (Horowitz, 2014). Every technique that reduces data movement (fusion, lower precision, bigger on-chip memory) is therefore an *energy* technique as well as a speed technique. At the facility scale, the limit is often the power that can be delivered to a site and the heat that can be removed from the racks.

---

## 4.2 Cost

### Estimating Training Compute

A widely used estimate for dense transformer models is

```text
training FLOPs ≈ 6 × N × D
```

where N is the number of parameters and D the number of training tokens (see Kaplan et al., 2020; Hoffmann et al., 2022). The factor 6 is the 2 FLOPs per MAC times 3 (forward plus about twice for backward). Hoffmann et al. found that, for a fixed compute budget, a good balance is roughly 20 tokens per parameter; our 7B model with 1.4 trillion tokens is exactly that.

### Worked Example 4.B: The Full Bill for the Running Example

*Illustrative numbers.* 256 chips, each 200 TFLOP/s peak at 40% MFU, 400 W per chip, $2 per chip-hour, PUE 1.1.

- FLOPs = 6 × 7 × 10⁹ × 1.4 × 10¹² = **5.88 × 10²²**.
- Cluster rate = 256 × 200 × 10¹² × 0.40 = 2.048 × 10¹⁶ FLOP/s.
- Time = 5.88 × 10²² ÷ 2.048 × 10¹⁶ ≈ 2.87 × 10⁶ s ≈ **33.2 days**.
- Chip-hours = 256 × 797.5 h ≈ **204,167**; cost ≈ **$408,333**.
- Chip energy = 256 × 0.4 kW × 797.5 h ≈ 81.7 MWh; with PUE 1.1, **≈ 89.8 MWh**.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
flops = 6 * 7e9 * 1.4e12
rate = 256 * 200e12 * 0.40
secs = flops / rate
hours = secs / 3600
chip_hours = 256 * hours
cost = chip_hours * 2
energy_mwh = 256 * 0.4 * hours / 1000 * 1.1

print(f'{flops:.2e}')
print(round(secs / 86400, 1), 'days')
print(round(chip_hours), 'chip-hours')
print(f'${round(cost)}')
print(round(energy_mwh, 1), 'MWh')
```

Expected output:

```text
5.88e+22
33.2 days
204167 chip-hours
$408333
89.8 MWh
```

Notice what the example shows: improving MFU from 40% to 50% would cut the time, the cost and the energy by 20% each, with no new hardware. This is why Section 3 matters.

> **Caution: what this number leaves out.** It is the cost of one successful run. Real projects pay also for failed runs, experiments and tuning (often several times the final run), data preparation, staff, and the capital cost of buildings and networks.

### Total Cost of Ownership and the Cost of Serving

**Total cost of ownership (TCO)** adds the purchase price (spread over the chip's life, typically a few years), electricity and cooling, networking, buildings, maintenance and staff. For serving, the useful figure is **cost per million tokens**:

```python
cost_per_chip_hour = 2.0
tokens_per_second = 5000
million_tokens_per_hour = tokens_per_second * 3600 / 1e6      # 18.0
print(round(cost_per_chip_hour / million_tokens_per_hour, 3))
```

Expected output:

```text
0.111
```

So a chip delivering 5,000 tokens per second at $2 per hour costs about **$0.11 per million tokens**. Anything that raises tokens per second (batching, quantisation) or utilisation lowers this directly. Training is a one-off cost; serving is a recurring cost that grows with every user, which is why many hardware generations are now designed with inference in mind (Module 01, Section 1.5).

---

## 4.3 Scalability: When Failures Become Routine

A single chip may run for years without failing. A system of thousands of chips will see failures every few hours, because the failure rates add up. If each chip has a mean time between failures (MTBF) of M hours and the system fails when *any* chip fails, the system MTBF is roughly M ÷ N.

### Worked Example 4.C: Failures in a 4,096-Chip Job

*Illustrative numbers.* Each chip's MTBF is 5 years = 43,800 hours.

- System MTBF ≈ 43,800 ÷ 4,096 ≈ **10.7 hours**.
- A month-long job will therefore be interrupted dozens of times. The defence is **checkpointing**: periodically saving the full training state so that, after a failure, the job restarts from the last save rather than from scratch.
- Saving too rarely wastes much work at each failure; saving too often wastes time on saving. A classic approximation (Young, 1974) for the best interval is **√(2 × C × MTBF)**, where C is the time to write one checkpoint.
- With C = 2 minutes (0.0333 h): interval ≈ √(2 × 0.0333 × 10.7) ≈ 0.84 h ≈ **51 minutes**, at a saving overhead of about 3.9%.

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
import math
mtbf_chip = 5 * 8760
system = mtbf_chip / 4096
c = 2 / 60
interval = math.sqrt(2 * c * system)

print(round(system, 2))
print(round(interval * 60, 1))
print(f'{c / interval:.1%}')
```

Expected output:

```text
10.69
50.7
3.9%
```

Failures are not only chips: links, power supplies, cooling and software bugs contribute. Large systems therefore also use spare chips, automatic detection of slow or faulty chips, and fast restart.

**Other scaling walls.** Besides reliability, growth is limited by: the **network** (communication overhead grows with scale; Module 02), the **memory wall** (compute grows faster than memory bandwidth; Module 01, Section 4), the **power and cooling** that a site can supply, and the **supply** of chips and memory.

---

## 4.4 The Trade-offs in One Table

**Table 4.1.** Common levers and their side effects.

| Lever | Helps | May hurt |
|---|---|---|
| More chips | Time to train | Cost, power, failures, communication overhead |
| Lower precision | Speed, energy, memory | Accuracy (needs care) |
| Larger batch | Utilisation, throughput | Latency in serving; learning quality in training |
| Frequent checkpoints | Resilience | Time and storage |
| Cleaner or cheaper power site | Carbon, cost | Location constraints, network distance |
| Custom accelerator (ASIC) | Energy and cost per operation | Flexibility, software ecosystem, design cost |

---

## 4.5 Activities for Section 4

- [ ] **Activity 4.A.** A facility has a computing load of 2 MW. Compare the annual energy at PUE 1.2 and PUE 1.5 (8,760 hours). How much extra energy does the less efficient facility use?
- [ ] **Activity 4.B.** A team trains a 3-billion-parameter model on 60 billion tokens on 64 chips, each delivering 200 TFLOP/s peak at 40% MFU. Calculate the training FLOPs and the training time in hours.
- [ ] **Activity 4.C.** A cluster has 1,024 chips, each with an MTBF of 4 years. Estimate the system MTBF in hours.
- [ ] **Activity 4.D.** Writing one checkpoint takes 3 minutes. Using the MTBF from Activity 4.C, find the best checkpoint interval with Young's approximation.
- [ ] **Activity 4.E.** A chip costing $3 per hour serves 6,000 tokens per second. What is the cost per million tokens?

### Answer Key for Activities

**Activity 4.A.**
- At PUE 1.2: 2 × 1.2 × 8,760 = **21,024 MWh**.
- At PUE 1.5: 2 × 1.5 × 8,760 = **26,280 MWh**.
- Extra energy = 26,280 − 21,024 = **5,256 MWh** per year.

**Activity 4.B.**
- FLOPs = 6 × 3 × 10⁹ × 60 × 10⁹ = 1.08 × 10²¹.
- Cluster rate = 64 × 200 × 10¹² × 0.40 = 5.12 × 10¹⁵ FLOP/s.
- Time = 1.08 × 10²¹ ÷ 5.12 × 10¹⁵ ≈ 2.11 × 10⁵ s ≈ **58.6 hours**.

**Activity 4.C.**
- MTBF per chip = 4 years × 8,760 = 35,040 hours.
- System MTBF ≈ 35,040 ÷ 1,024 ≈ **34.2 hours**.

**Activity 4.D.**
- C = 3 minutes = 0.05 hours.
- Interval = √(2 × 0.05 × 34.2) = √3.42 ≈ 1.85 hours ≈ **111 minutes** (about 1.85 hours).

**Activity 4.E.**
- Million tokens per hour = 6,000 × 3,600 ÷ 1,000,000 = 21.6.
- Cost per million tokens = 3 ÷ 21.6 ≈ **$0.139**.

---

## 4.6 Summary

- Energy is power × time; PUE multiplies the computing energy to give facility energy, and a 0.5 difference in PUE on a 1 MW load is worth about 4,380 MWh per year.
- Training compute is about 6 × N × D; the 7B-parameter, 1.4-trillion-token example needs 5.88 × 10²² FLOPs, about 33 days on 256 chips at 40% MFU, about $408,000 and about 90 MWh.
- Serving cost is best measured per million tokens; at 5,000 tokens/s and $2/hour it is about $0.11.
- System MTBF is roughly chip MTBF ÷ chip count, so a 4,096-chip job fails about every 10.7 hours and needs checkpoints about every 51 minutes (with a 2-minute checkpoint).
- Energy, cost and scalability are linked: most levers that help one place pressure on another.

---

## References

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, B., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., ... Sifre, L. (2022). Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*.

Horowitz, M. (2014). 1.1 Computing's energy problem (and what we can do about it). *IEEE International Solid-State Circuits Conference (ISSCC) Digest of Technical Papers*, 10–14. https://doi.org/10.1109/ISSCC.2014.6757323

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.

Young, J. W. (1974). A first order approximation to the optimum checkpoint interval. *Communications of the ACM, 17*(9), 530–531. https://doi.org/10.1145/361147.361115
