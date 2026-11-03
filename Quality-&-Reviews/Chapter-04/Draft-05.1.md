# Section 5: Case Study — Tesla's Autonomous Driving Technology

> [!IMPORTANT]
> **Big idea** Tesla made the most debated sensor decision in the industry: drive with cameras and a neural network alone. The case shows every idea in this module at once, and it is also a lesson in separating what a company *claims* from what independent evidence *shows*.

**Learning objectives.** By the end of this section you should be able to:

- Describe Tesla's sensor history and its current camera-only (Tesla Vision) design.
- Explain how Tesla's software replaces radar and ultrasonic sensors with learned perception (bird's-eye view, occupancy networks, end-to-end networks).
- Link the in-car computer to Module 01 (MACs, TOPS, memory bandwidth, quantisation) with a calculation.
- Compare Tesla's approach with a sensor-rich alternative (Waymo) using a table.
- Evaluate public claims critically and say what evidence would settle the camera-versus-LiDAR argument.

> [!NOTE]
> **Currency note.** This case study was checked against public sources in early November 2026. Tesla's software, hardware plans and regulatory status change quickly, and several figures below come from news reports of company statements, not from independent measurement. Each such figure is marked *reported* or *company claim*.

## 5.1 A short history of Tesla's sensors

| Period                 | Sensor suite and software milestone                                                                                                                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2016                   | Early Autopilot hardware: eight cameras, a forward-facing radar and twelve ultrasonic sensors, with a plan to reach full self-driving by software updates                                                                   |
| 2019                   | Tesla's own FSD computer ('HW3') with two custom chips, described by Tesla as about 144 TOPS in total                                                                                                                       |
| 2021                   | Radar removed from new Model 3 and Model Y cars, and from Model S and Model X in 2022 (Tesla Support: Tesla Vision)                                                                                                         |
| October 2022           | Ultrasonic sensors dropped from new Model 3 and Model Y; a vision-based **occupancy network** replaces their inputs (Tesla Support; reports in TechCrunch and Electrek)                                                     |
| 2023 onward            | 'HW4' computer and new cameras in new vehicles                                                                                                                                                                              |
| Early 2024 (FSD v12)   | End-to-end neural network replaces much of the hand-written control code (reported: about 300,000 lines of C++)                                                                                                             |
| October 2025 (FSD v14) | Larger model; available on HW4 vehicles first (reported)                                                                                                                                                                    |
| 2025–2026              | Robotaxi service launched in Austin in June 2025 and expanded to further cities during 2026 (reported); Tesla stated that older HW3 vehicles would not be capable of unsupervised driving (Q1 2026 earnings call, reported) |
| 2026–2027              | Next-generation 'AI5' chip: small volumes reported for late 2026, high volume later, after delays                                                                                                                           |

Table 5.1. Timeline of Tesla's sensor and computing changes. Sources: Tesla Support (2021, 2022); Talpes et al. (2020); TechCrunch (2022); Electrek (2022); press coverage of Tesla earnings calls (2026). Dates and volumes marked *reported* were not independently verified.

## 5.2 The sensor suite today

Current Tesla cars use **eight external cameras** giving 360° coverage, with no radar, no LiDAR and no ultrasonic sensors (in markets where Tesla Vision is fully deployed). In Tesla's earlier public descriptions of the layout, three cameras look forward with different lenses (wide, main, narrow; the narrow one for long range, with a maximum range quoted as about 250 m), two look forward and sideways from the B-pillars, two look backward and sideways from the front fenders, and one looks to the rear. Camera resolution is reported to have risen from about 1.2 megapixels (HW3) to about 5 megapixels (HW4).

```
Figure 5.1  Camera coverage (top view, schematic, not to scale)

                  narrow  main  wide
                      \    |    /
          fwd-side L  .---------.  fwd-side R
                  \  /           \  /
                   ||    CAR      ||
                  /  \           /  \
          rear-side L  '---------'  rear-side R
                          |
                         rear
```

There is no directly measured depth anywhere in this system. Every distance comes from a neural network's inference from images. That makes the system a direct test of the depth limits discussed in Section 3.6.

## 5.3 From pixels to a driving decision

```
Figure 5.2  Tesla's perception-to-control pipeline (simplified from public talks)

 8 cameras -> ISP / calibration -> shared image backbone per camera
                                          |
                                          v
                     multi-camera fusion into a 3-D / bird's-eye-view space
                          (attention-based network; uses several past frames)
                                          |
              +---------------------------+---------------------------+
              v                           v                           v
     objects, lanes, signs      3-D OCCUPANCY (what is solid,      motion / velocity
                                even for unknown objects)           per object
              \___________________________|___________________________/
                                          v
                    end-to-end driving network (v12 and later)
                                          v
                       steering and acceleration commands
```

Three ideas from the public talks stand out:

1. **Multi-camera fusion in bird's-eye view.** Instead of detecting objects in each image and merging lists (late fusion), the network fuses *image features* from all eight cameras into one top-down 3-D representation (Tesla AI Day, 2021; Karpathy, 2021). This is mid-level fusion, as described in Section 3.2.
2. **Occupancy networks.** The network predicts which small 3-D cells around the car are occupied, even by objects it cannot name (a fallen ladder, an unusual trailer). This gives geometry like a LiDAR would, but inferred from video (Tesla AI Day, 2022), and it is what Tesla used to replace the ultrasonic sensors.
3. **Video, not single images.** The network uses a memory of recent frames, so it can estimate speed and see through brief occlusions. Using several frames at slightly different positions is like having a moving stereo baseline (Section 3.6).

The **data engine** is the second half of the system. It is a loop that turns driving mistakes into training data.

```
Figure 5.3  The data-engine loop (training in the data centre, inference in the car)

   fleet drives -> rare or difficult case detected -> clip uploaded -> labelled
        ^                                                               |
        |                                                               v
   over-the-air update <- validated in simulation and tests <- network retrained
   (inference runs in the car on the NPU)                  (data centre, GPUs)
```

This is the training-versus-inference split from Module 01, Section 1.5, applied at a scale of millions of vehicles.

## 5.4 The in-car computer: Module 01 in one chip

Tesla's FSD computer uses custom chips, each with two neural-network accelerators. Tesla described each accelerator as a **96 × 96 array of multiply-accumulate (MAC) units** doing 8-bit integer multiplications at about 2 GHz (Talpes et al., 2020). This is the same design idea as the systolic arrays and INT8 quantisation of Module 01, Sections 2 and 3.

### Worked example 5.A: how many TOPS?

One MAC counts as 2 operations (Module 01, Section 1.3).

```python
macs = 96 * 96
clock = 2e9
tops_npu = macs * 2 * clock / 1e12
print(round(tops_npu, 1))
print(round(tops_npu * 4, 1))
print(round(tops_npu * 4 * 0.4, 1))
```

Expected output:

```
36.9
147.5
59.0
```

Four accelerators (two chips of two each) give about 147 TOPS peak, close to Tesla's quoted 144 TOPS. At 40% utilisation (Module 01, Section 1.6), about 59 TOPS are usable, which exceeds the 28.8 TFLOP/s needed by the illustrative camera network of Worked example 2.C. So the arithmetic is plausible.

But Module 01, Section 4, warned that **memory, not arithmetic, is often the limit**. At batch size 1 on a continuous stream, a perception network is typically memory-bound. This matches a report from Tesla's Q1 2026 earnings call, in which the company said HW3 has about one-eighth of the memory bandwidth of HW4, and that bandwidth is a key requirement for unsupervised driving (reported). In roofline terms (Module 01, Section 4.3), the same TOPS figure cannot be reached if the memory cannot feed it; bigger models on the older computer are bandwidth-starved.

## 5.5 Why camera-only? The arguments

| Argument for camera-only                                                                                                      | Argument against                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Humans drive with vision, and roads are designed for eyes (Tesla's public reasoning)                                          | Humans also have a large brain and common sense that networks do not reliably match, so copying the sensor does not copy the capability                |
| Cameras are cheap, so the system can be fitted to every car and scale to millions of vehicles that supply training data       | Cost savings do not remove the cost of the compute, the data engine and any accidents                                                                  |
| Many sensors can disagree; one sensor and one network avoid sensor-conflict problems (Tesla's explanation for dropping radar) | Disagreement is information; a LiDAR or radar that measures geometry directly gives an independent check on the network (Section 4.5)                  |
| High-resolution cameras plus large networks have improved fast                                                                | Camera depth error grows with the square of distance (Worked example 3.C), and cameras are weak in glare, darkness and heavy precipitation (Table 3.1) |
| The data engine finds rare cases at fleet scale                                                                               | All sensing shares one failure mode (common cause); redundancy of the same modality is less powerful than diverse sensing                              |

Table 5.2. Arguments for and against camera-only perception.

> [!WARNING]
> **Caution** A strong argument is not evidence. This table lists arguments. Only measurements, ideally independent ones, can show which side is right.

## 5.6 A contrast: Waymo

| Feature                  | Tesla (consumer cars)                                                                                                                                        | Waymo (robotaxi, sixth-generation Driver)                               |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Cameras                  | 8 external                                                                                                                                                   | 13                                                                      |
| LiDAR                    | None                                                                                                                                                         | 4                                                                       |
| Radar                    | None                                                                                                                                                         | 6 (plus external audio receivers)                                       |
| Direct depth measurement | None; inferred                                                                                                                                               | LiDAR and radar                                                         |
| Role of the human        | FSD (Supervised) is a driver-assistance feature; the driver must supervise (SAE Level 2). Driverless Robotaxi service operates in selected cities (reported) | Driverless service in defined areas (Level 4 style)                     |
| Hardware cost            | Low per car                                                                                                                                                  | Higher; sixth-generation hardware reported at under US\$20,000 per unit |
| Maps                     | Relies mainly on live perception and standard navigation maps                                                                                                | Uses detailed prior maps, matched to live sensor data                   |
| Operating domain         | Broad (anywhere the car can go), with supervision                                                                                                            | Geofenced service areas                                                 |

Table 5.3. Two philosophies. Sources: Tesla Support (2021, 2022); Waymo (2024); Electrek (2026). Waymo's sixth-generation system reduced its sensor count from 29 cameras and 5 LiDARs in the previous generation while adding a 17-megapixel imager (reported).

Notice that Waymo is also *reducing* sensors as its cameras and software improve, while keeping LiDAR and radar. The industry therefore did not settle on 'more sensors is always better'. It is converging on 'enough *diverse* sensors, with better software'.

## 5.7 Evaluating the evidence

Tesla has publicly claimed (company claim, April 2026, from a company executive's post) that supervised FSD has several times fewer collisions per mile than average human drivers. How should a student judge this?

- **Who measured it?** A company's own data is not independent.
- **What is the exposure?** FSD may be used mostly on easier roads (motorways) in good weather; human averages include all roads and conditions.
- **Who is really driving?** With a supervising human, the combined system is measured, not the software alone.
- **What counts as a collision or an intervention?** Definitions differ between companies and regulators.
- **What do regulators find?** The US National Highway Traffic Safety Administration (NHTSA) has opened several investigations into FSD behaviour, including bad-weather crashes and traffic-law violations; these investigations are slow, and their outcomes should be checked at nhtsa.gov (reported).

What would settle the camera-only question? Independent, comparable data: crashes and disengagements per million miles, split by road type, weather and light; third-party tests of low-visibility scenarios; and open reporting of incidents. Until then, the honest conclusion is that camera-only driving is technically *possible to demonstrate* and *not yet proven equal to* a sensor-rich design in all conditions.

## 5.8 Case-study questions

1. Using Worked example 3.C, explain why a camera-only system finds long-range distance harder than a LiDAR system. What techniques does Tesla use to compensate?
2. A report says HW3 has one-eighth of HW4's memory bandwidth. Using the roofline model of Module 01, explain why this could limit performance even if the TOPS figure were similar.
3. Check Worked example 5.A: recompute the TOPS if the array were 128 × 128 at 1.5 GHz.
4. A company says its driver-assistance system is seven times safer than an average driver. List four questions you would ask before accepting this.
5. Removing ultrasonic sensors saves cost and cabling. Give one benefit and one risk, using the idea of common-cause failure.
6. Classify FSD (Supervised) on the SAE scale. What does this imply for driver monitoring?
7. Choose three real-world conditions where you would most want an additional radar or LiDAR on a camera-only car, and explain each with a number or a fact from this module.

## 5.9 Conclusion

Tesla's design shows what is possible when sensing is reduced to cameras and *everything else* is moved into software, data and computing power. The approach connects every section of this module: it needs cameras (Section 1), a high-throughput, low-latency data path (Section 2), a learned fusion and depth-estimation system (Section 3), an end-to-end decision network on a purpose-built NPU (Section 4), and a data engine that depends on training in the data centre. Its weakness is the other side of the same coin: a single sensing modality has correlated failures and an inferred depth. The wiser lesson is not 'cameras versus LiDAR', but *what evidence is needed before a system is trusted without a human*.

## 5.10 Summary

- Tesla removed radar (from 2021) and ultrasonic sensors (from late 2022) and now relies on eight cameras and neural networks; Waymo's sixth-generation Driver uses 13 cameras, 4 LiDARs and 6 radars.
- Tesla's software fuses all cameras in a bird's-eye-view space, predicts 3-D occupancy, uses video over time and, since FSD v12, uses end-to-end networks, trained on fleet data in the data centre and run in the car on an NPU.
- Tesla's FSD computer uses 96 × 96 MAC arrays at 8-bit precision, about 36.9 TOPS each and about 147 TOPS for four, and memory bandwidth is reported to be the key limit on older hardware.
- The camera-only debate is about cost, scale and capability on one side and direct depth, independence and robustness on the other.
- Company safety claims are not independent evidence; the missing evidence is comparable, independently audited data.

