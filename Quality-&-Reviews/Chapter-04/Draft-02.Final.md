# Section 2: Data Acquisition and Real-Time Processing

> **Big idea.** A sensor reading is only useful if it reaches the processor with the right timestamp, in time to act on. In a moving machine, every millisecond of delay is distance travelled blind.

---

## Learning Objectives

By the end of this section, you should be able to:

1. Describe the data-acquisition chain from sensing element to processor.
2. Calculate the raw data rate of a camera array and compare it with a LiDAR.
3. Build a latency budget and compute how far a vehicle travels while the system is still thinking.
4. Explain why timestamp synchronisation and motion compensation matter.
5. Estimate whether a compute platform can keep up with a sensor array, linking back to Module 01 (TOPS, utilisation).

---

## 2.1 The Story in One Picture

A self-driving car is moving at 100 km/h down a motorway. Its cameras are capturing images, its LiDAR is spinning, and a neural network is deciding whether the car in front is slowing down. But all of that information has to travel through a chain of hardware and software, and every stage takes time. If that chain is slow, the car has already moved several metres before the computer even begins to brake. This is the whole story of this section.

**Figure 2.1.** The data-acquisition chain. Hardware stages (sensing element, ADC, transfer, memory) are shown in green; software or logic stages (pre-processing, timestamping, neural network) are shown in purple. *Source: author's own diagram.*

```mermaid
flowchart LR
  A["<b>Sensing element</b><br/>photons in"]
  B["<b>ADC</b><br/>analog to digital"]
  C["<b>Pre-processing</b><br/>camera: ISP<br/>LiDAR: filter"]
  D["<b>Timestamp</b><br/>and sync"]
  E["<b>Transfer</b><br/>bus, link"]
  F["<b>Memory</b><br/>DRAM / SRAM"]
  G["<b>Neural network</b><br/>on CPU / GPU / NPU"]

  A --> B --> C --> D --> E --> F --> G

  classDef hw fill:#e1f5ee,stroke:#0f6e56,color:#04342c,stroke-width:1px
  classDef sw fill:#eeedfe,stroke:#534ab7,color:#26215c,stroke-width:1px

  class A,B,E,F hw
  class C,D,G sw
```

Everything else in this section is a detail of this picture.

---

## 2.2 The Acquisition Chain

- **Sensing element:** photodiodes in a camera, a laser receiver in a LiDAR, a microphone-like transducer in an ultrasonic sensor.
- **ADC (analogue-to-digital converter):** converts the voltage to a number. A 12-bit ADC gives 4,096 levels.
- **ISP (image signal processor):** hardware that turns raw sensor data into a usable image: removing noise, correcting the lens, adjusting exposure and colour.
- **Timestamp and synchronisation:** each measurement is labelled with the time it was taken.
- **Transfer:** the data crosses a cable or bus. Bandwidth here is limited just like memory bandwidth in Module 01.
- **Memory and compute:** the data lands in memory and a neural network runs on a CPU, GPU or NPU.

---

## 2.3 How Much Data?

### Worked Example 2.A: Data Rate of a Camera Array

*Illustrative array:* 8 cameras, each 1,920 × 1,080 pixels, 12-bit raw output, 36 frames per second.

**Figure 2.2.** The calculation flow for the data rate of the camera array. Inputs are shown in white, intermediate calculations in purple, and the final results in green. *Source: author's own diagram.*

```mermaid
flowchart LR
  subgraph IN["Inputs"]
    direction TB
    W["<b>w × h</b><br/>1,920 × 1,080 pixels"]
    BITS["<b>bits</b><br/>12-bit raw"]
    FPS["<b>fps</b><br/>36 frames per second"]
    N["<b>n</b><br/>8 cameras"]
  end

  FRAME["<b>Frame size</b><br/>w × h × bits ÷ 8<br/>= 3,110,400 bytes"]
  CAM["<b>Per camera</b><br/>frame × fps ÷ 1e6<br/>≈ 112.0 MB/s"]
  TOTAL["<b>Total</b><br/>frame × fps × n ÷ 1e9<br/>≈ 0.9 GB/s"]

  W --> FRAME
  BITS --> FRAME
  FRAME --> CAM
  FPS --> CAM
  CAM --> TOTAL
  N --> TOTAL

  classDef input fill:#ffffff,stroke:#6d5bd0,color:#1f1f1f,stroke-width:1px
  classDef calc fill:#eeedfe,stroke:#534ab7,color:#26215c,stroke-width:1px
  classDef result fill:#e1f5ee,stroke:#0f6e56,color:#04342c,stroke-width:1.5px

  class W,BITS,FPS,N input
  class FRAME,CAM calc
  class TOTAL result

  style IN fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
```

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
w, h, bits, fps, n = 1920, 1080, 12, 36, 8
frame_bytes = w * h * bits / 8
print(frame_bytes)
print(round(frame_bytes * fps / 1e6, 1), "MB/s per camera")
print(round(frame_bytes * fps * n / 1e9, 1), "GB/s total")
```

Expected output:

```text
3110400.0
112.0 MB/s per camera
0.9 GB/s total
```

One camera alone (112 MB/s) produces about three times the data of the 128-beam LiDAR in Worked Example 1.C (37 MB/s). Cameras, not LiDARs, usually dominate the data volume. This data must pass through memory, and Module 01 taught us that moving data costs more energy than computing on it, so keeping raw images on the chip, or compressing them early, matters.

---

## 2.4 Real-Time Deadlines

A **real-time** system must produce a result before a deadline. In a *hard* real-time system, missing the deadline is a failure (a braking command that arrives too late). In a *soft* real-time system, a late result is merely less useful (a delayed video frame). Driving needs hard guarantees for safety functions.

Two different quantities matter, as defined in Module 01:

- **Throughput:** how many frames per second the system can process.
- **Latency:** how long one frame takes from photon to action.

A system can have high throughput and still be dangerously slow to react if the latency of each frame is long.

### Worked Example 2.B: Distance Travelled While Thinking

A car drives at 100 km/h (27.78 m/s) and can brake with a deceleration of 6 m/s². Braking distance = v² ÷ (2a) = 771.6 ÷ 12 ≈ 64.3 m.

**Figure 2.3.** The braking and blind distance calculation. The blind distance is how far the car travels while the system is still processing the sensor data. *Source: author's own diagram.*

```mermaid
flowchart LR
  V["<b>Speed v</b><br/>100 ÷ 3.6<br/>= 27.78 m/s"]
  A["<b>Deceleration a</b><br/>6.0 m/s²"]
  BRAKE["<b>Braking distance</b><br/>v² ÷ (2a)<br/>≈ 64.3 m"]

  subgraph L1["Latency 0.1 s"]
    direction TB
    B1["<b>Blind distance</b><br/>v × 0.1<br/>≈ 2.78 m"]
    T1["<b>Total</b><br/>≈ 67.1 m"]
    B1 --> T1
  end

  subgraph L2["Latency 0.3 s"]
    direction TB
    B2["<b>Blind distance</b><br/>v × 0.3<br/>≈ 8.33 m"]
    T2["<b>Total</b><br/>≈ 72.6 m"]
    B2 --> T2
  end

  V --> BRAKE
  A --> BRAKE
  V --> B1
  V --> B2
  BRAKE --> T1
  BRAKE --> T2

  classDef input fill:#ffffff,stroke:#6d5bd0,color:#1f1f1f,stroke-width:1px
  classDef calc fill:#eeedfe,stroke:#534ab7,color:#26215c,stroke-width:1px
  classDef result fill:#e1f5ee,stroke:#0f6e56,color:#04342c,stroke-width:1.5px

  class V,A input
  class BRAKE,B1,B2 calc
  class T1,T2 result

  style L1 fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
  style L2 fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
```

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
v = 100 / 3.6
a = 6.0
brake = v ** 2 / (2 * a)
print(round(brake, 1), "m braking distance")
for lat in (0.1, 0.3):
    blind = v * lat
    print(round(blind, 2), "m blind,", round(brake + blind, 1), "m total")
```

Expected output:

```text
64.3 m braking distance
2.78 m blind, 67.1 m total
8.33 m blind, 72.6 m total
```

An extra 200 ms of latency adds 5.6 m to the total stopping distance, which can be the difference between a near miss and a collision.

### A Latency Budget

A **latency budget** splits the allowed end-to-end time among the stages. The numbers below are illustrative.

**Table 2.1.** An illustrative latency budget. If the target is 100 ms, the system is already 2 ms over. The biggest slice, inference, is where the hardware ideas of Module 01 (parallelism, low precision, on-chip memory) pay off.

| Stage | Time (ms) |
|---|---|
| Exposure and readout | 10 |
| ISP | 5 |
| Transfer to memory | 2 |
| Neural-network inference | 30 |
| Fusion and tracking | 10 |
| Planning | 15 |
| Actuation (brake build-up) | 30 |
| **Total** | **102** |

---

## 2.5 Timestamps and Motion Compensation

Sensors do not sample at the same instant, and some take time to scan.

- A spinning LiDAR needs 100 ms (at 10 Hz) to complete one sweep. At 20 m/s the car moves 2 m during that sweep, so the first and last points of one frame are measured from places 2 m apart. Without **ego-motion compensation** (correcting each point using the car's own motion, from wheel and IMU data) the point cloud is smeared.
- If the camera and LiDAR timestamps differ by 10 ms at 20 m/s, an object in one sensor's view is displaced by 0.2 m from the same object in the other, which can cause a fused detection to split in two.
- Hardware solutions include a shared clock (for example, precision time protocol over Ethernet) and **hardware triggering**, which makes cameras expose at the instant the LiDAR beam passes their field of view.

---

## 2.6 Can the Computer Keep Up?

This is where Module 01 and Module 04 meet.

### Worked Example 2.C: Compute Needed for a Camera Array

Suppose one frame needs a 100 GFLOP network (illustrative). With 8 cameras at 36 fps:

**Figure 2.4.** The compute calculation. The sustained need is the product of per-frame cost, camera count and frame rate; the peak is the sustained need divided by the utilisation. *Source: author's own diagram.*

```mermaid
flowchart LR
  subgraph IN["Inputs"]
    direction TB
    F["<b>flops_per_frame</b><br/>100 GFLOP"]
    C["<b>cams</b><br/>8 cameras"]
    P["<b>fps</b><br/>36 frames per second"]
  end

  NEED["<b>Compute needed</b><br/>flops_per_frame × cams × fps<br/>= 28.8 TFLOP/s"]
  PEAK["<b>Peak at 40% utilisation</b><br/>need ÷ 0.4<br/>= 72.0 TFLOP/s"]

  F --> NEED
  C --> NEED
  P --> NEED
  NEED --> PEAK

  classDef input fill:#ffffff,stroke:#6d5bd0,color:#1f1f1f,stroke-width:1px
  classDef calc fill:#eeedfe,stroke:#534ab7,color:#26215c,stroke-width:1px
  classDef result fill:#e1f5ee,stroke:#0f6e56,color:#04342c,stroke-width:1.5px

  class F,C,P input
  class NEED calc
  class PEAK result

  style IN fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
```

The following Python program checks the numbers. Its complete expected output is shown below the code.

```python
flops_per_frame = 100e9  # 100 GFLOP
cams = 8
fps = 36
need = flops_per_frame * cams * fps
print(round(need / 1e12, 1), "TFLOP/s needed")
print(round(need / 0.4 / 1e12, 1), "TFLOP/s peak at 40% utilisation")
```

Expected output:

```text
28.8 TFLOP/s needed
72.0 TFLOP/s peak at 40% utilisation
```

The hypothetical accelerator in Module 01 (10 TFLOP/s peak) is far too small. And because the cameras are processed continuously, the realistic requirement is the *peak* figure that gives the needed performance after losses: here about 72 TFLOP/s. This is why self-driving computers carry hundreds of TOPS, and why Module 01's lessons (utilisation, memory-bound versus compute-bound, batch size) decide whether a system works. Note the contrast with a phone unlocking by Face ID: that was one image, one inference, batch 1. A car makes hundreds of inferences per second, continuously, and cannot skip one.

> **Bridge to Networks.** Inside a vehicle, sensors and computers are connected by automotive Ethernet and other buses. Module 02 explains how network bandwidth and latency limit how much sensor data can be delivered.

---

## 2.7 Activities for Section 2

- [ ] **Activity 2.A.** Six cameras, each 1,280 × 960 pixels, 10-bit raw, 30 fps. What is the data rate per camera and in total?
- [ ] **Activity 2.B.** A vehicle travels at 72 km/h, brakes at 5 m/s², and has a total system latency of 150 ms. What is its total stopping distance?
- [ ] **Activity 2.C.** In Table 2.1, inference grows from 30 ms to 60 ms. What is the new total, and by how much does it miss a 100 ms target? Name two hardware or software changes that could recover the time.
- [ ] **Activity 2.D.** Twelve cameras at 30 fps with a 60 GFLOP network. What sustained TFLOP/s is needed, and what peak is required at 30% utilisation?
- [ ] **Activity 2.E.** The camera and LiDAR clocks are 15 ms apart and the car moves at 30 m/s. How far apart are the two views of the same object?

### Answer Key for Activities

**Activity 2.A.**
- Frame size = 1,280 × 960 × 10 ÷ 8 = 1,536,000 bytes.
- Per camera = 1,536,000 × 30 ÷ 1e6 = **46.08 MB/s**.
- Total for 6 cameras = 1,536,000 × 30 × 6 ÷ 
