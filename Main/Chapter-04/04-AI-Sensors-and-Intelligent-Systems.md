# Module 04: AI Sensors & Intelligent Systems

|                        |                                                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Course**             | Hardware, Networks & Behaviourist AI                                                                              |
| **Version**            | V1.0 (substantive draft, Stage 1)                                                                                 |
| **Audience**           | Beginners with basic programming knowledge and no hardware background                                             |
| **Prerequisites**      | Basic arithmetic and Module 01 vocabulary: MAC, TOPS, bandwidth, latency, throughput, batch size, utilisation     |
| **Position in course** | Fourth of five modules; builds on Module 01 (AI hardware) and bridges to Module 02 (networks) and behaviourist AI |

## Table of contents

1. [Module overview](#module-overview)
2. [Learning objectives](#learning-objectives)
3. [A running example](#a-running-example)
4. [Section 1: Types of Sensors (Cameras, LiDAR, Ultrasonic)](#section-1-types-of-sensors-cameras-lidar-ultrasonic)
5. [Section 2: Data Acquisition and Real-Time Processing](#section-2-data-acquisition-and-real-time-processing)
6. [Section 3: Sensor Fusion and Environmental Perception](#section-3-sensor-fusion-and-environmental-perception)
7. [Section 4: Integration of Hardware and AI Decision-Making](#section-4-integration-of-hardware-and-ai-decision-making)
8. [Section 5: Case Study — Tesla's Autonomous Driving Technology](#section-5-case-study--teslas-autonomous-driving-technology)
9. [Glossary (all sections)](#glossary-all-sections)
10. [Answer Key (all sections)](#answer-key-all-sections)
11. [References (all sections)](#references-all-sections)
12. [Scope notes and known gaps](#scope-notes-and-known-gaps)

## Module overview

Module 01 explained the chips that run AI. But a chip can only process what it is given. An intelligent system that acts in the physical world (a car, a robot, a drone, a phone) must first **sense** that world, turn the signals into numbers, combine them into a trustworthy picture, and then **decide** what to do, all within a few tenths of a second.

This module follows that chain from the physical sensor to the decision:

- how cameras, LiDAR and ultrasonic sensors measure the world, and what each is bad at,
- how raw signals are acquired, timestamped and processed under real-time deadlines,
- how sensor fusion combines imperfect measurements into one better estimate,
- how perception, prediction, planning and control are mapped onto hardware, and
- how all of this appears in a real, controversial system: Tesla's camera-only driving stack, compared with sensor-rich designs such as Waymo's.

> [!NOTE]
> **Note** Where this module is going. Section 1 introduces the sensors and a comparison table. Section 2 follows the data from sensor to processor and shows why latency and data rate matter. Section 3 shows how fusion works, with a hand-calculated Kalman filter. Section 4 connects perception to decision-making and safety. Section 5 applies everything to Tesla, and discusses what can and cannot be concluded from public evidence.

```
mermaid
flowchart LR
  subgraph S1["S1 Sense"]
    direction TB
    cam[Camera]
    lidar[LiDAR]
    ultra[Ultrasonic]
    radar[Radar]
  end

  subgraph S2["S2 Acquire"]
    direction TB
    adc[ADC]
    isp[ISP]
    ts[Timestamps]
    rt[Real-time deadlines]
  end

  subgraph S3["S3 Fuse & perceive"]
    direction TB
    det[Detect]
    trk[Track]
    fuse[Fuse sensors]
    unc[Estimate uncertainty]
  end

  subgraph S4["S4 Decide & act"]
    direction TB
    pred[Predict]
    plan[Plan]
    ctrl[Control]
    safe[Safety monitors]
  end

  S5["S5 Case study:<br/>Tesla vs sensor-rich designs"]

  S1 --> S2 --> S3 --> S4
  S3 --> S5

  classDef item fill:#ffffff,stroke:#6d5bd0,color:#1f1f1f,stroke-width:1px
  classDef optional fill:#ffffff,stroke:#6d5bd0,color:#1f1f1f,stroke-dasharray:4 3
  classDef study fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a,stroke-width:1.5px

  class cam,lidar,ultra,adc,isp,ts,rt,det,trk,fuse,unc,pred,plan,ctrl,safe item
  class radar optional
  class S5 study

  style S1 fill:#e1f5ee,stroke:#0f6e56,color:#04342c
  style S2 fill:#e1f5ee,stroke:#0f6e56,color:#04342c
  style S3 fill:#eeedfe,stroke:#534ab7,color:#26215c
  style S4 fill:#eeedfe,stroke:#534ab7,color:#26215c
## Learning objectives

By the end of this module, you will be able to:

1. Explain how cameras, LiDAR and ultrasonic sensors measure the world, and compare them on range, resolution, weather sensitivity, direct depth, data rate and cost.
2. Calculate simple sensor quantities by hand: pixels on a target, time-of-flight distance, ultrasonic echo distance, and LiDAR data rate.
3. Describe the data-acquisition chain and compute a latency budget, a stopping distance and a compute requirement for a multi-camera system.
4. Explain why sensor fusion is needed, distinguish early, mid-level and late fusion, and carry out an inverse-variance fusion and one Kalman-filter cycle by hand.
5. Describe how perception, prediction, planning and control are integrated with hardware, and explain how safety and uncertainty shape decisions.
6. Analyse Tesla's sensor strategy using public evidence, compare it with a sensor-rich alternative, and judge the trade-offs without relying on company claims alone.

## A running example

Throughout the module we return to one concrete scene:

> **Running example: a ball rolls into the road at dusk.** A car is driving at 50 km/h. It is getting dark and it has just stopped raining. A ball rolls out between two parked cars, and a child may follow it. The system must notice the ball, work out how far away it is, decide whether to brake, and brake in time. It has a fraction of a second.

Every idea here exists because of pressures like these: dim light, wet surfaces, partial occlusion, a hard deadline and a high cost for a mistake.

# Section 1: Types of Sensors (Cameras, LiDAR, Ultrasonic)

> [!IMPORTANT]
> **Big idea** No sensor sees everything. Each one measures a different physical quantity (light, laser echoes, sound echoes), and each fails in different conditions. Choosing sensors is choosing which failures you can live with.

**Learning objectives.** By the end of this section you should be able to:

- Define a sensor and place it in the sense–decide–act loop.
- Distinguish passive from active and exteroceptive from proprioceptive sensors.
- Explain how a camera, a LiDAR and an ultrasonic sensor each work, and calculate a basic quantity for each.
- Use a comparison table to choose sensors for a scenario and justify the choice.

## 1.1 What is a sensor?

A **sensor** is a device that converts a physical quantity (light, sound, distance, acceleration) into an electrical signal that a computer can read. In Module 01 we described Face ID as a small *sense–decide–act* loop. The same loop, at larger scale, is the structure of every intelligent system:

```
Figure 1.1  The sense-decide-act loop

 WORLD --> SENSORS --> ACQUISITION --> PERCEPTION --> PLANNING --> CONTROL --> ACTUATORS --> WORLD
 (light,   camera       digitise,       detect,         predict,     steering,    motors,       (the loop
  echoes)  LiDAR        timestamp       track, fuse     decide       braking      brakes        closes)
           ultrasonic
```

Two useful ways to classify sensors:

| Classification     | Meaning                                  | Examples                                                                     |
|--------------------|------------------------------------------|------------------------------------------------------------------------------|
| **Passive**        | Only receives energy that already exists | Camera (uses ambient light), microphone                                      |
| **Active**         | Emits energy and measures the echo       | LiDAR (laser), radar (radio waves), ultrasonic (sound)                       |
| **Exteroceptive**  | Measures the outside world               | Camera, LiDAR, radar, ultrasonic                                             |
| **Proprioceptive** | Measures the machine itself              | Wheel-speed sensor, IMU (accelerometer and gyroscope), steering-angle sensor |

Table 1.1. Two ways to classify sensors.

This module concentrates on three exteroceptive sensors (camera, LiDAR, ultrasonic) and also mentions radar, because it appears in almost every real comparison.

## 1.2 Cameras

A **camera** is a passive sensor. A lens focuses light onto an image sensor, a grid of light-sensitive cells (pixels). Each pixel measures how much light reached it, and a colour filter pattern gives red, green and blue values.

**What a camera is good at.** It gives very high resolution and colour, so it can read traffic signs, lane markings, traffic-light colour and brake lights. It is cheap, small and has no moving parts. Because a camera produces a 2-D image, a neural network must *infer* distance from visual cues (object size, perspective, motion, or two cameras at once).

**What it is bad at.** It does not measure distance directly. It depends on ambient light, so low light, glare from a low sun or oncoming headlights, and fast changes in brightness hurt it. Rain, fog and dirt on the lens degrade the image.

**Key camera quantities.**

- **Resolution:** number of pixels, such as 1920 × 1080 (about 2 megapixels).
- **Field of view (FOV):** the angle the camera sees. A wide FOV sees more, but each object covers fewer pixels.
- **Frame rate:** images per second (fps).
- **Dynamic range:** the ratio between the brightest and darkest things it can capture in one frame; important for tunnels and sunsets.

### Worked example 1.A: how many pixels does a child cover?

Illustrative numbers. A camera has 1,920 pixels across and a horizontal field of view of 60°. The target is 0.5 m wide (a small child seen from the front). Under the pinhole model, the focal length expressed in pixels is

```
f = (W / 2) / tan(FOV / 2) = 960 / tan(30°) = 960 / 0.5774 ≈ 1,663 pixels
```

The width of the target in the image is (f × real width) ÷ distance:

- At 50 m: 1,663 × 0.5 ÷ 50 ≈ **16.6 pixels**
- At 150 m: 1,663 × 0.5 ÷ 150 ≈ **5.5 pixels**

At 150 m the child is only about five pixels wide, which is hard for a network to classify reliably. This is why range and resolution are linked.

```python
import math
W, hfov, size = 1920, 60, 0.5
f = (W / 2) / math.tan(math.radians(hfov / 2))
for d in (50, 150):
    print(d, 'm:', round(f * size / d, 1), 'pixels')
```

Expected output:

```
50 m: 16.6 pixels
150 m: 5.5 pixels
```

## 1.3 LiDAR

**LiDAR** (Light Detection and Ranging) is an active sensor. It fires very short laser pulses and measures how long each pulse takes to bounce back. Because light travels at a known speed (c ≈ 3 × 10⁸ m/s), the delay gives the distance directly. This is called **time of flight (ToF)**:

```
distance = (c × t) / 2        (divide by 2 because the pulse goes out and back)
```

By sweeping the laser in many directions (a spinning head, a mirror, or a solid-state design), the sensor builds a **point cloud**: a set of 3-D points (x, y, z) plus an intensity value for the strength of each return.

**What LiDAR is good at.** It measures distance directly and accurately (typically to a few centimetres), works in darkness because it brings its own light, and gives a 3-D shape of the surroundings. It is excellent for detecting obstacles that a camera might not recognise (an unusual object lying in the road).

**What it is bad at.** It does not see colour or read text, so it cannot read a sign or a traffic light. Heavy rain, fog, snow and dust scatter the laser pulses. Very dark or very shiny surfaces may return little energy. The sensor is more expensive than a camera, although costs have fallen as manufacturers moved to solid-state and mass-production designs. A point cloud is sparse: far objects may be hit by only a few points.

### Worked example 1.B: time of flight

A pulse returns after **400 ns** (4 × 10⁻⁷ s).

```
distance = (3 × 10⁸ × 4 × 10⁻⁷) / 2 = 120 / 2 = 60 m
```

### Worked example 1.C: how much data does a LiDAR produce?

Illustrative sensor: 128 beams (vertical channels), a horizontal step of 0.2° over 360° (so 360 ÷ 0.2 = 1,800 steps), spinning at 10 Hz. Each point is stored as four 32-bit numbers (x, y, z, intensity) = 16 bytes.

```python
beams, az, hz = 128, 1800, 10
pts = beams * az
print(pts)
print(pts * hz)
print(pts * 16 / 1e6, 'MB per frame')
print(pts * 16 * hz / 1e6, 'MB/s')
```

Expected output:

```
230400
2304000
3.6864 MB per frame
36.864 MB/s
```

So this LiDAR produces about 2.3 million points per second and about 37 MB/s. Remember this number: in Section 2 we compare it with a camera.

## 1.4 Ultrasonic sensors

An **ultrasonic sensor** is an active sensor that emits a short burst of sound above human hearing (typically about 40 kHz) and listens for the echo. It uses the same round-trip idea as LiDAR, but with sound, which is about a million times slower than light:

```
distance = (v × t) / 2,   where v ≈ 343 m/s in air at 20 °C
```

The speed of sound depends on temperature: v ≈ 331.3 + 0.606 × T (T in °C).

**What it is good at.** It is very cheap, works in the dark and measures short distances to nearly any solid object, including low obstacles below a camera's view. This is why it is used for parking assistance.

**What it is bad at.** Its range is short (typically a few metres), its beam is wide and its angular resolution is poor, so it cannot tell *what* the object is. Because sound is slow, it cannot track fast-moving objects. Wind, other ultrasonic sources and soft, sound-absorbing surfaces (such as clothing) affect it.

### Worked example 1.D: echo to distance, and the temperature error

\(a\) An echo returns after 5.8 ms. With v = 343 m/s: distance = 343 × 0.0058 ÷ 2 ≈ **0.99 m**.

\(b\) An object is truly 1.00 m away on a freezing day (0 °C, where v = 331.3 m/s). The echo takes 2 ÷ 331.3 = 6.04 ms. If the sensor wrongly assumes 343 m/s, it reports 343 × 0.00604 ÷ 2 ≈ 1.035 m, an error of about 3.5%. Good sensors measure temperature and correct for it.

```python
v20, v0 = 343.0, 331.3
t = 2 * 1.00 / v0
print(round(t * 1e3, 2), 'ms')
print(round(v20 * t / 2, 3), 'm')
```

Expected output:

```
6.04 ms
1.035 m
```

## 1.5 A fourth sensor: radar

Radar emits radio waves (automotive radar commonly uses the 76–81 GHz band) and measures the echo. Its special strength is velocity: by the **Doppler effect**, a moving object shifts the frequency of the echo, so radar measures speed directly, and it keeps working in rain, fog and darkness. Its weakness is coarse angular resolution, so it struggles to separate nearby objects or identify shapes.

```
v = (Δf × c) / (2 × f0)
```

Example: at f0 = 77 GHz, a shift of Δf = 10.27 kHz gives v = (10,270 × 3 × 10⁸) ÷ (2 × 77 × 10⁹) ≈ **20 m/s** (72 km/h).

## 1.6 Comparison table

| Feature                      | Camera                                          | LiDAR                                  | Ultrasonic                       | Radar (for reference)                      |
|------------------------------|-------------------------------------------------|----------------------------------------|----------------------------------|--------------------------------------------|
| **Physical principle**       | Light (passive)                                 | Laser time of flight (active)          | Sound echo (active)              | Radio-wave echo and Doppler (active)       |
| **Measures**                 | Colour and texture image                        | 3-D point cloud                        | Short-range distance             | Range and radial velocity                  |
| **Typical useful range**     | Long (tens to 250 m or more, depending on lens) | Roughly 100–250 m for automotive units | About 0.2–5 m                    | Roughly 5–250 m                            |
| **Resolution**               | Very high                                       | High in range, sparse far away         | Very low                         | Low to medium                              |
| **Direct depth?**            | No (must be inferred)                           | Yes, accurate                          | Yes, short range                 | Yes                                        |
| **Direct velocity?**         | No (from frame-to-frame change)                 | No (from successive frames)            | No                               | Yes (Doppler)                              |
| **Works in darkness?**       | Poorly without light                            | Yes                                    | Yes                              | Yes                                        |
| **Rain, fog, snow**          | Degraded                                        | Degraded (scattering)                  | Mostly unaffected at short range | Least affected                             |
| **Reads signs and colours?** | Yes                                             | No                                     | No                               | No                                         |
| **Approximate data rate**    | High (about 100 MB/s per camera, raw)           | Moderate (tens of MB/s)                | Tiny                             | Low to moderate                            |
| **Approximate cost**         | Low                                             | Medium to high, falling                | Very low                         | Low to medium                              |
| **Typical use**              | Lane, sign and object recognition               | 3-D mapping, obstacle detection        | Parking, close-range alerts      | Adaptive cruise control, collision warning |

Table 1.2. Comparison of automotive sensors. Ranges and costs are typical orders of magnitude and vary by product and year; the data rates are derived in Sections 1.3 and 2.2. Sources for the principles: Winner et al. (2016); Thrun, Burgard & Fox (2005); Rosique et al. (2019).

> [!WARNING]
> **Caution** Numbers in a comparison table are *typical*, not guaranteed. A specific sensor can be far better or worse. Always check the datasheet of the product you actually use.

## 1.7 Activities for Section 1

- **Activity 1.A.** A 3,840-pixel-wide camera has a 90° horizontal field of view. How many pixels wide is a 1.8 m car at 100 m?
- **Activity 1.B.** A LiDAR pulse returns after 1.0 µs. How far away is the object?
- **Activity 1.C.** An ultrasonic echo returns after 11.7 ms (v = 343 m/s). How far away is the object?
- **Activity 1.D.** A radar at 77 GHz measures a Doppler shift of 15.4 kHz. What is the object's radial speed?
- **Activity 1.E.** A LiDAR has 64 beams and a horizontal step of 0.4° over 360°, spinning at 10 Hz. How many points per frame, and how many MB per frame at 16 bytes per point?
- **Activity 1.F.** For each situation, say which sensor of the four you would rely on most, and why: (a) reversing into a garage with 30 cm clearance, (b) reading a speed-limit sign, (c) detecting a stalled car in dense fog on a motorway, (d) judging the exact 3-D shape of an unusual object on the road at night.

## 1.8 Summary

- A sensor turns a physical quantity into a signal; sensors can be passive or active, and exteroceptive or proprioceptive.
- Cameras are cheap and rich in detail but measure no distance directly and depend on light; at 150 m a 0.5 m child covers only about 5.5 pixels in our example.
- LiDAR measures distance directly with time of flight (distance = c × t ÷ 2) and produces a point cloud, about 37 MB/s in our example, but cannot read signs and is degraded by precipitation.
- Ultrasonic sensors are cheap and short-range; their accuracy depends on the temperature-dependent speed of sound.
- Radar adds direct velocity and weather robustness but has coarse resolution.
- No sensor dominates on every row of the comparison table, which is the motivation for fusion in Section 3.

# Section 2: Data Acquisition and Real-Time Processing

> [!IMPORTANT]
> **Big idea** A sensor reading is only useful if it reaches the processor with the right timestamp, in time to act on. In a moving machine, every millisecond of delay is distance travelled blind.

**Learning objectives.** By the end of this section you should be able to:

- Describe the data-acquisition chain from sensing element to processor.
- Calculate the raw data rate of a camera array and compare it with a LiDAR.
- Build a latency budget and compute how far a vehicle travels while the system is still thinking.
- Explain why timestamp synchronisation and motion compensation matter.
- Estimate whether a compute platform can keep up with a sensor array, linking back to Module 01 (TOPS, utilisation).

## 2.1 The acquisition chain

```
Figure 2.1  From photons to a perception network

 sensing   ->  ADC      ->  pre-processing  ->  timestamp  ->  transfer   ->  memory   ->  neural network
 element       (analog      (camera: ISP;       and sync       (bus, link)    (DRAM /      on CPU/GPU/NPU
               to digital)   LiDAR: filter)                                   SRAM)
```

- **Sensing element:** photodiodes in a camera, a laser receiver in a LiDAR, a microphone-like transducer in an ultrasonic sensor.
- **ADC (analogue-to-digital converter):** converts the voltage to a number. A 12-bit ADC gives 4,096 levels.
- **ISP (image signal processor):** hardware that turns raw sensor data into a usable image: removing noise, correcting the lens, adjusting exposure and colour.
- **Timestamp and synchronisation:** each measurement is labelled with the time it was taken.
- **Transfer:** the data crosses a cable or bus. Bandwidth here is limited just like memory bandwidth in Module 01.
- **Memory and compute:** the data lands in memory and a neural network runs on a CPU, GPU or NPU.

## 2.2 How much data?

### Worked example 2.A: data rate of a camera array

Illustrative array: 8 cameras, each 1,920 × 1,080 pixels, 12-bit raw output, 36 frames per second.

```python
w, h, bits, fps, n = 1920, 1080, 12, 36, 8
frame = w * h * bits / 8
print(frame)
print(round(frame * fps / 1e6, 1), 'MB/s per camera')
print(round(frame * fps * n / 1e9, 2), 'GB/s total')
```

Expected output:

```
3110400.0
112.0 MB/s per camera
0.9 GB/s total
```

One camera alone (112 MB/s) produces about three times the data of the 128-beam LiDAR in Worked example 1.C (37 MB/s). Cameras, not LiDARs, usually dominate the data volume. This data must pass through memory, and Module 01 taught us that moving data costs more energy than computing on it, so keeping raw images on the chip, or compressing them early, matters.

## 2.3 Real-time deadlines

A **real-time** system must produce a result before a deadline. In a *hard* real-time system, missing the deadline is a failure (a braking command that arrives too late). In a *soft* real-time system, a late result is merely less useful (a delayed video frame). Driving needs hard guarantees for safety functions.

Two different quantities matter, as defined in Module 01:

- **Throughput:** how many frames per second the system can process.
- **Latency:** how long one frame takes from photon to action.

A system can have high throughput and still be dangerously slow to react if the latency of each frame is long.

### Worked example 2.B: distance travelled while thinking

A car drives at 100 km/h (27.78 m/s) and can brake with a deceleration of 6 m/s². Braking distance = v² ÷ (2a) = 771.6 ÷ 12 ≈ 64.3 m.

```python
v = 100 / 3.6
a = 6.0
brake = v ** 2 / (2 * a)
for lat in (0.1, 0.3):
    blind = v * lat
    print(lat, round(blind, 2), round(brake, 1), round(blind + brake, 1))
```

Expected output (latency in s, blind distance, braking distance, total):

```
0.1 2.78 64.3 67.1
0.3 8.33 64.3 72.6
```

An extra 200 ms of latency adds 5.6 m to the total stopping distance, which can be the difference between a near miss and a collision.

### A latency budget

A **latency budget** splits the allowed end-to-end time among the stages. The numbers below are illustrative.

| Stage                      | Time (ms) |
|----------------------------|-----------|
| Exposure and readout       | 10        |
| ISP                        | 5         |
| Transfer to memory         | 2         |
| Neural-network inference   | 30        |
| Fusion and tracking        | 10        |
| Planning                   | 15        |
| Actuation (brake build-up) | 30        |
| **Total**                  | **102**   |

Table 2.1. An illustrative latency budget. If the target is 100 ms, the system is already 2 ms over. The biggest slice, inference, is where the hardware ideas of Module 01 (parallelism, low precision, on-chip memory) pay off.

## 2.4 Timestamps and motion compensation

Sensors do not sample at the same instant, and some take time to scan.

- A spinning LiDAR needs 100 ms (at 10 Hz) to complete one sweep. At 20 m/s the car moves 2 m during that sweep, so the first and last points of one frame are measured from places 2 m apart. Without **ego-motion compensation** (correcting each point using the car's own motion, from wheel and IMU data) the point cloud is smeared.
- If the camera and LiDAR timestamps differ by 10 ms at 20 m/s, an object in one sensor's view is displaced by 0.2 m from the same object in the other, which can cause a fused detection to split in two.
- Hardware solutions include a shared clock (for example, precision time protocol over Ethernet) and **hardware triggering**, which makes cameras expose at the instant the LiDAR beam passes their field of view.

## 2.5 Can the computer keep up?

This is where Module 01 and Module 04 meet.

### Worked example 2.C: compute needed for a camera array

Suppose one frame needs a 100 GFLOP network (illustrative). With 8 cameras at 36 fps:

```python
flops_per_frame = 100e9
cams, fps = 8, 36
need = flops_per_frame * cams * fps
print(round(need / 1e12, 1), 'TFLOP/s')
print(round(need / 1e12 / 0.4, 1), 'TFLOP/s peak at 40% utilisation')
```

Expected output:

```
28.8 TFLOP/s
72.0 TFLOP/s peak at 40% utilisation
```

The hypothetical accelerator in Module 01 (10 TFLOP/s peak) is far too small. And because the cameras are processed continuously, the realistic requirement is the *peak* figure that gives the needed performance after losses: here about 72 TFLOP/s. This is why self-driving computers carry hundreds of TOPS, and why Module 01's lessons (utilisation, memory-bound versus compute-bound, batch size) decide whether a system works. Note the contrast with a phone unlocking by Face ID: that was one image, one inference, batch 1. A car makes hundreds of inferences per second, continuously, and cannot skip one.

> [!NOTE]
> **Note** Bridge to Networks. Inside a vehicle, sensors and computers are connected by automotive Ethernet and other buses. Module 02 explains how network bandwidth and latency limit how much sensor data can be delivered.

## 2.6 Activities for Section 2

- **Activity 2.A.** Six cameras, each 1,280 × 960 pixels, 10-bit raw, 30 fps. What is the data rate per camera and in total?
- **Activity 2.B.** A vehicle travels at 72 km/h, brakes at 5 m/s², and has a total system latency of 150 ms. What is its total stopping distance?
- **Activity 2.C.** In Table 2.1, inference grows from 30 ms to 60 ms. What is the new total, and by how much does it miss a 100 ms target? Name two hardware or software changes that could recover the time.
- **Activity 2.D.** Twelve cameras at 30 fps with a 60 GFLOP network. What sustained TFLOP/s is needed, and what peak is required at 30% utilisation?
- **Activity 2.E.** The camera and LiDAR clocks are 15 ms apart and the car moves at 30 m/s. How far apart are the two views of the same object?

## 2.7 Summary

- The acquisition chain runs from sensing element through ADC, ISP, timestamping and transfer into memory, where a neural network processes it.
- A raw camera produces about 112 MB/s in our example, three times a 128-beam LiDAR; an 8-camera array produces about 0.9 GB/s.
- Real-time systems must meet deadlines; latency, not just throughput, decides safety. At 100 km/h, 100 ms of latency means 2.8 m travelled blind; 300 ms means 8.3 m.
- A latency budget splits the allowed time among stages; inference is usually the largest slice.
- Synchronisation and ego-motion compensation prevent smeared point clouds and mismatched detections.
- A multi-camera array needs tens of TFLOP/s sustained (28.8 in our example), so the utilisation and memory ideas of Module 01 determine feasibility.

# Section 3: Sensor Fusion and Environmental Perception

> [!IMPORTANT]
> **Big idea** Every sensor is wrong in its own way. Sensor fusion combines them so that one sensor's strength covers another's weakness, and so that the system knows *how uncertain* its picture is.

**Learning objectives.** By the end of this section you should be able to:

- Explain why fusion is needed, using complementary failure modes.
- Distinguish early, mid-level (feature) and late (object-level) fusion, and draw the fusion flow.
- Fuse two measurements by inverse-variance weighting and carry out one Kalman-filter cycle by hand.
- Explain data association and tracking, and why camera depth error grows with distance.
- Describe the main perception tasks and metrics (IoU, precision, recall), and name the common causes of fusion failure.

## 3.1 Why fuse?

Section 1 ended with a comparison table in which no sensor won every row. Fusion exploits the fact that the *failures are different*.

| Condition                    | Camera        | LiDAR               | Radar           | Ultrasonic           |
|------------------------------|---------------|---------------------|-----------------|----------------------|
| Night                        | Weak          | Strong              | Strong          | Strong (short range) |
| Heavy rain, fog, snow        | Weak          | Weak to medium      | Strong          | Medium               |
| Low sun glare                | Weak          | Strong              | Strong          | Strong               |
| Dark, non-reflective object  | Weak at night | Weaker returns      | Strong if metal | Strong               |
| Reading signs and lights     | Strong        | Cannot              | Cannot          | Cannot               |
| Measuring speed directly     | Indirect      | Indirect            | Strong          | Weak                 |
| Close range below the bumper | Blind spot    | Possible blind spot | Weak            | Strong               |

Table 3.1. Complementary strengths and weaknesses (qualitative). Based on Rosique et al. (2019) and Feng et al. (2021).

In the running example (a ball at dusk after rain), the camera sees the ball's colour but struggles with low light and wet-road reflections, while a LiDAR or radar return gives the distance regardless of light. Together they are more reliable than either alone.

## 3.2 Where does fusion happen? Three levels

| Level                   | What is fused                                                                                      | Advantage                                          | Disadvantage                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------------------------------|
| **Early (raw-data)**    | Raw pixels and raw points, for example by projecting LiDAR points onto the image                   | Keeps all information                              | Needs precise calibration and synchronisation; heavy computation and data rates |
| **Mid-level (feature)** | Features from each sensor's neural network, merged in a shared space such as bird's-eye view (BEV) | Learns how to combine sensors; strong accuracy     | Needs joint training; harder to debug                                           |
| **Late (object-level)** | Lists of detected objects from each sensor, merged with a tracker                                  | Simple; modular; one failed sensor is easy to drop | Information already thrown away before fusion                                   |

Table 3.2. Three levels of sensor fusion. After Hall & Llinas (1997) and Liu et al. (2023).

```
Figure 3.1  The sensor-fusion flow (feature-level fusion in bird's-eye view)

  CAMERAS ----> [image network] ----> image features --+
                                                        |
  LiDAR   ----> [point network] ----> point features ---+--> [FUSION in common
                                                        |     BEV frame]  --> fused features
  RADAR   ----> [radar network] ----> radar features ---+                          |
                                                                                   v
  IMU + wheel speed --> [ego-motion estimate] --------------> [time alignment]  [detect, segment,
                                                                                 occupancy heads]
                                                                                   |
                                                                                   v
                                            [data association + tracker (Kalman filter)]
                                                                                   |
                                                                                   v
                                  tracked objects with position, velocity, uncertainty -> PLANNING
```

A **bird's-eye view (BEV)** is a top-down map around the vehicle. Putting every sensor's features into the same BEV grid makes it natural to combine them (Philion & Fidler, 2020; Li et al., 2022; Liu et al., 2023).

## 3.3 The simplest fusion: weigh by confidence

Suppose two sensors measure the same distance. A more precise sensor deserves more weight. If a sensor's error has standard deviation σ, its **variance** is σ², and the optimal weight is 1 ÷ σ² (**inverse-variance weighting**).

### Worked example 3.A: radar plus camera

A radar measures the range to the car ahead as 50.0 m with σ = 0.5 m. A camera network estimates 52.0 m with σ = 2.0 m.

- Weights: radar 1 ÷ 0.25 = 4; camera 1 ÷ 4 = 0.25.
- Fused estimate = (4 × 50.0 + 0.25 × 52.0) ÷ (4 + 0.25) = 213 ÷ 4.25 ≈ **50.12 m**.
- Fused variance = 1 ÷ 4.25 ≈ 0.235, so σ ≈ **0.49 m**, slightly better than the best sensor alone.

```python
z = [50.0, 52.0]
s = [0.5, 2.0]
w = [1 / x ** 2 for x in s]
fused = sum(wi * zi for wi, zi in zip(w, z)) / sum(w)
sigma = (1 / sum(w)) ** 0.5
print(round(fused, 1), round(sigma, 2))
```

Expected output:

```
50.1 0.49
```

> [!WARNING]
> **Caution** This assumes the errors are independent, unbiased and roughly bell-shaped. If both sensors share a flaw (for example, both are degraded by fog), the fused result is *less* trustworthy than the formula claims.

## 3.4 Fusion over time: the Kalman filter

Measurements arrive repeatedly, and objects move. The **Kalman filter** (Kalman, 1960) keeps a running estimate *and* its uncertainty, and alternates two steps:

1. **Predict:** use a motion model to move the estimate forward in time. Uncertainty grows, because the model is imperfect (process noise Q).
2. **Update:** when a measurement z arrives (with noise variance R), correct the estimate. The **Kalman gain** K = P ÷ (P + R) decides how much to trust the measurement versus the prediction. Then x ← x + K(z − x), and P ← (1 − K)P.

### Worked example 3.B: one predict step and two updates

Distance to the lead car: prior estimate x = 50.0 m with variance P = 4.0. Process noise Q = 1.0.

- **Predict:** P = 4.0 + 1.0 = 5.0.
- **Update 1 (radar):** z = 52.0, R = 1.0. K = 5 ÷ 6 = 0.833. x = 50 + 0.833 × 2 = 51.67. P = (1 − 0.833) × 5 = 0.83.
- **Update 2 (camera):** z = 50.5, R = 4.0. K = 0.833 ÷ 4.833 = 0.172. x = 51.67 + 0.172 × (50.5 − 51.67) = 51.47. P = (1 − 0.172) × 0.833 = 0.69.

```python
x, P = 50.0, 4.0
Q = 1.0
P = P + Q

def update(x, P, z, R):
    K = P / (P + R)
    return x + K * (z - x), (1 - K) * P

x, P = update(x, P, 52.0, 1.0)
print(round(x, 2), round(P, 2))
x, P = update(x, P, 50.5, 4.0)
print(round(x, 2), round(P, 2))
```

Expected output:

```
51.67 0.83
51.47 0.69
```

Notice three things. The precise radar pulled the estimate strongly; the noisier camera pulled it only a little; and the uncertainty P *fell* with each update. Real trackers use a state with position and velocity (a constant-velocity model), so they also estimate speed, but the idea is identical. Thrun, Burgard & Fox (2005) give a full treatment.

## 3.5 Association and tracking

Before fusing, the system must decide which detection belongs to which object. This is **data association**.

- **Gating:** only detections within a window around the predicted position are considered. If a track is predicted at 30.0 m with a gate of ±3 m, a detection at 29.2 m is accepted and one at 33.5 m is rejected.
- **Assignment:** when several tracks and detections compete, an algorithm such as the Hungarian algorithm finds the best one-to-one matching.
- **Track life cycle:** a new detection starts a *tentative* track; it becomes *confirmed* after being seen on several frames; it is *deleted* if unmatched for too long. This suppresses single-frame false alarms.

## 3.6 Depth from cameras and why it gets worse with range

A single camera does not measure depth. With **two cameras** a known distance B apart (the baseline), a point appears shifted between the two images by a **disparity** d (in pixels), and depth is

```
Z = (f × B) / d
```

### Worked example 3.C: stereo depth and its sensitivity

Let f = 1,000 pixels and B = 0.30 m.

```python
f, B = 1000, 0.3
for d in (5, 6, 7):
    print(d, round(f * B / d, 1))
```

Expected output:

```
5 60.0
6 50.0
7 42.9
```

A one-pixel error in disparity moves the answer from 50 m to 42.9 m or 60 m. The depth error per pixel of disparity error is approximately Z² ÷ (f × B): at 50 m, 2,500 ÷ 300 ≈ 8.3 m. Because of the *square*, camera depth error rises quickly with range, whereas a LiDAR's error stays at a few centimetres. Modern systems also learn depth from a single image or video using neural networks, and some convert it into a pseudo-LiDAR point cloud (Wang et al., 2019); this works well on average but is still an estimate, not a measurement.

## 3.7 What perception outputs, and how it is scored

| Task                       | Output                                                         |
|----------------------------|----------------------------------------------------------------|
| **Object detection**       | 3-D boxes with class (car, pedestrian, cyclist)                |
| **Semantic segmentation**  | A class label for every pixel or point                         |
| **Tracking**               | Identity, velocity and history of each object                  |
| **Occupancy / free space** | Which parts of 3-D space are occupied, even by unknown objects |
| **Lane and map elements**  | Lane lines, curbs, crosswalks, traffic lights, signs           |

Table 3.3. The main perception tasks.

Public datasets such as KITTI (Geiger, Lenz & Urtasun, 2012), nuScenes (Caesar et al., 2020) and the Waymo Open Dataset (Sun et al., 2020) provide labelled camera, LiDAR and radar data, so methods can be compared fairly.

**Scoring a detection.** **Intersection over Union (IoU)** measures how well a predicted box overlaps the true box: IoU = overlap area ÷ union area. **Precision** = true detections ÷ all detections made. **Recall** = true detections ÷ all real objects.

### Worked example 3.D: IoU, precision and recall

A predicted box is 4 m × 2 m (8 m²) and the true box is the same size but shifted 1 m along its length. Overlap = 3 × 2 = 6 m². Union = 8 + 8 − 6 = 10 m². IoU = 6 ÷ 10 = **0.6**.

A detector makes 100 detections, of which 90 are correct and 10 are false alarms, and it misses 30 real objects. Precision = 90 ÷ 100 = **0.90**. Recall = 90 ÷ 120 = **0.75**. In driving, false alarms cause phantom braking and misses cause collisions; Section 4 shows how to weigh them.

## 3.8 How fusion fails

- **Miscalibration.** Fusion needs the exact position and angle of every sensor (the *extrinsic* calibration). A 1° yaw error between a camera and a LiDAR shifts an object at 50 m sideways by 50 × tan(1°) ≈ 0.87 m, enough to put a pedestrian in the wrong lane.
- **Poor synchronisation.** See Section 2.4.
- **Sensor disagreement.** When the camera says there is a car and the radar says there is not, which is right? Voting, confidence weights and learned fusion all try to answer this, and none is perfect.
- **Common-cause failures.** Fog or a dirty windscreen can degrade several sensors at once, breaking the independence assumption (Section 4.5).
- **Adversarial and rare cases.** Stickers on signs, reflective surfaces and unusual objects can fool networks, which is a reason to keep a sensor that measures geometry directly.

## 3.9 Activities for Section 3

- **Activity 3.A.** A LiDAR measures 30.0 m with σ = 0.1 m; a camera estimates 31.0 m with σ = 1.0 m. Compute the inverse-variance fused estimate and its σ.
- **Activity 3.B.** Starting from x = 20.0, P = 2.0, Q = 0.5, apply one predict step and then one update with z = 21.0 and R = 1.5. Give x and P.
- **Activity 3.C.** Stereo cameras with f = 1,200 pixels and B = 0.5 m see a disparity of 12 pixels. What is the depth, and what is the depth if the disparity is 11 pixels?
- **Activity 3.D.** A predicted box is 5 m × 2 m and the true box overlaps it by 4 m × 2 m (same size boxes). Compute the IoU.
- **Activity 3.E.** A detector makes 80 detections, 72 of them correct, and misses 8 objects. Compute precision and recall.
- **Activity 3.F.** A camera–LiDAR yaw calibration error is 0.5°. How large is the sideways offset at 80 m?
- **Activity 3.G.** Explain, in two or three sentences, why fusing a camera and a LiDAR in thick fog is less beneficial than fusing a camera and a radar.

## 3.10 Summary

- Sensors fail in different ways, so combining them improves reliability and measurably reduces uncertainty.
- Fusion can happen on raw data (early), on features (mid-level, often in bird's-eye view) or on object lists (late).
- Inverse-variance weighting gave a fused range of 50.1 m with σ ≈ 0.49 m in our example; a Kalman filter does the same over time and tracks uncertainty (51.67 m then 51.47 m, with P falling from 5.0 to 0.69).
- Association (gating, assignment, track life cycle) decides which measurement belongs to which object.
- Stereo depth error grows with the square of distance (about 8.3 m per pixel at 50 m in our example), which is why direct range sensors remain valuable.
- Perception is scored with IoU, precision and recall; fusion fails through miscalibration, poor timing, disagreement and common-cause conditions.

# Section 4: Integration of Hardware and AI Decision-Making

> [!IMPORTANT]
> **Big idea** Perception is only useful if it changes what the machine does. Turning an uncertain picture into a safe action is a chain of software modules mapped onto hardware, each with its own deadline, and with safety checks wrapped around it.

**Learning objectives.** By the end of this section you should be able to:

- Draw the autonomy stack (perception, prediction, planning, control) and map each stage to hardware.
- Compare modular and end-to-end architectures.
- Calculate whether a vehicle can stop in time given detection range and latency.
- Use a cost-based rule to choose a decision threshold under uncertainty.
- Describe SAE automation levels, functional safety and redundancy, and why redundancy needs independence.

## 4.1 The autonomy stack

```
Figure 4.1  A modular autonomy stack with safety monitoring

 sensors -> ACQUIRE -> PERCEPTION ---> LOCALISATION -> PREDICTION -> PLANNING -> CONTROL -> actuators
                       (detect, track,  (where am I?)    (where will    (route,      (steering,   (steer, brake,
                        fuse)                             others go?)    behaviour,   throttle     accelerate)
                                                                         trajectory)  commands)
                                  ^                                                      |
                                  |_________ feedback: the vehicle's own motion _________|

            SAFETY MONITOR (runs beside the stack): checks plausibility, deadlines,
            sensor health, and can trigger a minimal-risk manoeuvre (safe stop)
```

- **Perception** (Section 3) outputs tracked objects with uncertainty.
- **Localisation** estimates the vehicle's own position, using GNSS, an IMU, wheel odometry, and maps or camera and LiDAR features.
- **Prediction** forecasts where other road users will be in the next few seconds, often as several possible trajectories with probabilities.
- **Planning** chooses a route, a behaviour (follow, overtake, stop) and a smooth, collision-free trajectory.
- **Control** converts the trajectory into steering, throttle and brake commands, usually at 50–100 Hz or faster.

### Modular versus end-to-end

The stack above is *modular*: each stage is a separate component with a defined interface. The alternative is *end-to-end*: one neural network takes sensor data and outputs a trajectory or controls. An early demonstration trained a convolutional network to steer directly from camera images (Bojarski et al., 2016), and more recent work trains one network with several linked tasks (Hu et al., 2023).

|                      | Modular                                         | End-to-end                                   |
|----------------------|-------------------------------------------------|----------------------------------------------|
| **Interfaces**       | Explicit (object lists, trajectories)           | Learned, implicit                            |
| **Debugging**        | Easier: find the failing module                 | Harder: behaviour is hidden in the weights   |
| **Safety argument**  | Easier to test and certify per module           | Harder to certify; needs very strong testing |
| **Information loss** | Detail is lost at each interface                | Less loss; the network can keep what matters |
| **Data needs**       | Each module can be trained and tuned separately | Needs very large amounts of driving data     |
| **Compute**          | Many specialised pieces                         | One or a few very large networks             |

Table 4.1. Modular versus end-to-end architectures. Real systems are often hybrids.

## 4.2 Mapping the stack onto hardware

Module 01 showed that different chips suit different work. A vehicle computer is a **heterogeneous** system that combines several:

| Stage                          | Typical workload                   | Typical hardware             | Typical rate   |
|--------------------------------|------------------------------------|------------------------------|----------------|
| Sensor capture and ISP         | Fixed image processing             | Dedicated ISP blocks         | 30–60 fps      |
| Neural-network perception      | Huge matrix arithmetic             | NPU or GPU (Module 01)       | 10–60 Hz       |
| Fusion, tracking, localisation | Many small, irregular computations | CPU, DSP                     | 10–100 Hz      |
| Prediction and planning        | Mixed neural and search code       | CPU plus accelerator         | About 10 Hz    |
| Control and safety monitor     | Simple but strictly timed code     | Safety-rated microcontroller | 100 Hz or more |

Table 4.2. A typical mapping of autonomy functions to hardware. The rates are illustrative.

Two lessons from earlier modules apply directly. First, **utilisation and memory** (Module 01): a perception network at batch size 1 on a continuous stream tends to be memory-bound, so on-chip memory and low precision matter. Second, **unsupported operations falling back to the CPU** (Module 01, Section 2.7) can silently blow the latency budget of Section 2.3.

## 4.3 Can the car stop in time?

The earlier running example becomes a calculation. The car drives at 50 km/h (13.89 m/s). System latency is 0.15 s and it can brake at 7 m/s². A ball (and perhaps a child) is detected at distance d. Will the car stop?

### Worked example 4.A: detection range decides the outcome

- Distance travelled during latency: 13.89 × 0.15 ≈ 2.08 m.
- Braking distance: 13.89² ÷ (2 × 7) ≈ 13.78 m. Total needed ≈ **15.9 m**.
- If detection happens at 25 m, the car stops with about 9 m to spare.
- If detection happens at only 15 m, the car cannot stop. After the 2.08 m latency it has 12.92 m left; its speed when it reaches the obstacle is √(13.89² − 2 × 7 × 12.92) ≈ 3.5 m/s (about 12.5 km/h).

```python
v = 50 / 3.6
a = 7.0
lat = 0.15

def impact(d):
    rem = d - v * lat
    v2 = v ** 2 - 2 * a * rem
    return 0.0 if v2 <= 0 else v2 ** 0.5

for d in (25, 15):
    print(d, round(impact(d), 1))
```

Expected output (detection distance in m, impact speed in m/s):

```
25 0.0
15 3.5
```

This ties the whole module together: detection range (Section 1: pixels on target, LiDAR range), latency (Section 2), and the quality of the fused estimate (Section 3) decide whether the decision can be carried out at all. Even a 3.5 m/s impact is a serious injury risk for a child, so early, reliable detection beats any amount of late braking.

## 4.4 Deciding under uncertainty

The perception system rarely outputs a certain answer. It outputs *probabilities* and *uncertainties*. The planner must act anyway. A principled way to choose is to compare **expected costs**.

### Worked example 4.B: when should the car brake?

Let the cost of an unnecessary hard brake (a false alarm) be 1 unit, and the cost of failing to brake for a real pedestrian be 1,000 units. Let p be the probability that the object is a real pedestrian.

- Expected cost of braking = (1 − p) × 1 (cost arises only if it was a false alarm).
- Expected cost of not braking = p × 1,000.
- Brake when (1 − p) \< 1,000p, that is when p \> 1 ÷ 1,001 ≈ **0.1%**.

```python
cost_fp, cost_fn = 1, 1000
print(round(cost_fp / (cost_fp + cost_fn), 6))
```

Expected output:

```
0.000999
```

The threshold is tiny because the two errors are so unequal. But if the car brakes at 0.1% confidence, it will brake often for harmless things (shadows, plastic bags): the *phantom braking* problem, which is irritating and can itself cause rear-end collisions. The real cost ratio is not 1,000:1 at every speed and in every traffic situation, so the thresholds are tuned with data, simulation and road testing. The lesson is that **precision and recall (Section 3.7) are not abstract scores; they set a safety-versus-comfort trade-off.**

The fused uncertainty also feeds the planner. In Worked example 3.B, the fused distance was 51.47 m with variance 0.69 (σ ≈ 0.83 m). A cautious planner may plan with a conservative distance of 51.47 − 2 × 0.83 ≈ 49.8 m (two standard deviations closer), so that a bad measurement does not cause a collision.

## 4.5 Safety, levels of automation and redundancy

**Levels of automation.** SAE J3016 defines six levels of driving automation (SAE International, 2021):

| Level | Name                                      | Who monitors the road?                          |
|-------|-------------------------------------------|-------------------------------------------------|
| 0     | No automation                             | Human                                           |
| 1     | Driver assistance (steering *or* speed)   | Human                                           |
| 2     | Partial automation (steering *and* speed) | Human must supervise at all times               |
| 3     | Conditional automation                    | System, but the human must take over on request |
| 4     | High automation                           | System, within a defined operating domain       |
| 5     | Full automation                           | System, everywhere                              |

Table 4.3. SAE levels of driving automation. Consumer driver-assist features that require constant human supervision are Level 2, whatever their marketing names.

**Functional safety.** ISO 26262 is the standard for the safety of electronic systems in cars; it classifies the risk of each function into integrity levels (ASIL A to D, with D the strictest) and sets design and testing requirements. **SOTIF** (ISO 21448, Safety Of The Intended Functionality) covers a different problem: hazards that occur even when nothing is broken, because a sensor or a neural network reaches its performance limits (for example, a camera blinded by glare).

**Redundancy.** Safety-critical systems duplicate important parts: two computers, two power supplies, and *diverse* sensors that measure the world in different ways. If a function fails, a **fail-operational** system keeps working in a reduced mode, while a **fail-safe** system stops safely. The vehicle executes a **minimal-risk manoeuvre**, such as pulling over, when it can no longer drive safely.

### Why redundancy needs independence

Suppose each sensing channel misses a hazard 5% of the time. If two channels fail *independently*, both miss it only 0.05 × 0.05 = **0.25%** of the time, which is 20 times better. If the failures are perfectly *correlated* (a single cause, such as fog or a dirty windscreen, blinds both), both miss it **5%** of the time, no better than one channel. Real systems fall in between. This is the engineering argument for using *different physical principles* (camera, radar, LiDAR) rather than several copies of the same sensor. The counter-argument (Section 5) is that a single, very capable sensing channel plus a very good network can be simpler, cheaper and avoid the problem of conflicting sensors.

## 4.6 Activities for Section 4

- **Activity 4.A.** A vehicle drives at 36 km/h, with 0.2 s latency and a braking deceleration of 6 m/s². What is the minimum detection distance needed to stop before an obstacle?
- **Activity 4.B.** Using the numbers of Worked example 4.A, at what detection distance is the impact speed exactly zero? (Hint: the answer is the total needed distance.)
- **Activity 4.C.** If a false alarm costs 2 units and a missed pedestrian costs 600 units, what is the probability threshold above which the car should brake?
- **Activity 4.D.** Two independent channels each miss a hazard with probability 0.02. What is the probability that both miss it? What if they are perfectly correlated?
- **Activity 4.E.** Name two advantages and two disadvantages of an end-to-end architecture compared with a modular one.
- **Activity 4.F.** Explain the difference between ISO 26262 and SOTIF using one example of each kind of hazard.

## 4.7 Summary

- The autonomy stack runs perception, localisation, prediction, planning and control in a loop, with a safety monitor beside it; modular designs are easier to debug and certify, end-to-end designs lose less information but are harder to verify.
- The stack maps onto heterogeneous hardware: ISPs, NPUs/GPUs, CPUs and safety-rated microcontrollers, each with its own rate and deadline.
- In our example a car at 50 km/h with 0.15 s latency needs about 15.9 m to stop; detection at 15 m still leaves a 3.5 m/s impact.
- Decisions under uncertainty compare expected costs; with a 1:1,000 cost ratio the braking threshold falls to about 0.1%, which shows why false alarms and misses must be balanced with data.
- SAE levels describe who is responsible; ISO 26262 and SOTIF address faults and performance limits; redundancy only helps if failures are independent (0.25% versus 5% in our example).

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

# Glossary (all sections)

| Term                              | Meaning                                                                                                                |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------|
| **Active sensor**                 | A sensor that emits energy (light, sound, radio) and measures the echo, such as LiDAR, radar and ultrasonic.           |
| **ADC**                           | Analogue-to-digital converter; turns a voltage into a number.                                                          |
| **ASIL**                          | Automotive Safety Integrity Level; ISO 26262 risk classes from A (lowest) to D (highest).                              |
| **Bandwidth**                     | Bytes per second that can be moved between two parts of a system (Module 01).                                          |
| **Batch size**                    | The number of inputs processed together; a driving system usually works at batch size 1 per camera stream (Module 01). |
| **Bird's-eye view (BEV)**         | A top-down map around the vehicle used as a common space for fusing sensors.                                           |
| **Calibration**                   | Measuring a sensor's internal properties (intrinsic) and its position and angle on the vehicle (extrinsic).            |
| **Camera**                        | A passive sensor that records an image of light; gives rich detail but no direct depth.                                |
| **Data association**              | Deciding which detection belongs to which tracked object.                                                              |
| **Data engine**                   | A loop that finds difficult cases in fleet driving and turns them into training data.                                  |
| **Disparity**                     | The pixel shift of a point between two camera images; depth Z = f × B ÷ d.                                             |
| **Doppler effect**                | The shift in frequency of an echo caused by relative motion; radar uses it to measure speed.                           |
| **Dynamic range**                 | The range between the brightest and darkest light a camera can capture in one frame.                                   |
| **Ego-motion compensation**       | Correcting measurements for the sensor platform's own motion during a scan.                                            |
| **End-to-end driving**            | A single neural network that maps sensor data to trajectory or controls.                                               |
| **Exteroceptive sensor**          | A sensor that measures the outside world.                                                                              |
| **Fail-operational / fail-safe**  | A system that keeps working in a reduced mode after a fault / a system that moves to a safe state, such as stopping.   |
| **Field of view (FOV)**           | The angle that a sensor can see.                                                                                       |
| **Frame rate**                    | Images per second (fps).                                                                                               |
| **Functional safety**             | The discipline (ISO 26262) of preventing hazards caused by faults in electronic systems.                               |
| **Gating**                        | Restricting data association to detections near a track's predicted position.                                          |
| **Hard real-time**                | A system in which missing a deadline counts as a failure.                                                              |
| **Hardware triggering**           | Making sensors capture at a shared, controlled instant.                                                                |
| **IMU**                           | Inertial measurement unit; accelerometers and gyroscopes that measure the vehicle's motion.                            |
| **Inference**                     | Running a trained network to get an answer (Module 01).                                                                |
| **Intersection over Union (IoU)** | Overlap area divided by union area of a predicted and a true box.                                                      |
| **ISP**                           | Image signal processor; hardware that converts raw sensor data into a usable image.                                    |
| **Kalman filter**                 | A predict-and-update algorithm that tracks a state and its uncertainty over time.                                      |
| **Kalman gain**                   | K = P ÷ (P + R); the weight given to a new measurement.                                                                |
| **Latency**                       | Time from the start to the end of one task (Module 01); for driving, from photons to action.                           |
| **Latency budget**                | The allocation of the allowed end-to-end time among the processing stages.                                             |
| **LiDAR**                         | Light Detection and Ranging; measures distance by laser time of flight and produces a point cloud.                     |
| **MAC**                           | Multiply-accumulate; one MAC counts as 2 operations (Module 01).                                                       |
| **Minimal-risk manoeuvre**        | A safe fallback action, such as pulling over, when the system can no longer drive safely.                              |
| **Modular stack**                 | An autonomy design made of separate modules (perception, prediction, planning, control) with explicit interfaces.      |
| **NPU**                           | Neural processing unit; a low-power AI accelerator (Module 01).                                                        |
| **Occupancy network**             | A network that predicts which 3-D cells around the vehicle are occupied.                                               |
| **Passive sensor**                | A sensor that only receives existing energy, such as a camera.                                                         |
| **Phantom braking**               | Braking caused by a false detection.                                                                                   |
| **Point cloud**                   | A set of 3-D points measured by a LiDAR.                                                                               |
| **Precision / Recall**            | The share of detections that are correct / the share of real objects found.                                            |
| **Proprioceptive sensor**         | A sensor that measures the machine itself, such as wheel speed or an IMU.                                              |
| **Radar**                         | A sensor using radio-wave echoes to measure range and, through Doppler, speed.                                         |
| **Real-time system**              | A system required to respond within a deadline.                                                                        |
| **Redundancy**                    | Duplicating functions or sensors so that one failure does not cause a hazard.                                          |
| **Roofline model**                | Attainable performance = min(peak compute, intensity × bandwidth) (Module 01).                                         |
| **SAE levels**                    | SAE J3016 classification of driving automation from Level 0 to Level 5.                                                |
| **Sensor**                        | A device that converts a physical quantity into an electrical signal.                                                  |
| **Sensor fusion**                 | Combining measurements from several sensors into one estimate.                                                         |
| **SOTIF**                         | Safety Of The Intended Functionality (ISO 21448); hazards from performance limits when nothing is broken.              |
| **Stereo vision**                 | Estimating depth from two cameras a known distance apart.                                                              |
| **Throughput**                    | Amount of work finished per second (Module 01).                                                                        |
| **Time of flight (ToF)**          | Measuring distance from the round-trip time of an emitted pulse: d = v × t ÷ 2.                                        |
| **TOPS**                          | Tera operations per second (10¹²) (Module 01).                                                                         |
| **Tracking**                      | Following an object's identity, position and velocity across time.                                                     |
| **Ultrasonic sensor**             | A sensor using ultrasound echoes for short-range distance.                                                             |
| **Utilisation**                   | The fraction of a chip's peak that is doing useful work (Module 01).                                                   |
| **Variance**                      | The square of the standard deviation σ; a measure of uncertainty.                                                      |

# Answer Key (all sections)

## Section 1

**Activity 1.A.** f = 1,920 ÷ tan(45°) = 1,920 pixels. Width in image = 1,920 × 1.8 ÷ 100 = 34.56, so about **34.6 pixels**.

**Activity 1.B.** Distance = (3 × 10⁸ × 1.0 × 10⁻⁶) ÷ 2 = **150 m**.

**Activity 1.C.** Distance = 343 × 0.0117 ÷ 2 ≈ **2.0 m**.

**Activity 1.D.** v = (15,400 × 3 × 10⁸) ÷ (2 × 77 × 10⁹) = **30 m/s** (108 km/h).

**Activity 1.E.** Steps per rotation = 360 ÷ 0.4 = 900. Points per frame = 64 × 900 = **57,600**. Data per frame = 57,600 × 16 = 921,600 bytes ≈ **0.92 MB** (about 9.2 MB/s at 10 Hz).

**Activity 1.F.** (a) Ultrasonic: short range, cheap, sees low obstacles below camera view. (b) Camera: only it reads text and colour. (c) Radar: unaffected by fog and measures speed; LiDAR and camera degrade in fog. (A caution: radar can struggle to separate a stationary vehicle from roadside clutter, which is another reason for fusion.) (d) LiDAR: measures 3-D shape directly and works in darkness.

## Section 2

**Activity 2.A.** Frame = 1,280 × 960 × 10 ÷ 8 = 1,536,000 bytes. Per camera = 1,536,000 × 30 = **46.08 MB/s**. Six cameras = **276.48 MB/s** (about 0.28 GB/s).

**Activity 2.B.** v = 20 m/s. Blind distance = 20 × 0.15 = 3.0 m. Braking distance = 20² ÷ (2 × 5) = 40 m. Total = **43.0 m**.

**Activity 2.C.** New total = 102 − 30 + 60 = **132 ms**, which misses the 100 ms target by **32 ms**. Possible fixes (any two): quantise the network to INT8, use a smaller or pruned network, reduce input resolution, fuse operators, ensure all operations are supported on the NPU so none fall back to the CPU, or run stages as an overlapping pipeline.

**Activity 2.D.** 60 × 10⁹ × 12 × 30 = 21.6 × 10¹² = **21.6 TFLOP/s** sustained. At 30% utilisation the peak needed is 21.6 ÷ 0.3 = **72 TFLOP/s**.

**Activity 2.E.** 0.015 s × 30 m/s = **0.45 m**.

## Section 3

**Activity 3.A.** Weights: LiDAR 1 ÷ 0.01 = 100; camera 1 ÷ 1 = 1. Fused = (100 × 30.0 + 1 × 31.0) ÷ 101 = 3,031 ÷ 101 ≈ **30.01 m**. σ = √(1 ÷ 101) ≈ **0.10 m**. The precise LiDAR dominates.

**Activity 3.B.** Predict: P = 2.0 + 0.5 = 2.5. K = 2.5 ÷ (2.5 + 1.5) = 0.625. x = 20.0 + 0.625 × (21.0 − 20.0) = **20.63**. P = (1 − 0.625) × 2.5 = **0.94**.

**Activity 3.C.** Z = 1,200 × 0.5 ÷ 12 = **50.0 m**. With 11 pixels: 600 ÷ 11 ≈ **54.5 m**. A one-pixel error moves the estimate by 4.5 m.

**Activity 3.D.** Each box is 5 × 2 = 10 m². Overlap = 4 × 2 = 8 m². Union = 10 + 10 − 8 = 12 m². IoU = 8 ÷ 12 ≈ **0.67**.

**Activity 3.E.** Precision = 72 ÷ 80 = **0.90**. Real objects = 72 + 8 = 80, so recall = 72 ÷ 80 = **0.90**.

**Activity 3.F.** 80 × tan(0.5°) ≈ **0.70 m**.

**Activity 3.G.** Fog scatters both visible light and laser light, so a camera and a LiDAR are degraded by the same cause and their errors are correlated; fusing them adds little independent information. Radio waves pass through fog far better, so a radar's error is largely independent of the camera's, and fusion helps much more.

## Section 4

**Activity 4.A.** v = 10 m/s. Blind distance = 10 × 0.2 = 2.0 m. Braking distance = 100 ÷ 12 ≈ 8.33 m. Total ≈ **10.3 m**.

**Activity 4.B.** Total needed = 2.08 + 13.78 ≈ **15.9 m**. At that detection distance the car stops exactly at the obstacle; farther than that it stops with a margin.

**Activity 4.C.** Brake when (1 − p) × 2 \< p × 600, i.e. p \> 2 ÷ 602 ≈ **0.33%**.

**Activity 4.D.** Independent: 0.02 × 0.02 = 0.0004 = **0.04%**. Perfectly correlated: **2%**, the same as one channel.

**Activity 4.E.** Advantages (any two): less information lost between modules; the whole system is optimised for one goal; fewer hand-written rules. Disadvantages (any two): hard to debug because behaviour is hidden in the weights; hard to certify; needs very large amounts of data; limited interpretability.

**Activity 4.F.** ISO 26262 covers hazards caused by faults, for example a failing brake controller or a corrupted memory value. SOTIF covers hazards where nothing is broken but the intended function reaches its limits, for example a camera blinded by low-sun glare, or a network misclassifying an unusual object.

## Section 5

**Question 1.** Depth from a camera is inferred. For stereo, depth error ≈ Z² ÷ (f × B), so it grows with the square of range, about 8.3 m per pixel of disparity error at 50 m in Worked example 3.C. A LiDAR measures range directly with centimetre-level error. Tesla compensates with learned depth, fusion of all eight cameras in a bird's-eye-view space, use of several frames over time (a moving baseline), and training on very large fleet data. These give good estimates but not measurements.

**Question 2.** At batch size 1 on a continuous stream, a network has low arithmetic intensity, which places it on the sloping, memory-limited part of the roofline. There, attainable performance = intensity × bandwidth, so one-eighth of the bandwidth means about one-eighth of the performance, however many TOPS the chip has. Larger models therefore cannot be fed fast enough on the older computer.

**Question 3.** 128 × 128 = 16,384 MACs. 16,384 × 2 × 1.5 × 10⁹ = 4.9152 × 10¹³ ≈ **49.2 TOPS** per accelerator.

**Question 4.** Any four of: Who measured it, and is it independent? What road types, weather and times of day are included, compared with the human baseline? How many miles of exposure, and what is the statistical uncertainty? Was a human supervising and intervening? How are collision and intervention defined? Is the comparison between the same kinds of vehicle and driver? Is the raw data available for audit?

**Question 5.** Benefit: lower cost, less wiring, fewer parts to fail, and a simpler supply chain. Risk: the short-range geometry that ultrasonic sensors measured directly must now be inferred from cameras, so one blocked, dirty, dark or glare-affected camera view removes information that no other sensor can supply. All the remaining sensing shares a single modality, a common-cause failure (Section 4.5).

**Question 6.** Level 2 (partial automation). The human must supervise at all times, so driver monitoring and clear hand-over rules are essential. A separate driverless robotaxi service is classified differently, and its classification should be taken from regulators and independent assessment, not from the product name.

**Question 7.** Three examples (others are acceptable): (1) Heavy fog or rain: radar penetrates it and measures speed, while cameras degrade (Table 3.1). (2) Night driving with a dark, unlit obstacle: LiDAR supplies direct range in darkness, whereas at 150 m a 0.5 m object is only about 5.5 pixels wide (Worked example 1.A). (3) Low-sun glare or a tunnel exit: dynamic range limits blind a camera for a moment, and the 100 ms latency is 2.8 m at 100 km/h (Worked example 2.B).

# References (all sections)

All web sources were accessed in early November 2026.

**Books, standards and academic papers**

- Bojarski, M., et al. (2016). *End to end learning for self-driving cars*. arXiv:1604.07316.
- Caesar, H., et al. (2020). nuScenes: A multimodal dataset for autonomous driving. *Proceedings of CVPR 2020*.
- Feng, D., et al. (2021). Deep multi-modal object detection and semantic segmentation for autonomous driving: Datasets, methods, and challenges. *IEEE Transactions on Intelligent Transportation Systems, 22*(3), 1341–1360.
- Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. *Proceedings of CVPR 2012*.
- Hall, D. L., & Llinas, J. (1997). An introduction to multisensor data fusion. *Proceedings of the IEEE, 85*(1), 6–23.
- Hu, Y., et al. (2023). Planning-oriented autonomous driving. *Proceedings of CVPR 2023*.
- ISO. (2018). *ISO 26262: Road vehicles — Functional safety*. International Organization for Standardization.
- ISO. (2022). *ISO 21448: Road vehicles — Safety of the intended functionality*. International Organization for Standardization.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering, 82*(1), 35–45.
- Li, Z., et al. (2022). BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers. *Proceedings of ECCV 2022*.
- Liu, Z., et al. (2023). BEVFusion: Multi-task multi-sensor fusion with unified bird's-eye view representation. *Proceedings of ICRA 2023*.
- Philion, J., & Fidler, S. (2020). Lift, Splat, Shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3D. *Proceedings of ECCV 2020*.
- Rosique, F., Navarro, P. J., Fernández, C., & Padilla, A. (2019). A systematic review of perception system and simulators for autonomous vehicles research. *Sensors, 19*(3), 648.
- SAE International. (2021). *J3016: Taxonomy and definitions for terms related to driving automation systems for on-road motor vehicles*.
- Sun, P., et al. (2020). Scalability in perception for autonomous driving: Waymo Open Dataset. *Proceedings of CVPR 2020*.
- Talpes, E., et al. (2020). Compute solution for Tesla's Full Self-Driving computer. *IEEE Micro, 40*(2), 25–35.
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Wang, Y., et al. (2019). Pseudo-LiDAR from visual depth estimation: Bridging the gap in 3D object detection for autonomous driving. *Proceedings of CVPR 2019*.
- Winner, H., Hakuli, S., Lotz, F., & Singer, C. (Eds.). (2016). *Handbook of Driver Assistance Systems*. Springer.
- Hennessy, J. L., & Patterson, D. A. (2019). A new golden age for computer architecture. *Communications of the ACM, 62*(2), 48–60.
- Horowitz, M. (2014). Computing's energy problem (and what we can do about it). *IEEE ISSCC*.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM, 52*(4), 65–76.
- Tesla. (2019). *Autonomy Day* \[investor presentation\].
- Tesla. (2021, 2022). *Tesla AI Day* \[presentations\]. Karpathy, A. (2021). Keynote, *CVPR Workshop on Autonomous Driving*.
- Tesla Support. (2021). *Transitioning to Tesla Vision*. tesla.com/support.
- Tesla Support. (2022). *Tesla Vision Update: Replacing Ultrasonic Sensors with Tesla Vision*. tesla.com/support.
- TechCrunch. (2022, October). *Tesla is now building Model 3 and Model Y vehicles without ultrasonic sensors*. techcrunch.com.
- Electrek. (2022, October). *Tesla announces it's moving away from ultrasonic sensors in favor of 'Tesla Vision'*. electrek.co.
- Waymo. (2024, August 19). *Meet the 6th-generation Waymo Driver*. https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/
- Electrek. (2026, February 12). *Waymo begins fully autonomous ops with 6th-gen Driver, targets 1M weekly rides*. https://electrek.co/2026/02/12/waymo-begins-fully-autonomous-ops-with-6th-gen-driver-targets-1m-weekly-rides/
- Teslarati. (2026). Coverage of Tesla's Q1 2026 earnings call on HW3 memory bandwidth. teslarati.com.
- Not a Tesla App. (2026). Coverage of Tesla AI5 chip timing. notateslaapp.com.
- Elluswamy, A. (2026, April 27). Post on X on camera-only FSD, as reported by Basenor. *Company claim; not independently verified.*
- National Highway Traffic Safety Administration. (n.d.). *Investigations and recalls*. nhtsa.gov.
