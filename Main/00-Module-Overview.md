# Hardware, Networks & Behaviourist AI — Module Overview

| | |
|---|---|
| **Course** | Introduction to Artificial Intelligence |
| **Team** | Fengwei Fried Rice |
| **Version** | Final (Stage 3), 28 November 2026 |
| **Structure** | Five chapters (Modules 01–05) of five sections each. Every chapter ends with its own glossary, answer key, references and photo credits. |

Start here. This page tells you what the module covers, who it is for, what you need to know first, and in which order to read the chapters.

---

## What this module is about

Modern AI is not only clever software. Every answer an AI gives is computed on **hardware** (Module 01), often carried over a **network** (Module 02), produced by a model that may have **learned from reward and punishment** (Module 03), fed by **sensors** that measure the world (Module 04), and run as part of a much larger **system** that has to be fast, affordable and energy-efficient (Module 05).

The five chapters follow one idea from small to large: a single multiply-and-add on a chip, a request travelling across a network, an agent learning to act, a car perceiving the road, and finally a data centre with thousands of chips. Each chapter uses a running example and real case studies (Apple Neural Engine, a 5G translation service, a walking robot dog, Tesla's camera-based driving system and Google's TPU pods).

## Intended learners

- First- and second-year undergraduate students in Computer Science or Information Technology.
- Beginners: no previous course in computer architecture, networking or machine learning is assumed.
- Suitable for self-study. Allow about **1.5–2 hours per chapter**, plus time for the activities.

## Prerequisites

Before you start, you should be comfortable with:

- **Basic arithmetic**: multiplication, division, powers of ten and percentages. Every worked example shows each step.
- **Programming basics**: what a variable, a loop and a list are. You do not need to write code.
- **A rough everyday idea** of what a CPU and the internet are.

No electronics, advanced mathematics or earlier AI knowledge is needed. Each technical term is defined where it first appears and collected in the chapter glossary. Later chapters reuse terms from earlier ones (for example MAC, bandwidth, latency and batch size from Module 01), and they point back to the section where each term was taught.

## Learning outcomes

By the end of the module you will be able to:

| # | Learning outcome | Taught in |
|---|---|---|
| 1 | Name CPU, GPU, TPU/NPU and memory, and explain what each does for AI. | Module 01 |
| 2 | Explain how networks move AI data, and choose between cloud and edge using latency and bandwidth. | Module 02 |
| 3 | Describe how an agent learns from rewards, and explain exploration versus exploitation. | Module 03 |
| 4 | Compare cameras, LiDAR and ultrasonic sensors, and explain sensor fusion. | Module 04 |
| 5 | Explain how hardware, networks and software are combined and optimised at scale. | Module 05 |

These are the same five learning goals used in the Final Study Guide (Recipes 01–05), the Test Bank and the slides.

## Recommended reading order

Read the chapters in order. Each one builds on the vocabulary of the one before.

| Order | Chapter | Sections | Case study |
|---|---|---|---|
| 1 | [Module 01: AI Hardware Architecture](Chapter-01/01-AI-Hardware-Architecture.md) | 1 Definitions · 2 CPU vs GPU vs TPU · 3 Parallel processing · 4 Memory hierarchy | 5 Apple Neural Engine |
| 2 | [Module 02: Computer Networks & AI Infrastructure](Chapter-02/02-Computer-Networks-and-AI-Infrastructure.md) | 1 Architecture & protocols · 2 TCP/IP · 3 Cloud vs edge · 4 Latency, bandwidth, scalability | 5 5G cloud AI service |
| 3 | [Module 03: Behaviourist AI & Reinforcement Learning](Chapter-03/03-Behaviourist-AI-and-Reinforcement-Learning.md) | 1 Behaviourist AI · 2 Agent–environment loop · 3 Rewards & Q-learning · 4 Exploration vs exploitation | 5 Robot dogs learning to walk |
| 4 | [Module 04: AI Sensors & Intelligent Systems](Chapter-04/04-AI-Sensors-and-Intelligent-Systems.md) | 1 Cameras, LiDAR, ultrasonic · 2 Real-time data · 3 Sensor fusion · 4 Hardware + decisions | 5 Tesla autonomous driving |
| 5 | [Module 05: AI System Integration & Optimization](Chapter-05/05-AI-System-Integration-and-Optimization.md) | 1 Hardware–software · 2 Distributed AI · 3 Optimisation · 4 Energy, cost, scalability | 5 Google TPU infrastructure |

## How each chapter works

Every section follows the same pattern, so you always know what comes next:

1. **Learning objectives** for the section.
2. **Explanation** in plain language, with diagrams and real-world photos.
3. **Worked examples** that show every step of a calculation.
4. **Activities** to try yourself.
5. **Summary** of the key points.

At the end of each chapter you will find the **Glossary**, the **Answer Key** (with explanations, so you can check your own work) and the **References**.

## How this Markdown relates to the other resources

This Markdown is the shared, checked source for the team's three learning resources. They present the same ideas in different ways:

- **Final Study Guide** (PDF) and **Test Bank**: the cook-along version for independent study. "Recipe 0X §X.Y" in the guide corresponds to "Module 0X, Section Y" here.
- **PPT slides**: for classroom teaching, with speaker notes and expected answers.
- **Video tutorials**: short explanations and demonstrations.

The full resource mapping, showing where each topic is used, is in the Final Submission Report and in [`docs/resource-mapping.md`](../docs/resource-mapping.md).
