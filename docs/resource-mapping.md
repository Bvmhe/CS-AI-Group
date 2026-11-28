# Resource Mapping — Final (Stage 3)

Team Fengwei Fried Rice · Hardware, Networks & Behaviourist AI · updated 28 November 2026

This table shows where each topic of the final Markdown content (`Main/`) is used in the team's other learning resources. The Markdown is the checked source; the resources present the same ideas in their own way.

**How the numbering lines up.** "Recipe 0X §X.Y" in the Study Guide, Test Bank and slide notes is the same topic as "Module 0X, Section Y" in the Markdown. For example, Recipe 01 §1.3 (Amdahl's law) is Module 01, Section 3.

**Where the resources are**

| Resource | File / location | Submitted under |
|---|---|---|
| Markdown content (source) | `Main/00-Module-Overview.md`, `Main/Chapter-01…05/`; PDF copy `Main/FengweiFriedRice_Stage3_FinalMarkdown.pdf` | This assignment (Final Markdown & Repository) |
| Final Study Guide (168 pages) | `Stage3-Final/FengweiFriedRice_Stage3_FinalStudyGuide.pdf` | Study Guide and Test Bank assignment |
| Final Test Bank (60 questions) | `Stage3-Final/FengweiFriedRice_Stage3_FinalTestBank.docx` / `.pdf`; game version `Stage3-Final/Supplements/Mamas_Quiz_Kitchen.html` | Study Guide and Test Bank assignment |
| PPT slides (171 slides with speaker notes) | `FengweiFriedRice_Stage2_PPT_V1_with_notes.pptx` (final version; the file name was kept from Stage 2) | PPT Slides assignment |
| Video tutorials | Recorded by the team (Video Lead 李秀春); large video files are not stored in GitHub | Video Tutorials assignment |
| Curated external videos | 16 YouTube videos listed on the "Watch & Learn" pages of each Recipe in the Study Guide | Part of the Study Guide |

**Legend:** — = not selected for that medium (covered in the Markdown and Study Guide only) · TB Q = Test Bank question numbers · slides = slide numbers whose speaker notes reference that section.

## Module 01 — AI Hardware Architecture

| Markdown section | Study Guide | Test Bank | Slides | Videos (guide "Watch & Learn") |
|---|---|---|---|---|
| 1 Definition and fundamental concepts (MACs, training vs inference) | Recipe 01 §1.1 | Q1, Q26 | 6–11 | 3Blue1Brown, *But what is a neural network?* |
| 2 CPU vs GPU vs TPU (systolic arrays, NPU) | Recipe 01 §1.2 | Q2, Q27, Q28 | 12–19 | Branch Education (GPU); Google Cloud (TPU) |
| 3 Parallel processing and acceleration (Amdahl's law, quantisation) | Recipe 01 §1.3 | Q3 | 13–24 | — |
| 4 Memory hierarchy and efficiency (roofline) | Recipe 01 §1.4 | Q4, Q29, Q51 | 25–28 | NHR@FAU, *Roofline model* |
| 5 Case study: Apple Neural Engine | Recipe 01 §1.5 | Q52 | 29–35 | — |

## Module 02 — Computer Networks & AI Infrastructure

| Markdown section | Study Guide | Test Bank | Slides | Videos (guide "Watch & Learn") |
|---|---|---|---|---|
| 1 Network architecture and protocols | Recipe 02 §2.1 | Q6 | 37–45 | Lesics, *How does the internet work?* |
| 2 TCP/IP and data transmission | Recipe 02 §2.2 | Q7, Q31, Q32 | 46–52 | Lesics, *How does the internet work?* |
| 3 Cloud vs edge computing | Recipe 02 §2.3 | Q33, Q53 | 53–59 | Google, *Inside a Google data center* |
| 4 Latency, bandwidth and scalability | Recipe 02 §2.4 | Q8, Q9, Q34, Q54 | 60–63 | internet-class, *What is bandwidth?* |
| 5 Case study: 5G and cloud AI service | Recipe 02 §2.5 | Q35 | 64–68 | — |

## Module 03 — Behaviourist AI & Reinforcement Learning

| Markdown section | Study Guide | Test Bank | Slides | Videos (guide "Watch & Learn") |
|---|---|---|---|---|
| 1 Definition and principles of behaviourist AI | Recipe 03 §3.1 | (Lab questions) | 70–74 | — |
| 2 Agent–environment interaction (return, value) | Recipe 03 §3.2 | Q11, Q12, Q36 | 75–85 | Steve Brunton, *Reinforcement Learning*; Computerphile, *Markov Decision Processes* |
| 3 Reward systems and Q-learning | Recipe 03 §3.3 | Q13, Q37, Q38, Q55 | 81–94 | Steve Brunton, *Reinforcement Learning* |
| 4 Exploration vs exploitation (bandits, ε-greedy, UCB) | Recipe 03 §3.4 | Q14, Q39 | 95–106 | — |
| 5 Case study: robot dogs learning to walk | Recipe 03 §3.5 | Q40 | 107–117 | AI Warehouse, *AI Learns to Walk* |
| Game: Pup's Pantry Maze (Q-learning made playable) | Bonus Kitchen Lab, p145 | — | — | — |

## Module 04 — AI Sensors & Intelligent Systems

| Markdown section | Study Guide | Test Bank | Slides | Videos (guide "Watch & Learn") |
|---|---|---|---|---|
| 1 Types of sensors (camera, LiDAR, ultrasonic, radar) | Recipe 04 §4.1 | Q16, Q17, Q18, Q41 | 120–127 | TED-Ed, *How do self-driving cars "see"?*; NEON Science, *How does LiDAR work?* |
| 2 Data acquisition and real-time processing | Recipe 04 §4.2 | Q42 | 128–131 | — |
| 3 Sensor fusion (Kalman filter, IoU) | Recipe 04 §4.3 | Q43, Q44 | 132–136 | Visually Explained, *Kalman Filters* |
| 4 Hardware and AI decision-making | Recipe 04 §4.4 | Q19, Q45, Q57 | 137–140 | — |
| 5 Case study: Tesla autonomous driving | Recipe 04 §4.5 | — | 141–143 | Tesla, *AI Day 2021* (primary source) |

## Module 05 — AI System Integration & Optimization

| Markdown section | Study Guide | Test Bank | Slides | Videos (guide "Watch & Learn") |
|---|---|---|---|---|
| 1 Hardware–software interaction (compilers, MFU) | Recipe 05 §5.1 | Q21, Q49 | 145–147 | — |
| 2 Distributed AI systems | Recipe 05 §5.2 | Q22, Q46 | 148–152 | TensorFlow, *A friendly introduction to distributed training* |
| 3 Performance optimisation and resource allocation | Recipe 05 §5.3 | Q47 | 153–159 | — |
| 4 Energy, cost and scalability | Recipe 05 §5.4 | Q23, Q24, Q59 | 155–158 | Google Cloud, *Inside a Google Cloud TPU Data Center* |
| 5 Case study: Google TPU infrastructure | Recipe 05 §5.5 | Q48 | 160–168 | — |

## Notes and checks

- **Test Bank** question tags were read from `FengweiFriedRice_Stage3_FinalTestBank.docx`. Questions not listed above (Q5, Q10, Q15, Q20, Q25, Q30, Q50, Q56, Q58, Q60) are tagged to the Kitchen Lab activities, not to one section.
- **Slides** ranges come from the "Reference: Study Guide … Recipe 0X §X.Y" line in the speaker notes. Ranges overlap where a slide revisits an earlier section.
- **Consistency check (28 Nov 2026):** 14 worked answers in the slide notes were compared with the Markdown and all agree, for example: systolic array y = [17, 39]; Amdahl p = 0.9, N = 100 → 9.2× (ceiling 10×); corridor values 1.0 / 0.9 / 0.81 / 0.729; Q-learning target 1.45; ε-greedy 0.77; robot-dog policy 23,680 MACs; latency budgets 140 / 56 / 36 ms.
- **Known difference:** the slide speaker notes still say "Study Guide V1.0" in their reference lines. The topics and answers they point to are unchanged in the Final Study Guide.
- **Team video tutorials:** the per-section list of the team's own recorded videos is held in the Video Tutorials submission. The "Videos" column above lists the 16 curated external videos on the Study Guide's Watch & Learn pages, matched by the section each video's description refers to (from `Study-Guide/content/vids.py`).
