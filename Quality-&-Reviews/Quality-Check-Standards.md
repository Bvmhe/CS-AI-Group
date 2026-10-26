What a Good Module Needs (Based on the Gaps in Yours)

I read through your module line by line and checked the maths and the structure. Below, each gap becomes a rule, so you can see why the rule matters.

1. Everything the module promises must actually be in it

Rule: Every learning objective needs a section that teaches it, an exercise that practises it, and an answer that checks it.

In your module:

Objective 3 promises to explain systolic arrays, but there is no section on them. Section 3.3 is missing, and Check 2.2 refers to a “2 × 2 systolic example” that never appears.
Objective 5 promises the roofline calculation, and the learner activity (4.5) appears, but the worked example that should come before it (4.4) is missing. Learners are asked to practise something they were never shown.
The module says the glossary is “at the end,” but there is no glossary, summary, answer key, or reference list.
2. Numbering must be continuous and unambiguous

Rule: No skipped numbers. Number sections, figures, tables, and exercises separately, so they never get confused.

In your module:

3.2 jumps to 3.4, and 4.3 jumps to 4.5.
Exercises are labelled “Check your understanding (2.1)”, “(2.2)”, and so on, which looks exactly like Section 2.1 and 2.2.
Only one figure is numbered (Figure 4.4), and it sits in Section 4.2. Other figures and all tables have no numbers.
3. Cross-references must point to something real

Rule: Before release, click through or search for every “see Section X” and check that the target exists.

In your module:

Section 2.5 says “Section 5 studies the Apple version,” and 2.6 mentions “case-study questions in Section 5.” But the case study isn’t numbered as Section 5, and it has no questions.
The roofline activity says “same hypothetical accelerator (10 TFLOP/s, 500 GB/s),” but the example that introduces it is gone.
4. No placeholders or internal notes in the final text

Rule: If a reader can see a note meant for the author, the module isn’t finished.

In your module:

“Alt text describing the image” appears on several images.
Figure 4.4 says “figure not included in this conversion.”
The Ironwood note says “verify against Google’s own documentation before citing.”
The cover says “V1.0 (substantive draft, Stage 1).”
5. Figures need real captions, sources, and alt text

Rule: Each figure gets a number, a caption saying what to look at, a proper source, and descriptive alt text.

In your module:

The case study images are sourced only as “tomshardware.com” and “techtime.news.” That is a website name, not a citation.
The memory hierarchy image has no caption or source at all.
6. Every calculation and code sample must be verified, including the output

Rule: Re-do every number by hand or by running it, and show the complete expected output.

In your module:

I rechecked the maths (4.2 million MACs, 1.4 ms vs 1.0 µs, the 1,365× speed-up, 640 ÷ 3.7 ≈ 173), and it all holds up. That is good.
The quantisation code prints a “max error” line, but the expected output in the module leaves it out. It should show max error: 0.0039.
7. Time-sensitive claims need a date

Rule: Anything that can go out of date (“newest,” “latest,” “current”) needs a year and a source, or should be reworded.

In your module:

“Google’s newest announced generation, Ironwood” and the 192 GB figure rely on press coverage and will age quickly. Say “as announced in 2025” and cite Google directly.
The TPU comparison is already labelled as 2015-2017 data. That’s a good habit to repeat everywhere.
8. Every section, including the case study, follows the same template

Rule: The case study is part of the module, not an add-on. It needs the same Big idea, objectives, citations, and exercises as everything else.

In your module: the Apple Neural Engine case study is the biggest gap.

It has no Big idea, objectives, exercises, or in-text citations.
It never uses the module’s own tools. There are no TOPS figures, no precision or memory numbers, and no roofline reasoning, even though the module’s goal is to “analyse… and judge its trade-offs.”
It repeats the training vs inference content from Section 1.5 without adding anything.
Its section numbers (1-7) clash with the main module’s numbering.
It switches to American spelling (“specialized”) while the rest uses British (“specialised”), and the tone is simpler and more bullet-heavy.
9. Consistent language and small polish

Rule: One spelling variant, one tone, one name per concept, and a proofread at the end.

In your module: there are also small errors like “Which of the three chips would you choose for? … Justify briefly… !”

10. Exercises need answers

Rule: Every exercise has a worked answer in an answer key, with a short explanation rather than just the final number.

In your module: none of the exercises has answers.

