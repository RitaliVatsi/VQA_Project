# Understanding Ablation Studies 

## 1. What is an Ablation Study?

An **Ablation Study** is like a scientific "dissection" of your AI model. The term comes from biology (removing part of an organ to see what happens). In AI, it means **removing or changing one component at a time** to understand its contribution to the final result.

**Goal:** To answer "Does this specific part actually matter, or is it useless?"

### Example

Imagine you built a **Super Race Car**.
To prove why it's fast, you do an ablation study:

1. **Remove the Spoiler:** Does it slow down? (Tests aerodynamics)
2. **Change the Tyres (Slicks vs Regular):** Does it lose grip? (Tests friction)
3. **Swap the Engine:** Does it lose power? (Tests horsepower)

If removing the spoiler makes no difference, then the spoiler is "useless bloat". If removing it causes a crash, it's "critical".

---

## 2. How to Conduct It?

The Golden Rule: **Change ONLY ONE thing at a time.**

1. **Define the Baseline:** The simplest version of your system (e.g., Pretrained model with no extras).
2. **Define the "Full Method":** Your final, best proposed system.
3. **Create Variations:**
    * **Method A:** Baseline + Feature X
    * **Method B:** Baseline + Feature Y (if valid)
    * **Method C:** Baseline + Feature X + Feature Y (Full Method)
4. **Evaluate:** Run *all* variations on the *same* dataset with the *same* metrics.

---

## 3. Our Plan for YOUR Paper 

We are building a **Robust OCR Pipeline** for low-quality text.
Your pipeline has 3 main upgrades over a standard one:

1. **Fine-tuned OCR** (instead of off-the-shelf)
2. **Super-Resolution (SwinIR)** (to fix blurry text)
3. **Fine-tuned SwinIR** (specialized for text, not just generic images)

We will remove each upgrade to prove they are necessary.

### The Experiments (The "Table" for your paper)

| Experiment ID | Configuration | Components | Why we do this? |
| :--- | :--- | :--- | :--- |
| **Exp 1 (Baseline)** | **Pretrained OCR** | Standard PaddleOCR | Reference point. "How bad is the default?" |
| **Exp 2** | **Fine-tuned OCR** | Your Trained PaddleOCR | Proves your OCR training worked. |
| **Exp 3** | **Pretrained SwinIR + FT OCR** | Generic SwinIR + Your OCR | Proves that *any* Super-Resolution helps. |
| **Exp 4 (Ours)** | **Fine-tuned SwinIR + FT OCR** | Your Text-SwinIR + Your OCR | **The Grand Finale.** Proves your specialized SwinIR is best. |

### What the results will likely show

* **Exp 1 -> Exp 2:** Accuracy jumps (e.g., 40% -> 43%). **Conclusion:** "Fine-tuning OCR is crucial."
* **Exp 2 -> Exp 3:** Accuracy improves slightly (e.g., 43% -> 44%). **Conclusion:** "Generic SR helps a bit."
* **Exp 3 -> Exp 4:** Accuracy jumps again (e.g., 44% -> 46%). **Conclusion:** "Our Fine-tuned SwinIR is a game changer."

---

## 4. Implementation Plan

We will create a script `eval_ablation_swinir.py` that can switch between models.

**Experiment Checklist:**
* [x] **Exp 1:** Done (40.10%)
* [x] **Exp 2:** Done (42.83%)
* [ ] **Exp 3:** Run pipeline with `swinir_medium_pretrained_bsrgan.pth` (Generic)
* [ ] **Exp 4:** Run pipeline with `swinir_medium_textzoom_curriculum.pth` (Your Fine-tuned)

This structure is widely accepted in top CVPR/ICCV/ECCV papers. It tells a complete scientific story.
