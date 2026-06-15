# OCR Pipeline Architectures & Fine-Tuning Documentation

## Overview

We evaluate **two modular OCR pipelines** for text-centric VQA under blur degradation:

1. **EasyOCR + TrOCR + Qwen2-VL**
2. **PaddleOCR + Qwen2-VL**

Both pipelines share the same VLM reasoning module (Qwen2-VL-2B) to ensure fair comparison.

---

## Pipeline 1: EasyOCR + TrOCR + Qwen2-VL

### Stage 1: Text Detection (CRAFT)

| Component | Details |
|-----------|---------|
| Model | CRAFT (Character Region Awareness for Text Detection) |
| Backbone | VGG16-BN (pretrained on ImageNet) |
| Output | Region Score Map + Affinity Score Map |
| Parameters | ~20M |

**How it works:** Instead of predicting bounding boxes directly, CRAFT predicts:

- **Region Score:** Probability that each pixel is the center of a character
- **Affinity Score:** Probability that each pixel is the space between two characters

This allows detection of curved/diagonal text that box detectors would fragment.

**Fine-tuned:**  No (we use pretrained CRAFT)
**Reason:** Detection is robust enough; recognition is the bottleneck under blur.

---

### Stage 2: Text Recognition (TrOCR)

| Component | Details |
|-----------|---------|
| Model | TrOCR (Transformer OCR) |
| Encoder | DeiT-base ViT (12 layers, 768-dim) |
| Decoder | RoBERTa-based autoregressive decoder |
| Input Size | 3 × 32 × 128 |
| Parameters | ~334M |
| Base Weights | `microsoft/trocr-base-stage1` |

**How it works:** Treats OCR as sequence-to-sequence translation (Image → Text), similar to machine translation.

### What We Fine-Tuned

| Layer/Component | Status | Why |
|----------------|--------|-----|
| ViT Encoder (all 12 layers) |  Frozen | Preserve pretrained visual features |
| RoBERTa Decoder - Self-Attention |  Fine-tuned | Learn blur-specific language patterns |
| RoBERTa Decoder - Cross-Attention |  Fine-tuned | Better image-text alignment under blur |
| RoBERTa Decoder - Feed Forward |  Fine-tuned | Adapt to degraded inputs |
| Token Embeddings |  Fine-tuned | Learn new character representations |

**Training Details:**

- Dataset: Custom blur-augmented data
- Augmentations: Gaussian blur (σ=1-5), Motion blur (5-15px), Downscale-upscale (2-4×)
- Epochs: 30
- Learning Rate: 5e-5
- Optimizer: AdamW

---

### Stage 3: VLM Reasoning (Qwen2-VL-2B)

| Component | Details |
|-----------|---------|
| Vision Encoder | ViT-based (576M params) |
| Language Model | Qwen2 decoder (2B params) |
| Inference | 4-bit quantization + Flash Attention 2 |

**Fine-tuned:**  No (used as-is)
**Reason:** Keeping VLM constant ensures differences in VQA accuracy come from OCR quality, not reasoning.

---

## Pipeline 2: PaddleOCR + Qwen2-VL

### Stage 1: Text Detection (DB++ / DBNet)

| Component | Details |
|-----------|---------|
| Model | DB++ (Differentiable Binarization) |
| Backbone | MobileNetV3 (3.4M params) |
| Neck | DBFPN (96-channel feature fusion) |
| Head | DBHead (probability + threshold maps) |
| Binarization | k=50 (amplification factor) |

**How it works:** Performs semantic segmentation, predicting probability map for text vs background. Uses adaptive threshold map learned during training.

**Fine-tuned:**  No (pretrained weights)

---

### Stage 2: Text Recognition (SVTR / PP-OCRv4)

| Component | Details |
|-----------|---------|
| Model | SVTR (Scene-text Visual Transformer with Spatial Mixing) |
| Backbone | PPLCNetV3 (scale=0.95) |
| Neck | SVTR Transformer (dims=120, depth=2) |
| Head | MultiHead (parallel CTC + NRTR) |
| Input Size | 3 × 48 × 320 |
| Parameters | ~12M |

### What We Fine-Tuned

| Layer/Component | Status | Why |
|----------------|--------|-----|
| PPLCNetV3 Backbone (all conv blocks) |  Fine-tuned | Learn blur-robust visual features |
| SVTR Neck - Self-Attention |  Fine-tuned | Better spatial mixing under degradation |
| CTC Head |  Fine-tuned | Adapt to blurry character sequences |
| NRTR Head |  Fine-tuned | Improved attention-based decoding |

**Training Strategy: Curriculum Learning (3 Stages)**

| Stage | Data | Learning Rate | Epochs |
|-------|------|--------------|--------|
| Stage 0 | Clear text only | 1e-3 | 10 |
| Stage 1 | Clear + Light blur | 2e-4 | 10 |
| Stage 2 | All + Heavy blur | 4e-5 | 10 |

**Why Curriculum Learning?**

- Progressive exposure prevents catastrophic forgetting
- Model first learns clean features, then adapts to blur
- Decreasing LR stabilizes training as blur increases

---

### Stage 3: VLM Reasoning (Qwen2-VL-2B)

Same as Pipeline 1 - kept constant for fair comparison.

---

## Summary: What Changed During Fine-Tuning

### Pipeline 1 (EasyOCR + TrOCR)

| Component | Pretrained | Fine-tuned |
|-----------|------------|------------|
| CRAFT Detector |  Used |  Not modified |
| TrOCR Encoder (ViT) |  Frozen |  Not modified |
| TrOCR Decoder |  Base weights |  **FINE-TUNED** |
| Qwen2-VL |  Used |  Not modified |

### Pipeline 2 (PaddleOCR)

| Component | Pretrained | Fine-tuned |
|-----------|------------|------------|
| DB++ Detector |  Used |  Not modified |
| PPLCNetV3 Backbone |  Base weights |  **FINE-TUNED** |
| SVTR Neck |  Base weights |  **FINE-TUNED** |
| CTC/NRTR Heads |  Base weights |  **FINE-TUNED** |
| Qwen2-VL |  Used |  Not modified |

---

## Results Summary

| Pipeline | Pretrained Acc | Fine-tuned Acc | Improvement |
|----------|----------------|----------------|-------------|
| EasyOCR + TrOCR + Qwen | 28.0% | 36.0% | **+8.0%** |
| PaddleOCR + Qwen | 32.5% | 40.1% | **+7.6%** |
| Qwen2-VL (End-to-End) | 38.0% | -- | -- |

**Key Finding:** Fine-tuning the recognition module on blur-degraded data yields 7-8% accuracy improvement, with PaddleOCR + Qwen achieving the best overall performance.
