# TrOCR Fine-tuned Model

Transformer-based OCR with LoRA-adapted decoder for blur-robust text recognition.

## Architecture
- **Encoder:** DeiT-base ViT (12 layers, 768-dim) — Frozen
- **Decoder:** Autoregressive language decoder (fine-tuned via LoRA)
- **Base weights:** `microsoft/trocr-base-stage1`

## Fine-tuning Details
- ViT encoder: Frozen (preserve pretrained features)
- Decoder self-attention, cross-attention, feed-forward: Fine-tuned
- Augmentations: Gaussian blur (σ=1-5), Motion blur (5-15px), Downscale (2-4×)
- Epochs: 30 | LR: 5e-5 | Optimizer: AdamW

## Weights
Model weights are not included in this repository.
The fine-tuned LoRA adapter was trained on Kaggle using 2×T4 GPUs.
See `docs/KAGGLE_2xT4_TRAINING.md` for reproduction instructions.
