# SA-DBNet Model

Self-Attention Enhanced Differentiable Binarization Network for robust text detection under visual degradation.

## Architecture
- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Bottleneck:** Global Self-Attention with learnable γ scaling
- **FPN:** Deformable Convolutions (DCN-v2) at every lateral level
- **Head:** Differentiable Binarization with k=50

## Training
- Pre-trained on TextVQA pseudo-labels (21,953 images)
- Fine-tuned on ICDAR 2015 (1,000 images) at 640×640 resolution
- Optimizer: AdamW with gradient clipping
- F1-score: 59% on detection benchmark

## Weights
Model weights are not included in this repository due to file size constraints.
To reproduce, follow the training instructions in `docs/VLM_TRAINING_WORKFLOW.md`.
