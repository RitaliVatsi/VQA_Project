# PaddleOCR Fine-tuned Model

PP-OCRv4 with curriculum-trained recognition for blur robustness.

## Architecture
- **Detection:** DB++ with MobileNetV3 backbone + DBFPN (pretrained, not fine-tuned)
- **Recognition:** SVTR with PPLCNetV3 backbone (fine-tuned)
- **Heads:** Parallel CTC + NRTR (fine-tuned)

## Curriculum Training (3 Stages)
| Stage | Data | Learning Rate | Epochs |
|-------|------|:---:|:---:|
| Stage 0 | Clear text only | 1e-3 | 10 |
| Stage 1 | Clear + light blur | 2e-4 | 10 |
| Stage 2 | All + heavy blur | 4e-5 | 10 |

## Weights
Model weights (.pdparams) are not included in this repository.
See `training/train_overnight.py` and `docs/KAGGLE_2xT4_TRAINING.md` for reproduction.
