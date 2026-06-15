# Dataset

The evaluation dataset is hosted on Kaggle due to its size (~1.3 GB):

**[OCR Pipeline Testing Dataset](https://www.kaggle.com/datasets/kagglemodeltraining/ocr-pipeline-testing-dataset)**

## Contents
- **4,013 images** with text in real-world scenes
- **7,000 question-answer pairs** (`textvqa_dataset.json`)
- Mixed viewpoints, background clutter, and non-uniform illumination

## Setup

1. Download the dataset from Kaggle
2. Extract it into this `data/` directory
3. Expected structure:
   ```
   data/
   ├── textvqa_dataset.json    # Question-answer annotations
   ├── IMG_001.jpg             # Scene images
   ├── IMG_002.jpg
   └── ...
   ```

## Degradation

Visual degradation (Gaussian blur σ=5.0) is applied at runtime during evaluation. The raw images in the dataset are clean originals.
