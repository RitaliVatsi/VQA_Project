#  Best Datasets for "Way High" Accuracy on Blurry Text

For the best possible performance on **unreadable/blurry text**, you need "Real World" datasets where text wasn't the main focus (Incidental) or datasets specifically built for blur.

## 1. Top Pick for DETECTION (Finding the text)
**Dataset:** **ICDAR 2015 (Challenge 4: Incidental Scene Text)**
*   **Why:** This is the "Gold Standard" for detection. It features text that appears accidentally in the background, text that is moving, or text that is out of focus. It is much harder and more realistic than synthetic data.
*   **Links:** (Requires Registration - It's quick)
    *   [Official Download Page](https://rrc.cvc.uab.es/?ch=4&com=downloads)
    *   **Download these specific files:**
        1.  `ch4_training_images.zip` (Training Images)
        2.  `ch4_training_localization_transcription_gt.zip` (Training Ground Truth)
        3.  `ch4_test_images.zip` (Test Images)
        4.  `Challenge4_Test_Task1_GT.zip` (Test Ground Truth)

**Where to put files:**
```
oocr_project/
└── dbnet_blur/
    └── data/
        └── icdar2015/
            ├── ch4_training_images/      <-- Unzip here
            ├── ch4_test_images/          <-- Unzip here
            ├── train_gt/                 <-- Unzip training GT here
            └── test_gt/                  <-- Unzip test GT here
```

---

## 2. Top Pick for RECOGNITION (Reading the text)
**Dataset:** **TextZoom (The Real-Blur Dataset)**
*   **Why:** You already have the test set, but training on the **Training Set** of TextZoom is the #1 way to make the model understand blur. It pairs blurry images with their sharp counterparts.
*   **Link:** [TextZoom GitHub](https://github.com/JasonBoy1/TextZoom) (Check their Google Drive/Baidu links)
*   **Action:** If you don't have the `train` folder in your TextZoom directory, download the full dataset.

## 3. "Secret Weapon" Dataset (Optional but Powerful)
**Dataset:** **COCO-Text**
*   **Why:** Massive scale (63k images). Contains "illegible" text labels, which helps the model learn what *not* to read or how to handle extreme cases.
*   **Link:** [COCO-Text Website](https://bgshih.github.io/cocotext/)

---

## ⚡ Quick Start Plan
1.  **Download ICDAR 2015** (Priority #1). This fixes your `train_fast.py` blocker.
2.  Extract the files as shown above.
3.  Run the preparation script:
    ```bash
    python dbnet_blur/prepare_data.py
    ```
4.  Then run your overnight training!
