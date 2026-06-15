#!/usr/bin/env python3
"""
Test PaddleOCR on BLURRY PARAGRAPHS - Base Performance Demo.
Shows detection rate drops significantly on blurry text.
Use this to show your friends/mentor why fine-tuning is needed!
"""
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from datetime import datetime

# Create output directory
OUTPUT_DIR = "./blur_paragraph_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_paragraph():
    """Create a test paragraph image."""
    # Create white background
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    
    # Add text lines
    lines = [
        "PaddleOCR Detection Test",
        "This is a sample paragraph",
        "to test OCR on blurry text.",
        "Line 4: Testing detection",
        "Line 5: More sample text"
    ]
    
    y = 40
    for line in lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 50
    
    return img

def apply_blur(img, blur_type, intensity):
    """Apply different blur types."""
    if blur_type == "gaussian":
        ksize = intensity * 2 + 1
        return cv2.GaussianBlur(img, (ksize, ksize), intensity)
    elif blur_type == "motion":
        kernel = np.zeros((intensity, intensity))
        kernel[intensity // 2, :] = 1.0 / intensity
        return cv2.filter2D(img, -1, kernel)
    elif blur_type == "downscale":
        h, w = img.shape[:2]
        scale = max(0.1, 1.0 - intensity * 0.15)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        return cv2.resize(small, (w, h))
    return img

def count_detections(result):
    """Count number of text regions detected."""
    if not result or len(result) == 0:
        return 0
    item = result[0]
    if 'rec_texts' in item:
        return len(item['rec_texts'])
    return 0

def main():
    print("=" * 70)
    print(" PaddleOCR BLUR PERFORMANCE TEST - Paragraph Detection")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: Show base model limitations on blurry text")
    print("=" * 70)
    
    # Initialize PaddleOCR
    print("\n Loading PaddleOCR (PP-OCRv5)...")
    ocr = PaddleOCR(lang='en', device='gpu')
    print("Ready!\n")
    
    # Create test image
    print("️ Creating test paragraph...")
    original = create_test_paragraph()
    cv2.imwrite(f"{OUTPUT_DIR}/00_original.jpg", original)
    
    # Test different blur levels
    blur_configs = [
        ("Clear (Original)", None, 0),
        ("Light Gaussian Blur", "gaussian", 2),
        ("Medium Gaussian Blur", "gaussian", 5),
        ("Heavy Gaussian Blur", "gaussian", 8),
        ("Light Motion Blur", "motion", 5),
        ("Heavy Motion Blur", "motion", 15),
        ("Light Downscale (50%)", "downscale", 3),
        ("Heavy Downscale (25%)", "downscale", 5),
    ]
    
    results = []
    print("\n" + "=" * 70)
    print(" DETECTION RESULTS")
    print("=" * 70)
    print(f"{'Condition':<30} {'Text Regions Detected':<25} {'Status'}")
    print("-" * 70)
    
    for i, (name, blur_type, intensity) in enumerate(blur_configs):
        # Apply blur
        if blur_type:
            img = apply_blur(original.copy(), blur_type, intensity)
        else:
            img = original.copy()
        
        # Save blurred image
        cv2.imwrite(f"{OUTPUT_DIR}/{i:02d}_{name.replace(' ', '_')}.jpg", img)
        
        # Run OCR
        result = ocr.predict(img,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False)
        
        detected = count_detections(result)
        expected = 5  # We have 5 lines
        
        status = " Good" if detected >= 4 else " Low" if detected >= 2 else " Failed"
        results.append((name, detected, expected, status))
        
        print(f"{name:<30} {detected}/{expected:<23} {status}")
    
    # Create comparison grid
    print("\n Creating comparison image...")
    
    rows = []
    for i, (name, blur_type, intensity) in enumerate(blur_configs):
        if blur_type:
            img = apply_blur(original.copy(), blur_type, intensity)
        else:
            img = original.copy()
        
        # Add label
        label_bar = np.ones((30, img.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(label_bar, name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        labeled = np.vstack([label_bar, img])
        rows.append(labeled)
    
    # Arrange in 2 columns
    col1 = np.vstack(rows[:4])
    col2 = np.vstack(rows[4:])
    
    # Pad to same height
    max_h = max(col1.shape[0], col2.shape[0])
    col1 = cv2.copyMakeBorder(col1, 0, max_h - col1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    col2 = cv2.copyMakeBorder(col2, 0, max_h - col2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    comparison = np.hstack([col1, col2])
    cv2.imwrite(f"{OUTPUT_DIR}/comparison_grid.jpg", comparison)
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY FOR YOUR PRESENTATION")
    print("=" * 70)
    
    clear_score = results[0][1]
    blur_scores = [r[1] for r in results[1:]]
    avg_blur = sum(blur_scores) / len(blur_scores)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │  PaddleOCR (PP-OCRv5) Base Performance                     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   Clear Text Detection:    {clear_score}/5 lines ({clear_score/5*100:.0f}%)             │
    │   Blurry Text Detection:   {avg_blur:.1f}/5 lines ({avg_blur/5*100:.0f}%)             │
    │                                                             │
    │   Performance Drop:        {(clear_score - avg_blur)/clear_score*100:.0f}% on blur                     │
    │                                                             │
    │   ➜ SOLUTION: Fine-tune with 50/50 clear+blur data         │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    print(f" Images saved to: {OUTPUT_DIR}/")
    print(f" Comparison grid: {OUTPUT_DIR}/comparison_grid.jpg")
    print("\n Use these images in your presentation to show the problem!")

if __name__ == "__main__":
    main()
