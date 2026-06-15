#!/usr/bin/env python3
"""Draw bounding boxes on all blur test images and save them."""
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

# Initialize PaddleOCR
print("Loading PaddleOCR...")
ocr = PaddleOCR(lang='en', device='gpu')
print("Ready!\n")

test_dir = "./test_images"
output_dir = "./blur_test_results"
os.makedirs(output_dir, exist_ok=True)

# Get all test images
images = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])

print("Processing images...\n")

for img_name in images:
    img_path = os.path.join(test_dir, img_name)
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Make image larger for visibility (scale 3x)
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    
    # Run OCR
    result = ocr.predict(img_path,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
    
    # Extract expected text
    expected = img_name.rsplit('_', 1)[-1].replace('.png', '')
    
    # Draw results
    detected_text = ""
    confidence = 0.0
    
    if result and len(result) > 0:
        item = result[0]
        if 'rec_texts' in item and item['rec_texts']:
            detected_text = item['rec_texts'][0]
            confidence = item['rec_scores'][0] if item['rec_scores'] else 0.0
            
            # Draw bounding box (scaled 3x)
            if 'rec_polys' in item and item['rec_polys']:
                poly = item['rec_polys'][0] * 3  # Scale polygon
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    # Add text overlay
    status = "✓ CORRECT" if detected_text.lower() == expected.lower() else "✗ WRONG"
    if not detected_text:
        detected_text = "NO TEXT DETECTED"
        status = "✗ FAILED"
    
    # Draw info box at top
    cv2.rectangle(img, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(img, f"Expected: {expected}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Got: {detected_text} ({confidence:.0%})", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if status.startswith("✓") else (0, 0, 255), 1)
    cv2.putText(img, status, (img.shape[1] - 100, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if status.startswith("✓") else (0, 0, 255), 1)
    
    # Save result
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, img)
    print(f"  {img_name}: {detected_text} ({confidence:.0%})")

# Create a combined comparison image
print("\nCreating combined comparison image...")

# Load all result images and create grid
blur_categories = ["Clear_(no_blur)", "Light_blur", "Medium_blur", "Heavy_blur", "Extreme_blur"]
texts = ["AND", "FOR", "THE"]

grid_rows = []
for category in blur_categories:
    row_imgs = []
    for text in texts:
        img_path = os.path.join(output_dir, f"{category}_{text}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            row_imgs.append(img)
    if row_imgs:
        # Resize all to same height
        min_h = min(img.shape[0] for img in row_imgs)
        row_imgs = [cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h)) for img in row_imgs]
        grid_rows.append(np.hstack(row_imgs))

if grid_rows:
    # Resize all rows to same width
    max_w = max(row.shape[1] for row in grid_rows)
    grid_rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)) 
                 for row in grid_rows]
    combined = np.vstack(grid_rows)
    cv2.imwrite(os.path.join(output_dir, "combined_comparison.jpg"), combined)
    print(f"\n Combined image saved to: {output_dir}/combined_comparison.jpg")

print(f"\n All results saved to: {output_dir}/")
