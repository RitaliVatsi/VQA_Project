
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path
from tqdm import tqdm

def get_image_files(directory, max_files=100):
    """Recursively find images."""
    exts = ['.jpg', '.png', '.jpeg', '.bmp']
    images = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                images.append(os.path.join(root, f))
                if len(images) >= max_files:
                    return images
    return images

def eval_model(model, image_paths, name):
    print(f"\nrunning {name} on {len(image_paths)} images...")
    detected_count = 0
    total_conf = 0
    
    for img_path in tqdm(image_paths):
        # PaddleOCR handles loading
        result = model.ocr(img_path, det=True, rec=False, cls=False)
        
        # Result is [[box1, box2, ...]] or [None] (if nothing found)
        # Actually structure is [[ [ [[x,y],..], conf ], ... ]] for det+rec
        # For det only: [ [ [[x,y],..], [[x,y],..] ] ]?
        # Let's check result format safely.
        
        if result and result[0]:
            detected_count += 1
            # We don't get confidence for pure ID detection easily unless we look deep,
            # but detection success is the main metric for "can it see the blur?"
            
    accuracy = (detected_count / len(image_paths)) * 100
    print(f"  {name} Detection Rate: {accuracy:.2f}% ({detected_count}/{len(image_paths)})")
    return accuracy

def main():
    print("="*60)
    print("⚔️  MODEL COMPARISON: Base vs Fine-Tuned")
    print("="*60)

    # 1. Setup Models
    # Base: Official PP-OCRv3
    base_model = PaddleOCR(
        use_textline_orientation=False, 
        lang='en', 
        text_detection_model_dir='./en_PP-OCRv3_det_infer',
    )
    
    # Tuned: Our fine-tuned model
    tuned_model = PaddleOCR(
        use_textline_orientation=False, 
        lang='en', 
        text_detection_model_dir='./output_real_dbnet/inference',
    )
    
    # 2. Setup Datasets
    # Clear: ICDAR 2015 Test (subset)
    icdar_path = "dbnet_blur/data/icdar2015/test"
    clear_images = get_image_files(icdar_path, max_files=100)
    
    # Blurry: TextZoom Hard (subset)
    # Note: TextZoom structure might be 'TextZoom/test/hard'
    blur_path = "TextZoom/test/hard"
    if not os.path.exists(blur_path):
        blur_path = "TextZoom/test/medium" # Fallback
    
    blur_images = get_image_files(blur_path, max_files=100)
    
    if not clear_images:
        print(" No clear images found in", icdar_path)
    if not blur_images:
        print(" No blur images found in", blur_path)
        
    # 3. Evaluate
    results = {}
    
    if clear_images:
        print("\n--- 1. Testing on CLEAR Text (ICDAR 2015) ---")
        acc_base_clear = eval_model(base_model, clear_images, "Base Model")
        acc_tuned_clear = eval_model(tuned_model, clear_images, "Fine-Tuned Model")
        results['clear_base'] = acc_base_clear
        results['clear_tuned'] = acc_tuned_clear
        
    if blur_images:
        print("\n--- 2. Testing on BLURRY Text (TextZoom) ---")
        acc_base_blur = eval_model(base_model, blur_images, "Base Model")
        acc_tuned_blur = eval_model(tuned_model, blur_images, "Fine-Tuned Model")
        results['blur_base'] = acc_base_blur
        results['blur_tuned'] = acc_tuned_blur
        
    print("\n" + "="*60)
    print(" FINAL SCORECARD")
    print("="*60)
    
    if 'clear_base' in results:
        diff = results['clear_tuned'] - results['clear_base']
        print(f"Clear Text Stability: {results['clear_tuned']:.1f}% vs {results['clear_base']:.1f}% ({diff:+.1f}%)")
        if diff < -5:
            print("    WARNING: Significant regression on clear text!")
        else:
            print("    Clear text performance preserved.")
            
    if 'blur_base' in results:
        diff = results['blur_tuned'] - results['blur_base']
        print(f"Blurry Text Improvement: {results['blur_tuned']:.1f}% vs {results['blur_base']:.1f}% ({diff:+.1f}%)")
        if diff > 0:
            print("    SUCCESS: Model is better at blur!")
        else:
            print("    FAIL: No improvement on blur yet.")

if __name__ == "__main__":
    main()
