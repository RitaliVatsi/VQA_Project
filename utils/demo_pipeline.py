"""
Quick Demo: Test the OCR pipeline on sample images.

Run after:
1. Download swinir_textzoom_best.pth from Kaggle
2. PaddleOCR Stage 2 training completes
"""

import os
import sys

# Add parent dir to path
sys.path.insert(0, '/home/aditya/ocr_project')

from ocr_pipeline import OCRPipeline

def demo():
    print("="*60)
    print("OCR Pipeline Demo")
    print("="*60)
    
    # Check if models exist
    sr_model = "/home/aditya/ocr_project/sr_training/swinir_textzoom_best.pth"
    rec_model = "/home/aditya/ocr_project/curriculum_str_project/models_textzoom_clean/stage2_hard/best_accuracy.pdparams"
    
    print("\nModel Status:")
    print(f"  SwinIR SR: {'✓ Found' if os.path.exists(sr_model) else '✗ Not found'}")
    print(f"  PaddleOCR Rec: {'✓ Found' if os.path.exists(rec_model) else '✗ Training...'}")
    
    if not os.path.exists(sr_model):
        print("\n⚠ Please download swinir_textzoom_best.pth from Kaggle to:")
        print(f"  {sr_model}")
        return
    
    # Create pipeline
    print("\nInitializing pipeline...")
    pipeline = OCRPipeline(use_sr=True, device='cuda')
    
    # Test images
    test_images = [
        "/home/aditya/.gemini/antigravity/brain/9ce8eeb3-426d-431e-9043-50fa0c97dc6c/uploaded_image_1768232397291.jpg",  # Menu image
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nProcessing: {os.path.basename(img_path)}")
            result = pipeline.process(img_path, save_sr=True)
            
            print("Recognized text:")
            for text in result['texts'][:10]:
                print(f"  - {text}")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    demo()
