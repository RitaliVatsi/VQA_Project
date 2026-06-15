"""
Full OCR Pipeline with Super-Resolution
=======================================

Pipeline: Blurry Image → SwinIR SR → PP-OCRv5 Detection → Trained Recognition

Usage:
    python ocr_pipeline.py --image path/to/image.jpg
    python ocr_pipeline.py --image path/to/image.jpg --save-sr  # Also save SR image
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image

# ============= Configuration =============
# Update these paths after training completes
SWINIR_MODEL_PATH = "/home/aditya/ocr_project/sr_training/swinir_textzoom_best.pth"
PADDLEOCR_REC_MODEL = "/home/aditya/ocr_project/curriculum_str_project/models_textzoom_clean/stage2_hard/best_accuracy"

# ============= SwinIR Model =============
def load_swinir_model(model_path, device='cuda'):
    """Load trained SwinIR model for super-resolution."""
    import torch
    
    # Clone SwinIR if needed
    swinir_path = "/home/aditya/ocr_project/sr_training/SwinIR"
    if not os.path.exists(swinir_path):
        os.system(f"git clone --depth 1 https://github.com/JingyunLiang/SwinIR.git {swinir_path}")
    
    sys.path.insert(0, swinir_path)
    from models.network_swinir import SwinIR
    
    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect',
        resi_connection='1conv'
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded SwinIR model from {model_path}")
    return model


def apply_super_resolution(model, image, device='cuda'):
    """Apply super-resolution to an image."""
    import torch
    from torchvision import transforms
    
    # Convert to tensor
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Apply SR
    with torch.no_grad():
        sr_tensor = model(img_tensor)
    
    # Convert back to numpy
    sr_image = sr_tensor.squeeze().cpu().numpy()
    sr_image = np.transpose(sr_image, (1, 2, 0))
    sr_image = (sr_image * 255).clip(0, 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    
    return sr_image


# ============= PaddleOCR =============
def load_paddleocr():
    """Load PaddleOCR with custom recognition model."""
    from paddleocr import PaddleOCR
    
    # Use default detection, custom recognition
    ocr = PaddleOCR(
        lang='en',
        # text_recognition_model_dir=PADDLEOCR_REC_MODEL,  # Use our trained model
    )
    
    print("✓ Loaded PaddleOCR")
    return ocr


def run_ocr(ocr, image):
    """Run OCR on an image."""
    result = ocr.predict(image)
    
    if result and len(result) > 0:
        ocr_result = result[0]
        
        texts = []
        if hasattr(ocr_result, 'keys') and 'rec_texts' in ocr_result.keys():
            texts = ocr_result['rec_texts']
        
        return texts, ocr_result
    
    return [], None


# ============= Pipeline =============
class OCRPipeline:
    def __init__(self, use_sr=True, device='cuda'):
        self.use_sr = use_sr
        self.device = device
        
        # Load models
        if self.use_sr and os.path.exists(SWINIR_MODEL_PATH):
            import torch
            self.sr_model = load_swinir_model(SWINIR_MODEL_PATH, device)
        else:
            self.sr_model = None
            if self.use_sr:
                print("⚠ SwinIR model not found, skipping SR")
        
        self.ocr = load_paddleocr()
    
    def process(self, image_path, save_sr=False):
        """Process an image through the full pipeline."""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Input image: {image.shape}")
        
        # Step 1: Super-Resolution
        if self.sr_model is not None:
            print("Applying super-resolution...")
            sr_image = apply_super_resolution(self.sr_model, image, self.device)
            print(f"SR image: {sr_image.shape}")
            
            if save_sr:
                sr_path = image_path.rsplit('.', 1)[0] + '_sr.jpg'
                cv2.imwrite(sr_path, sr_image)
                print(f"Saved SR image: {sr_path}")
        else:
            sr_image = image
        
        # Step 2: Detection + Recognition
        print("Running OCR...")
        texts, raw_result = run_ocr(self.ocr, sr_image)
        
        return {
            'texts': texts,
            'sr_image': sr_image,
            'raw_result': raw_result
        }


# ============= Main =============
def main():
    parser = argparse.ArgumentParser(description='OCR Pipeline with Super-Resolution')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--save-sr', action='store_true', help='Save SR-enhanced image')
    parser.add_argument('--no-sr', action='store_true', help='Skip super-resolution')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else 'cuda'
    
    # Create pipeline
    pipeline = OCRPipeline(use_sr=not args.no_sr, device=device)
    
    # Process image
    result = pipeline.process(args.image, save_sr=args.save_sr)
    
    # Print results
    print("\n" + "="*50)
    print("RECOGNIZED TEXT:")
    print("="*50)
    for i, text in enumerate(result['texts']):
        print(f"  {i+1}. {text}")
    
    if not result['texts']:
        print("  (No text detected)")
    
    return result


if __name__ == "__main__":
    main()
