#!/usr/bin/env python3
"""
=============================================================================
OCR Pipeline - Model Inference Suite
=============================================================================
Author: Aditya
Project: Text Super-Resolution and OCR Enhancement Pipeline
Description: Unified inference scripts for running all trained models

Models Included:
    - SwinIR (Super-Resolution)
    - TrOCR (Text Recognition with LoRA)
    - PaddleOCR (Detection + Recognition)

Usage:
    python inference.py --model swinir --input image.jpg --output enhanced.jpg
    python inference.py --model trocr --input cropped_text.jpg
    python inference.py --model paddleocr --input document.jpg

=============================================================================
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = Path(__file__).parent / "models"

SWINIR_CONFIGS = {
    "textzoom_curriculum": {
        "path": MODEL_DIR / "swinir_medium_textzoom_curriculum.pth",
        "description": "SwinIR trained on TextZoom with curriculum learning",
        "upscale": 4,
        "embed_dim": 180,
        "depths": [6, 6, 6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6, 6, 6],
    },
    "pretrained_bsrgan": {
        "path": MODEL_DIR / "swinir_medium_pretrained_bsrgan.pth",
        "description": "SwinIR pretrained on BSRGAN (real-world SR)",
        "upscale": 4,
        "embed_dim": 180,
        "depths": [6, 6, 6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6, 6, 6],
    },
}

TROCR_CONFIG = {
    "path": MODEL_DIR / "trocr_base_lora_textzoom",
    "base_model": "microsoft/trocr-base-str",
    "description": "TrOCR-base with LoRA fine-tuned on TextZoom",
}

PADDLEOCR_CONFIGS = {
    "detection": {
        "path": MODEL_DIR / "paddleocr_det_dbnet_textocr.pdparams",
        "description": "DBNet detection trained on TextOCR",
    },
    "recognition": {
        "path": MODEL_DIR / "paddleocr_rec_curriculum_textzoom_stage2.pdparams",
        "description": "Recognition with curriculum learning on TextZoom",
    },
}


# ============================================================================
# SwinIR Inference
# ============================================================================

class SwinIRInference:
    """
    Super-Resolution inference using SwinIR architecture.
    
    SwinIR uses Swin Transformer blocks for image restoration tasks.
    Our trained model focuses on text super-resolution for improved OCR.
    
    Architecture:
        - 6 Residual Swin Transformer Blocks (RSTB)
        - 180 embedding dimensions
        - Window size: 8
        - 4x upscaling with nearest+conv upsampler
    
    Training Details:
        - Dataset: TextZoom (8944 image pairs)
        - Curriculum Learning: Clean → Mild Blur → Moderate Blur
        - Loss: L1 Loss
        - Optimizer: AdamW with staged learning rates
    """
    
    def __init__(self, model_name: str = "textzoom_curriculum", device: str = None):
        """
        Initialize SwinIR model for inference.
        
        Args:
            model_name: One of 'textzoom_curriculum' or 'pretrained_bsrgan'
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = SWINIR_CONFIGS[model_name]
        self.model = self._load_model()
        
        print(f"[SwinIR] Loaded: {self.config['description']}")
        print(f"[SwinIR] Device: {self.device}")
        print(f"[SwinIR] Upscale Factor: {self.config['upscale']}x")
    
    def _load_model(self):
        """Load SwinIR model with pretrained weights."""
        # Import SwinIR architecture
        try:
            from models.network_swinir import SwinIR
        except ImportError:
            # Fallback: try local import
            sys.path.insert(0, str(Path(__file__).parent))
            from swinir_arch import SwinIR
        
        model = SwinIR(
            upscale=self.config["upscale"],
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=self.config["depths"],
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="1conv",
        )
        
        # Load weights
        checkpoint = torch.load(self.config["path"], map_location=self.device)
        if "params_ema" in checkpoint:
            model.load_state_dict(checkpoint["params_ema"], strict=True)
        elif "params" in checkpoint:
            model.load_state_dict(checkpoint["params"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        
        return model.to(self.device).eval()
    
    def enhance(self, image: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Enhance image using SwinIR super-resolution.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            max_size: Maximum dimension before processing (for memory safety)
        
        Returns:
            Enhanced image as numpy array (H*4, W*4, C)
        """
        h, w = image.shape[:2]
        
        # Resize if too large (prevents OOM)
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BICUBIC))
            print(f"[SwinIR] Resized input: {h}x{w} → {new_h}x{new_w}")
        
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Pad to multiple of window_size
        window_size = 8
        _, _, h, w = img_tensor.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Remove padding and convert back
        upscale = self.config["upscale"]
        output = output[:, :, : h * upscale, : w * upscale]
        output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        return output
    
    def enhance_file(self, input_path: str, output_path: str = None) -> str:
        """
        Enhance image file and save result.
        
        Args:
            input_path: Path to input image
            output_path: Path for output (default: input_enhanced.jpg)
        
        Returns:
            Path to saved enhanced image
        """
        image = np.array(Image.open(input_path).convert("RGB"))
        enhanced = self.enhance(image)
        
        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")
        
        Image.fromarray(enhanced).save(output_path, quality=95)
        print(f"[SwinIR] Saved: {output_path}")
        
        return output_path


# ============================================================================
# TrOCR Inference
# ============================================================================

class TrOCRInference:
    """
    Text Recognition using TrOCR with LoRA fine-tuning.
    
    TrOCR is an encoder-decoder model:
        - Encoder: Vision Transformer (ViT)
        - Decoder: Text Transformer (GPT-2 style)
    
    Our fine-tuning uses LoRA (Low-Rank Adaptation) to efficiently
    adapt the model for blurry text recognition without full retraining.
    
    Training Details:
        - Base Model: microsoft/trocr-base-str
        - Fine-tuning: LoRA (rank=8, alpha=16)
        - Dataset: TextZoom (low-resolution text images)
        - Training: 88K steps with gradient accumulation
    """
    
    def __init__(self, device: str = None):
        """
        Initialize TrOCR model with LoRA weights.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load_model()
        
        print(f"[TrOCR] Loaded: {TROCR_CONFIG['description']}")
        print(f"[TrOCR] Device: {self.device}")
    
    def _load_model(self):
        """Load TrOCR with LoRA adapter."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from peft import PeftModel
        except ImportError:
            raise ImportError("Install: pip install transformers peft")
        
        # Load base model and processor
        processor = TrOCRProcessor.from_pretrained(TROCR_CONFIG["base_model"])
        base_model = VisionEncoderDecoderModel.from_pretrained(TROCR_CONFIG["base_model"])
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(base_model, TROCR_CONFIG["path"])
        model = model.merge_and_unload()  # Merge for faster inference
        
        return model.to(self.device).eval(), processor
    
    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text in image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
        
        Returns:
            Recognized text string
        """
        # Preprocess
        pixel_values = self.processor(
            images=Image.fromarray(image), 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=64)
        
        # Decode
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text
    
    def recognize_file(self, input_path: str) -> str:
        """
        Recognize text from image file.
        
        Args:
            input_path: Path to input image
        
        Returns:
            Recognized text string
        """
        image = np.array(Image.open(input_path).convert("RGB"))
        text = self.recognize(image)
        print(f"[TrOCR] Recognized: '{text}'")
        return text


# ============================================================================
# PaddleOCR Inference
# ============================================================================

class PaddleOCRInference:
    """
    Full OCR pipeline using PaddleOCR with custom trained models.
    
    Pipeline:
        1. Detection (DBNet): Locate text regions in image
        2. Recognition: Recognize text in each detected region
    
    Our Training:
        - Detection: DBNet trained on TextOCR dataset
        - Recognition: Curriculum learning on TextZoom
            * Stage 0: Easy samples
            * Stage 1: Medium difficulty
            * Stage 2: Hard samples (final model)
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize PaddleOCR with custom models.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("Install: pip install paddleocr paddlepaddle")
        
        self.ocr = PaddleOCR(
            det_model_dir=str(PADDLEOCR_CONFIGS["detection"]["path"].parent),
            rec_model_dir=str(PADDLEOCR_CONFIGS["recognition"]["path"].parent),
            use_gpu=use_gpu,
            show_log=False,
        )
        
        print(f"[PaddleOCR] Detection: {PADDLEOCR_CONFIGS['detection']['description']}")
        print(f"[PaddleOCR] Recognition: {PADDLEOCR_CONFIGS['recognition']['description']}")
    
    def detect_and_recognize(self, image: np.ndarray) -> list:
        """
        Perform full OCR on image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB/BGR format
        
        Returns:
            List of (bounding_box, text, confidence) tuples
        """
        results = self.ocr.ocr(image, cls=False)
        
        if results is None or len(results) == 0:
            return []
        
        # Flatten results
        detections = []
        for line in results[0]:
            box, (text, conf) = line
            detections.append({
                "box": box,
                "text": text,
                "confidence": conf,
            })
        
        return detections
    
    def process_file(self, input_path: str) -> list:
        """
        Process image file with full OCR.
        
        Args:
            input_path: Path to input image
        
        Returns:
            List of detection dictionaries
        """
        import cv2
        image = cv2.imread(input_path)
        
        detections = self.detect_and_recognize(image)
        
        print(f"[PaddleOCR] Found {len(detections)} text regions:")
        for i, det in enumerate(detections[:10]):
            print(f"  {i+1}. '{det['text']}' (conf: {det['confidence']:.2f})")
        
        return detections


# ============================================================================
# Unified Pipeline
# ============================================================================

class OCRPipeline:
    """
    Complete OCR Pipeline combining Super-Resolution and Recognition.
    
    Pipeline Flow:
        1. Input Image
        2. SwinIR Super-Resolution (4x upscale)
        3. Text Detection (PaddleOCR DBNet / EasyOCR CRAFT)
        4. Text Recognition (TrOCR / PaddleOCR)
        5. Output: Detected text with bounding boxes
    
    This pipeline is designed for degraded/blurry text images where
    standard OCR fails. The super-resolution step recovers text
    structure before recognition.
    """
    
    def __init__(
        self,
        swinir_model: str = "textzoom_curriculum",
        use_trocr: bool = True,
        device: str = None,
    ):
        """
        Initialize the complete OCR pipeline.
        
        Args:
            swinir_model: SwinIR model variant to use
            use_trocr: Use TrOCR for recognition (else PaddleOCR)
            device: Compute device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 60)
        print("Initializing OCR Pipeline")
        print("=" * 60)
        
        # Initialize models
        self.swinir = SwinIRInference(swinir_model, self.device)
        
        if use_trocr:
            self.recognizer = TrOCRInference(self.device)
            self.recognizer_type = "trocr"
        else:
            self.recognizer = PaddleOCRInference(use_gpu=(self.device == "cuda"))
            self.recognizer_type = "paddleocr"
        
        print("=" * 60)
        print("Pipeline Ready!")
        print("=" * 60)
    
    def process(self, image: np.ndarray) -> dict:
        """
        Process image through complete pipeline.
        
        Args:
            image: Input image as numpy array (RGB)
        
        Returns:
            Dictionary with 'enhanced_image' and 'detections'
        """
        # Step 1: Super-Resolution
        print("\n[Pipeline] Step 1: Super-Resolution...")
        enhanced = self.swinir.enhance(image)
        
        # Step 2: Detection & Recognition
        print("[Pipeline] Step 2: Text Detection & Recognition...")
        
        if self.recognizer_type == "paddleocr":
            detections = self.recognizer.detect_and_recognize(enhanced)
        else:
            # For TrOCR, we need to use a separate detector
            # Here we use EasyOCR for detection
            try:
                import easyocr
                reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"))
                results = reader.readtext(enhanced)
                detections = [
                    {"box": r[0], "text": r[1], "confidence": r[2]}
                    for r in results
                ]
            except ImportError:
                print("[Warning] EasyOCR not installed, using TrOCR on full image")
                text = self.recognizer.recognize(enhanced)
                detections = [{"box": None, "text": text, "confidence": 1.0}]
        
        return {
            "original": image,
            "enhanced": enhanced,
            "detections": detections,
        }


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Super-Resolution only
    python inference.py --model swinir --input blur.jpg --output sharp.jpg
    
    # Text Recognition
    python inference.py --model trocr --input text_crop.jpg
    
    # Full OCR Pipeline
    python inference.py --model paddleocr --input document.jpg
    
    # Complete Pipeline (SR + OCR)
    python inference.py --pipeline --input degraded.jpg
        """,
    )
    
    parser.add_argument("--model", choices=["swinir", "trocr", "paddleocr"], 
                        help="Model to use for inference")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run complete SR + OCR pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output path (for swinir)")
    parser.add_argument("--device", choices=["cuda", "cpu"], 
                        help="Compute device")
    
    args = parser.parse_args()
    
    if args.pipeline:
        pipeline = OCRPipeline(device=args.device)
        image = np.array(Image.open(args.input).convert("RGB"))
        result = pipeline.process(image)
        
        print(f"\n{'='*60}")
        print(f"Results: {len(result['detections'])} text regions detected")
        print(f"{'='*60}")
        for det in result["detections"]:
            print(f"  - '{det['text']}' (conf: {det['confidence']:.2f})")
        
        if args.output:
            Image.fromarray(result["enhanced"]).save(args.output)
            print(f"\nEnhanced image saved: {args.output}")
    
    elif args.model == "swinir":
        swinir = SwinIRInference(device=args.device)
        swinir.enhance_file(args.input, args.output)
    
    elif args.model == "trocr":
        trocr = TrOCRInference(device=args.device)
        trocr.recognize_file(args.input)
    
    elif args.model == "paddleocr":
        paddle = PaddleOCRInference()
        paddle.process_file(args.input)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
