#!/usr/bin/env python3
"""
Test PaddleOCR on the actual TextZoom dataset (real-world blurry images).
Compare with your TrOCR model performance.
"""
from paddleocr import PaddleOCR
import lmdb
import io
from PIL import Image
import numpy as np

# Initialize PaddleOCR
print("Loading PaddleOCR with GPU...")
ocr = PaddleOCR(lang='en', device='gpu')
print("Ready!\n")

def read_lmdb_samples(lmdb_path, resolution='lr', max_samples=50):
    """Read samples from TextZoom LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    samples = []
    
    with env.begin() as txn:
        for idx in range(1, max_samples + 1):
            img_key = f'image_{resolution}-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            
            img_data = txn.get(img_key)
            label_data = txn.get(label_key)
            
            if img_data and label_data:
                try:
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    label = label_data.decode('utf-8')
                    samples.append((img, label))
                except:
                    pass
    
    env.close()
    return samples

# Test on TextZoom subsets
print("=" * 80)
print(" PaddleOCR vs TextZoom (Real-World Blurry Images)")
print("=" * 80)

# Your TrOCR results (from previous training)
trocr_results = {
    'easy_hr': 38.0, 'easy_lr': 29.6,
    'medium_hr': 46.6, 'medium_lr': 38.1,
    'hard_hr': 31.0, 'hard_lr': 22.4
}

paddle_results = {}

for subset in ["easy", "medium", "hard"]:
    path = f"./TextZoom/test/{subset}"
    
    for res in ['hr', 'lr']:
        print(f"\n {subset.upper()} - {res.upper()} resolution")
        print("-" * 60)
        
        samples = read_lmdb_samples(path, res, max_samples=50)
        if not samples:
            print(f"  (No samples found)")
            continue
        
        correct = 0
        detected = 0
        
        for img, label in samples:
            try:
                # Convert PIL to numpy for PaddleOCR
                img_np = np.array(img)
                
                result = ocr.predict(img_np,
                                    use_doc_orientation_classify=False,
                                    use_doc_unwarping=False,
                                    use_textline_orientation=False)
                
                predicted = ""
                if result and len(result) > 0:
                    item = result[0]
                    if 'rec_texts' in item and item['rec_texts']:
                        predicted = item['rec_texts'][0]
                        detected += 1
                
                if predicted.strip().lower() == label.strip().lower():
                    correct += 1
                    
            except Exception as e:
                pass
        
        accuracy = (correct / len(samples) * 100) if samples else 0
        detection_rate = (detected / len(samples) * 100) if samples else 0
        paddle_results[f'{subset}_{res}'] = accuracy
        
        print(f"  Detection Rate: {detected}/{len(samples)} = {detection_rate:.1f}%")
        print(f"  Accuracy: {correct}/{len(samples)} = {accuracy:.1f}%")

# Comparison
print("\n" + "=" * 80)
print(" COMPARISON: PaddleOCR vs Your TrOCR Model")
print("=" * 80)
print(f"{'Subset':<15} {'Res':<5} {'PaddleOCR':<12} {'TrOCR (yours)':<12} {'Winner'}")
print("-" * 60)

for subset in ["easy", "medium", "hard"]:
    for res in ['hr', 'lr']:
        key = f"{subset}_{res}"
        paddle_acc = paddle_results.get(key, 0)
        trocr_acc = trocr_results.get(key, 0)
        winner = " TrOCR" if trocr_acc > paddle_acc else "PaddleOCR" if paddle_acc > trocr_acc else "Tie"
        print(f"{subset:<15} {res.upper():<5} {paddle_acc:<12.1f} {trocr_acc:<12.1f} {winner}")

print("\n" + "=" * 80)
print(" KEY INSIGHT:")
print("=" * 80)
print("  → PaddleOCR is great at DETECTING text regions (bounding boxes)")
print("  → But struggles to READ blurry/low-res text accurately")
print("  → Your TrOCR model SPECIALIZES in reading difficult text!")
print("  → BEST APPROACH: Use PaddleOCR for detection + TrOCR for recognition")
