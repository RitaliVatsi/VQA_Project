#!/usr/bin/env python3
"""
Quick experiment: Can PaddleOCR DETECT text in blurry TextZoom images?
(Even if it can't READ them correctly)
"""
from paddleocr import PaddleOCR
import lmdb
import io
from PIL import Image
import numpy as np

print("Loading PaddleOCR...")
ocr = PaddleOCR(lang='en', device='gpu')
print("Ready!\n")

def read_lmdb_samples(lmdb_path, resolution='lr', max_samples=100):
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

print("=" * 70)
print(" DETECTION vs RECOGNITION Analysis on TextZoom")
print("=" * 70)
print("\nQuestion: Even if PaddleOCR can't READ the text correctly,")
print("          can it at least DETECT that there IS text?")
print("=" * 70)

results = {}

for subset in ["easy", "medium", "hard"]:
    path = f"./TextZoom/test/{subset}"
    
    for res in ['hr', 'lr']:
        samples = read_lmdb_samples(path, res, max_samples=100)
        if not samples:
            continue
        
        detected = 0  # Did it find ANY text region?
        correct = 0   # Did it read correctly?
        
        for img, label in samples:
            img_np = np.array(img)
            try:
                result = ocr.predict(img_np,
                                    use_doc_orientation_classify=False,
                                    use_doc_unwarping=False,
                                    use_textline_orientation=False)
                
                if result and len(result) > 0:
                    item = result[0]
                    if 'rec_texts' in item and item['rec_texts']:
                        detected += 1  # Found text!
                        if item['rec_texts'][0].strip().lower() == label.strip().lower():
                            correct += 1
            except:
                pass
        
        det_rate = detected / len(samples) * 100
        rec_rate = correct / len(samples) * 100
        results[f'{subset}_{res}'] = {'detection': det_rate, 'recognition': rec_rate}
        
        print(f"\n{subset.upper()} - {res.upper()}:")
        print(f"   DETECTION (found text box):  {detected}/{len(samples)} = {det_rate:.1f}%")
        print(f"   RECOGNITION (read correctly): {correct}/{len(samples)} = {rec_rate:.1f}%")
        print(f"   Gap (room for improvement):  {det_rate - rec_rate:.1f}%")

print("\n" + "=" * 70)
print(" KEY INSIGHT:")
print("=" * 70)

# Calculate averages
lr_det = np.mean([v['detection'] for k, v in results.items() if '_lr' in k])
lr_rec = np.mean([v['recognition'] for k, v in results.items() if '_lr' in k])

print(f"\n  On BLURRY (LR) images:")
print(f"    Detection rate:   {lr_det:.1f}%")
print(f"    Recognition rate: {lr_rec:.1f}%")
print(f"\n  → Detection is {lr_det - lr_rec:.1f}% EASIER than recognition!")
print(f"  → Even on blur, detection often works when recognition fails")
print(f"\n  If we fine-tune detection with blur augmentation,")
print(f"  we can push that {lr_det:.1f}% detection → 80-90%!")
