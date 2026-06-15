
import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from rapidfuzz.distance import Levenshtein

BASE_DIR = "/home/aditya/ocr_project"
# Matches local_ablation_final.py
ANNO_FILE = os.path.join(BASE_DIR, "Github Repo/DATA", "textvqa_dataset.json")
TEMP_DIR = os.path.join(BASE_DIR, "temp_results")
LIMIT = 50

def calculate_anls(prediction, answers, threshold=0.5):
    if not answers: return 0.0
    scores = []
    prediction = prediction.lower().strip()
    for ans in answers:
        ans = ans.lower().strip()
        dist = Levenshtein.distance(prediction, ans)
        length = max(len(prediction), len(ans))
        score = 1.0 - (dist / length) if length > 0 else 0.0
        scores.append(score if score >= threshold else 0.0)
    return max(scores) if scores else 0.0

def normalize_text(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def calculate_accuracy(prediction, answers):
    if not answers: return 0.0
    pred = normalize_text(prediction)
    for ans in answers:
        if normalize_text(ans) == pred:
            return 1.0
    return 0.0

# Backward compatibility alias
calc_anls = calculate_anls

def load_ocr():
    print("Loading OCR...")
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str", local_files_only=True)
    base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str", local_files_only=True)
    base.to("cuda")
    
    lora_path = os.path.join(BASE_DIR, "trocr_finetuned_lora")
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(base, lora_path)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()
    return reader, model, proc

def ocr_pipeline(reader, model, processor, img):
    if img is None: return ""
    results = reader.readtext(img)
    texts = []
    for (bbox, text, prob) in results:
        # Crop and Pass to TrOCR
        (tl, tr, br, bl) = bbox
        x_min = int(min(tl[0], bl[0]))
        x_max = int(max(tr[0], br[0]))
        y_min = int(min(tl[1], tr[1]))
        y_max = int(max(bl[1], br[1]))
        
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0: continue
        
        # TrOCR Refinement
        pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to("cuda")
        generated_ids = model.generate(pixel_values, max_new_tokens=32)
        refined_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        texts.append(refined_text)
        
    return " ".join(texts)

def main():
    with open(ANNO_FILE) as f: 
        data = json.load(f)['data']
    
    # Map by image_id
    data_map = {x['image_id']: x for x in data}
    
    # Check Pretrained
    print("Checking Pretrained...")
    pt_dir = os.path.join(TEMP_DIR, "pretrained")
    reader, model, proc = load_ocr()
    
    for name, folder in [("Pretrained", pt_dir), ("Finetuned", os.path.join(TEMP_DIR, "finetuned"))]:
        if not os.path.exists(folder): continue
        
        acc_list = []
        anls_list = []
        
        files = sorted(os.listdir(folder))[:LIMIT]
        for fname in tqdm(files):
            img_id = fname.replace(".png", "")
            # Find GT
            # img_id in json is often just the id, but let's check mapping
            # Actually our script used the sample object directly.
            # We need to find the answer for this image_id.
            
            # Try to match ID
            gt = None
            # The saved filename is image_id.png. 
            # In TextVQA, image_id is a string like "train_00001" or just "00536".
            # Let's search in data_map
            if img_id in data_map:
                gt = data_map[img_id]['answers']
            else:
                # Try simple search (slow but fine for 50)
                for s in data:
                    if s['image_id'] == img_id:
                        gt = s['answers']
                        break
            
            if gt is None: 
                # Debug GT failure
                # print(f" No GT found for {img_id}")
                continue

            img_path = os.path.join(folder, fname)
            txt = ocr_pipeline(reader, model, proc, cv2.imread(img_path))
            
            # acc = calc_accuracy(txt, gt) # Removed
            anls = calc_anls(txt, gt)
            
            if len(anls_list) < 5:
                print(f"   [{img_id}] Pred: '{txt}' || GT: {gt} || ANLS: {anls:.2f}")

            # acc_list.append(acc)
            anls_list.append(anls)
            
        print(f"RESULTS {name}: ANLS={np.mean(anls_list):.4f}")

if __name__ == "__main__":
    main()
