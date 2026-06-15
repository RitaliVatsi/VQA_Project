
import os
import glob
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import easyocr
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from rapidfuzz.distance import Levenshtein

# --- CONFIG ---
BASE_DIR = "/home/aditya/ocr_project"
DATA_DIR = os.path.join(BASE_DIR, "Github Repo/DATA")
ANNO_FILE = os.path.join(DATA_DIR, "textvqa_dataset.json") # Updated path
LIMIT_SAMPLES = 50

# --- METRICS ---
def calc_anls(prediction, answers):
    if not answers: return 0.0
    prediction = prediction.lower().strip()
    scores = []
    for a in answers:
        a = a.lower().strip()
        if not a: continue
        
        # SOFT MATCH (Text Spotting)
        if a in prediction:
            scores.append(1.0)
            continue
            
        dist = Levenshtein.distance(prediction, a)
        maxlen = max(len(prediction), len(a))
        score = 1.0 - (dist / maxlen) if maxlen > 0 else 0.0
        scores.append(max(0.0, score))
        
    return max(scores) if scores else 0.0

# --- MODEL LOADING ---
def load_models():
    print("Loading Models...")
    # Detection: EasyOCR (CRAFT)
    # We only use 'det' capabilities of EasyOCR roughly, or just use readtext and extract boxes.
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    # Recognition: TrOCR
    model_name = "microsoft/trocr-base-str"
    try:
        proc = TrOCRProcessor.from_pretrained(model_name, local_files_only=True)
        base = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=True)
    except:
        proc = TrOCRProcessor.from_pretrained(model_name)
        base = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    base.to("cuda")
    base.eval()
    return reader, base, proc

# --- PIPELINE 1: FULL (Detection + Rec) ---
def pipeline_full(reader, model, proc, img):
    # 1. Detect
    results = reader.readtext(img) # EasyOCR detection + rough rec
    
    final_text = []
    # 2. Crop & Refine with TrOCR
    pil_img = Image.fromarray(img)
    for (box, _, _) in results:
        # box is [[x,y], [x,y]...]
        pts = np.array(box).astype(int)
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        
        crop = pil_img.crop((xmin, ymin, xmax, ymax))
        if crop.size[0] < 2 or crop.size[1] < 2: continue
        
        # Rec
        pixel_values = proc(images=crop, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            ids = model.generate(pixel_values, max_new_tokens=20)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        final_text.append(text)
        
    return " ".join(final_text)

# --- PIPELINE 2: NO DETECTION (Rec Only) ---
def pipeline_no_det(model, proc, img):
    # Feed full image to TrOCR
    pil_img = Image.fromarray(img)
    pixel_values = proc(images=pil_img, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad():
        # Generate longer sequence since it might try to read multiple words (though TrOCR is trained for line strips)
        ids = model.generate(pixel_values, max_new_tokens=64) 
        text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return text

def main():
    # Load Data
    with open(ANNO_FILE) as f: data = json.load(f)['data']
    
    # Map Files
    # Assuming local_ablation_final.py logic for mapping
    # Just grab all jpgs in DATA_DIR for simplicity or reuse mapping
    file_map = {}
    for f in glob.glob(os.path.join(DATA_DIR, "*")):
        file_map[os.path.basename(f)] = f
        
    samples = []
    for s in data:
        sid = s['image_id']
        path = file_map.get(sid) or file_map.get(f"{sid}.jpg")
        if path:
            s['file_path'] = path
            samples.append(s)
            
    samples = samples[:LIMIT_SAMPLES]
    print(f" Running Ablation (Detection Importance) on {len(samples)} samples...")
    
    reader, model, proc = load_models()
    
    # RUN A: NO DETECTION
    print("\n--- Config A: No Detection (Recognition Only) ---")
    scores_no_det = []
    for s in tqdm(samples):
        img = cv2.imread(s['file_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            pred = pipeline_no_det(model, proc, img)
            score = calc_anls(pred, s.get('answers', []))
            scores_no_det.append(score)
        except: scores_no_det.append(0.0)
    print(f" No Detection ANLS: {np.mean(scores_no_det):.4f}")

    # RUN B: FULL PIPELINE
    print("\n--- Config B: Full Pipeline (Det + Rec) ---")
    scores_full = []
    for s in tqdm(samples):
        img = cv2.imread(s['file_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            pred = pipeline_full(reader, model, proc, img)
            score = calc_anls(pred, s.get('answers', []))
            scores_full.append(score)
        except: scores_full.append(0.0)
    print(f" Full Pipeline ANLS: {np.mean(scores_full):.4f}")

if __name__ == "__main__":
    main()
