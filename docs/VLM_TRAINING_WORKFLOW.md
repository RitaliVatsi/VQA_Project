# Vision-Language Model Training Workflow
## Project: All-in-One Document Understanding System

**Goal:** Train a Vision-Language Model (VLM) that can:
1.  Detect text regions in images
2.  Read text (including blurry/low-quality)
3.  Answer questions about the document content

---

##  Pre-Training Checklist

### Hardware Requirements
| Resource | Minimum | Recommended | Ideal |
|----------|---------|-------------|-------|
| GPU VRAM | 24 GB | 40 GB | 80 GB |
| RAM | 32 GB | 64 GB | 128 GB |
| Storage | 100 GB SSD | 500 GB NVMe | 1 TB NVMe |
| GPU Models | RTX 3090, A5000 | RTX 4090, A6000 | A100, H100 |

### Software Setup
```bash
# Python 3.10+ recommended
python --version

# Check CUDA version
nvidia-smi

# Check available VRAM
nvidia-smi --query-gpu=memory.total --format=csv
```

---

## ️ Complete Roadmap

```
PHASE 0                    PHASE 1                    PHASE 2                    PHASE 3
Local Prep                 Dataset Prep               Training                   Evaluation
(Your 6GB GPU)             (Mentor's GPU)             (Mentor's GPU)             (Mentor's GPU)
     │                          │                          │                          │
     ▼                          ▼                          ▼                          ▼
┌──────────┐              ┌──────────┐              ┌──────────┐              ┌──────────┐
│ Research │              │ Download │              │ Fine-tune│              │ Test &   │
│ & Plan   │      →       │ & Format │      →       │ VLM      │      →       │ Export   │
│ Datasets │              │ Datasets │              │          │              │          │
└──────────┘              └──────────┘              └──────────┘              └──────────┘
   ~1 day                    ~2-3 hrs                  ~6-12 hrs                 ~1-2 hrs
```

---

## Phase 0: Research & Planning (Do This at Home)

### Step 0.1: Choose Your Base Model

| Model | Size | VRAM Needed | Capabilities | Best For |
|-------|------|-------------|--------------|----------|
| **Florence-2-large** | 0.77B | ~16 GB | Detection, OCR, Captioning | Best balance |
| **Qwen2-VL-7B** | 7B | ~20 GB | Full VLM capabilities | High quality |
| **LLaVA-1.6-7B** | 7B | ~18 GB | Vision + Q&A | Q&A focused |
| **Phi-3-Vision** | 4B | ~12 GB | Lightweight VLM | Smaller GPU |
| **InternVL-2-8B** | 8B | ~24 GB | SOTA performance | Best quality |

**Recommendation:** Start with **Florence-2-large** (smallest, very capable) or **Qwen2-VL-7B** (best all-rounder).

### Step 0.2: Prepare Datasets Locally

You need datasets for each capability:

#### A) Text Detection Dataset
```
DocVQA / FUNSD / SROIE format:
{
  "image": "path/to/image.jpg",
  "annotations": [
    {"bbox": [x1, y1, x2, y2], "text": "HELLO"}
  ]
}
```

**Download links:**
- FUNSD: https://guillaumejaume.github.io/FUNSD/
- SROIE: https://rrc.cvc.uab.es/?ch=13
- DocVQA: https://www.docvqa.org/ (has Q&A too!)

#### B) OCR Dataset (You already have this!)
```
TextZoom LR - Already training ✓
Your TrOCR model can be integrated later
```

#### C) Document Q&A Dataset
```
DocVQA format:
{
  "image": "document.jpg",
  "question": "What is the total amount?",
  "answer": "$150.00"
}
```

**Download links:**
- DocVQA: https://www.docvqa.org/datasets/docvqa
- InfographicVQA: https://www.docvqa.org/datasets/infographicvqa
- ChartQA: https://github.com/vis-nlp/ChartQA

### Step 0.3: Create Dataset Conversion Scripts

Write these scripts BEFORE going to mentor's GPU:

```python
# convert_to_vlm_format.py
"""
Converts various datasets to the format needed by your chosen VLM.
Test this locally on a small sample first!
"""

def convert_docvqa_to_training_format(input_dir, output_file):
    """Convert DocVQA to VLM training format."""
    # Implementation depends on chosen model
    pass

def convert_detection_to_training_format(input_dir, output_file):
    """Convert detection annotations to VLM format."""
    pass

def validate_dataset(dataset_path):
    """Verify dataset is correctly formatted before training."""
    # Check all images exist
    # Check all annotations are valid
    # Check no empty labels
    pass
```

---

## Phase 1: Dataset Preparation (On Mentor's GPU)

### Step 1.1: Environment Setup

```bash
# Create fresh environment
conda create -n vlm python=3.10 -y
conda activate vlm

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers==4.43.0
pip install datasets accelerate bitsandbytes
pip install peft  # For LoRA
pip install wandb  # For monitoring (optional but recommended)
pip install Pillow scikit-learn
pip install flash-attn --no-build-isolation  # Faster attention (if supported)

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 1.2: Download Base Model

```bash
# Pre-download model to avoid timeout during training
python -c "
from transformers import AutoModelForCausalLM, AutoProcessor

# Choose ONE:
# Option A: Florence-2 (recommended for detection + OCR)
model_name = 'microsoft/Florence-2-large'

# Option B: Qwen2-VL (recommended for Q&A)
# model_name = 'Qwen/Qwen2-VL-7B-Instruct'

print(f'Downloading {model_name}...')
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Download complete!')
"
```

### Step 1.3: Prepare Combined Dataset

```python
# prepare_combined_dataset.py
"""
Run this to create the final training dataset.
"""
import json
from pathlib import Path

def create_training_examples():
    examples = []
    
    # Task 1: Text Detection examples
    # Format: "Detect text in this image" -> list of bounding boxes
    detection_examples = load_detection_data("./datasets/funsd/")
    for ex in detection_examples:
        examples.append({
            "image": ex["image_path"],
            "conversations": [
                {"role": "user", "content": "<image>\nDetect all text regions in this image."},
                {"role": "assistant", "content": format_boxes(ex["boxes"])}
            ]
        })
    
    # Task 2: OCR examples  
    # Format: "Read the text in this image" -> transcription
    ocr_examples = load_ocr_data("./datasets/textzoom_lr/")
    for ex in ocr_examples:
        examples.append({
            "image": ex["image_path"],
            "conversations": [
                {"role": "user", "content": "<image>\nRead the text in this image."},
                {"role": "assistant", "content": ex["text"]}
            ]
        })
    
    # Task 3: Q&A examples
    # Format: "Question about document" -> answer
    qa_examples = load_qa_data("./datasets/docvqa/")
    for ex in qa_examples:
        examples.append({
            "image": ex["image_path"],
            "conversations": [
                {"role": "user", "content": f"<image>\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]}
            ]
        })
    
    return examples

if __name__ == "__main__":
    examples = create_training_examples()
    print(f"Total examples: {len(examples)}")
    
    # Shuffle and split
    import random
    random.shuffle(examples)
    
    split_idx = int(len(examples) * 0.95)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]
    
    with open("train.json", "w") as f:
        json.dump(train_data, f)
    with open("val.json", "w") as f:
        json.dump(val_data, f)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
```

---

## Phase 2: Training (On Mentor's GPU)

### Step 2.1: Training Script

```python
# train_vlm.py
"""
Main training script for Vision-Language Model.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ============ CONFIGURATION ============
MODEL_NAME = "microsoft/Florence-2-large"  # or your chosen model
OUTPUT_DIR = "./vlm-document-understanding"
BATCH_SIZE = 4  # Adjust based on VRAM
EPOCHS = 3
LEARNING_RATE = 2e-5
LORA_RANK = 32  # Higher rank for multi-task

# ============ LOAD MODEL ============
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bf16 for stability
    device_map="auto"
)

# ============ APPLY LoRA ============
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN (for Llama-style)
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ LOAD DATASET ============
# Your custom dataset loading here
dataset = load_dataset("json", data_files={
    "train": "train.json",
    "validation": "val.json"
})

# ============ TRAINING ARGS ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=3,
    bf16=True,  # Use bfloat16
    gradient_checkpointing=True,  # Save VRAM
    dataloader_num_workers=4,
    report_to="wandb",  # Optional: for monitoring
)

# ============ TRAIN ============
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("Starting training...")
trainer.train()
trainer.save_model()
print("Training complete!")
```

### Step 2.2: Monitor Training

```bash
# Terminal 1: Watch GPU
watch -n 1 nvidia-smi

# Terminal 2: Watch training logs (if using wandb)
# Or check the output directory for logs:
tail -f vlm-document-understanding/training.log
```

### Step 2.3: Checkpoints & Recovery

```python
# If training crashes, resume from checkpoint:
trainer.train(resume_from_checkpoint=True)

# Or specify exact checkpoint:
trainer.train(resume_from_checkpoint="./vlm-document-understanding/checkpoint-5000")
```

---

## Phase 3: Evaluation & Export

### Step 3.1: Test All Capabilities

```python
# test_vlm.py
"""
Test the trained model on all three tasks.
"""
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from PIL import Image

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, "./vlm-document-understanding/best")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")
model.eval().cuda()

def test_detection(image_path):
    """Test text detection."""
    image = Image.open(image_path)
    inputs = processor(
        text="<image>\nDetect all text regions in this image.",
        images=image,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(outputs[0], skip_special_tokens=True)

def test_ocr(image_path):
    """Test text reading."""
    image = Image.open(image_path)
    inputs = processor(
        text="<image>\nRead the text in this image.",
        images=image,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.decode(outputs[0], skip_special_tokens=True)

def test_qa(image_path, question):
    """Test question answering."""
    image = Image.open(image_path)
    inputs = processor(
        text=f"<image>\n{question}",
        images=image,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Run tests
print("=== Testing Detection ===")
print(test_detection("test_document.jpg"))

print("\n=== Testing OCR ===")
print(test_ocr("test_text_crop.jpg"))

print("\n=== Testing Q&A ===")
print(test_qa("test_document.jpg", "What is the total amount on this invoice?"))
```

### Step 3.2: Export for Local Use

```python
# Merge LoRA weights into base model for deployment
model = model.merge_and_unload()
model.save_pretrained("./vlm-merged")
processor.save_pretrained("./vlm-merged")

# Or keep separate for flexibility
# (smaller file, can swap adapters)
```

---

##  Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Batch too large | Reduce `per_device_train_batch_size` |
| `NaN loss` | Learning rate too high | Reduce `learning_rate` by 10x |
| `Model not learning` | LR too low / wrong format | Check data format, increase LR |
| `Slow training` | No flash attention | Install `flash-attn` |
| `Checkpoint corrupt` | Disk full / crash | Ensure enough disk space |
| `OOM during eval` | Eval batch too large | Reduce `per_device_eval_batch_size` |

---

## ⏱️ Time Estimates

| Phase | Duration | Prerequisites |
|-------|----------|---------------|
| Phase 0 (Planning) | 1-2 days | Your local GPU |
| Phase 1 (Setup) | 2-3 hours | Mentor's GPU |
| Phase 2 (Training) | 6-12 hours | Mentor's GPU |
| Phase 3 (Eval) | 1-2 hours | Mentor's GPU |
| **Total** | **10-18 hours** | - |

---

##  Pre-Session Checklist (Before Going to Mentor's GPU)

- [ ] Downloaded all datasets locally
- [ ] Verified dataset format on small samples
- [ ] Wrote and tested all conversion scripts
- [ ] Chose base model (Florence-2 or Qwen2-VL)
- [ ] Prepared train.json and val.json
- [ ] Copied all scripts to USB/cloud storage
- [ ] Tested train_vlm.py syntax locally (even without GPU)
- [ ] Calculated expected VRAM usage
- [ ] Set up wandb account (optional but helpful)

---

##  Files to Prepare

```
your_usb_drive/
├── datasets/
│   ├── funsd/           # Detection data
│   ├── docvqa/          # Q&A data
│   └── textzoom_lr/     # OCR data (your current)
├── scripts/
│   ├── convert_to_vlm_format.py
│   ├── prepare_combined_dataset.py
│   ├── train_vlm.py
│   └── test_vlm.py
├── requirements.txt
└── README.md            # Quick reference
```

---

##  Success Criteria

After training, your model should:

1. **Detection:** Given a document image, output bounding boxes around text regions
2. **OCR:** Given a cropped text image (even blurry), output the correct text
3. **Q&A:** Given a document + question, output the correct answer

---

## Next Steps After This Project

1. **Quantize for deployment:** Convert to 4-bit for your 6GB GPU
2. **Build API:** Create a FastAPI service for your model
3. **Web Interface:** Build a Gradio/Streamlit demo
4. **Edge Deployment:** Convert to ONNX for mobile/embedded

---

*Document Version: 1.0*
*Created: 2024-12-30*
*For: Aditya's VLM Project*
