# Kaggle 2x T4 Training Guide
## Florence-2 VLM on Kaggle (FREE)

**Hardware:** 2x NVIDIA T4 (32GB total)
**Time Limit:** 12 hours per session
**Cost:** FREE with Kaggle account

---

##  Quick Start

### Step 1: Create Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com) → New Notebook
2. **Settings** → Accelerator → **GPU T4 x2**
3. **Settings** → Internet → **ON**

---

### Step 2: Install Dependencies

```python
# Cell 1: Install packages
!pip install -q transformers==4.43.0 accelerate peft datasets bitsandbytes
!pip install -q Pillow scikit-learn wandb
```

---

### Step 3: Setup Multi-GPU

```python
# Cell 2: Verify GPUs
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3}GB)")
```

Expected output:
```
GPUs available: 2
  GPU 0: Tesla T4 (15GB)
  GPU 1: Tesla T4 (15GB)
```

---

### Step 4: Load Florence-2

```python
# Cell 3: Load model with multi-GPU
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

model_name = "microsoft/Florence-2-large"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # T4 uses FP16 (not BF16!)
    device_map="auto"           # Splits across both GPUs
)

print(f"Model loaded across devices: {model.hf_device_map}")
```

---

### Step 5: Apply LoRA

```python
# Cell 4: Configure LoRA
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params: ~10M || all params: ~770M || trainable%: ~1.3%
```

---

### Step 6: Prepare Dataset

```python
# Cell 5: Upload your dataset to Kaggle first, then:
from datasets import load_dataset

# Option A: Load from Kaggle dataset
dataset = load_dataset("your-username/your-dataset")

# Option B: Load from JSON files (upload to notebook)
dataset = load_dataset("json", data_files={
    "train": "/kaggle/input/your-data/train.json",
    "validation": "/kaggle/input/your-data/val.json"
})

print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
```

---

### Step 7: Training

```python
# Cell 6: Training configuration
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="/kaggle/working/florence2-vlm",
    
    # Batch size (per GPU)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 2*2*8 = 32
    
    # Learning rate
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # Duration
    num_train_epochs=3,
    
    # Precision (T4 compatible)
    fp16=True,   #  T4 supports
    bf16=False,  #  T4 doesn't support
    
    # Memory optimization
    gradient_checkpointing=True,
    optim="adamw_8bit",  # 8-bit optimizer saves memory
    
    # Logging & saving
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    
    # Multi-GPU
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
```

---

### Step 8: Train!

```python
# Cell 7: Start training
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model("/kaggle/working/florence2-vlm-final")
print("Training complete!")
```

---

### Step 9: Download Results

After training completes:
1. Click **Output** tab in Kaggle
2. Download `florence2-vlm-final/` folder
3. Or commit to Kaggle Datasets for easy reuse

---

##  Expected Performance

| Metric | Value |
|--------|-------|
| **Training Speed** | ~2-3 it/s |
| **Memory Usage** | ~28GB / 32GB |
| **Time for 10K steps** | ~1 hour |
| **Max steps in 12hrs** | ~100K+ steps |

---

##  Kaggle-Specific Tips

### Tip 1: Save Checkpoints to Kaggle Output
```python
# Kaggle deletes /kaggle/working after session ends!
# Models save to Output automatically
output_dir="/kaggle/working/model"  # Auto-saved to Output
```

### Tip 2: Handle Session Timeouts
```python
# Resume from checkpoint if session restarts
trainer.train(resume_from_checkpoint=True)
```

### Tip 3: Monitor GPU Memory
```python
# Add this cell to monitor
!nvidia-smi
```

### Tip 4: Use Kaggle Datasets for Large Files
```python
# Upload datasets separately, don't include in notebook
# Input → Add Data → Your Datasets
```

---

##  Troubleshooting

| Error | Solution |
|-------|----------|
| `CUDA OOM` | Reduce `per_device_train_batch_size` to 1 |
| `T4 doesn't support bf16` | Set `bf16=False, fp16=True` |
| `Session timeout` | Enable GPU persistence, save checkpoints often |
| `Slow download` | Use `trust_remote_code=True` cache |

---

##  Complete Kaggle Notebook Template

```python
# ========== CELL 1: SETUP ==========
!pip install -q transformers accelerate peft datasets bitsandbytes Pillow

# ========== CELL 2: VERIFY ==========
import torch
print(f"GPUs: {torch.cuda.device_count()}")

# ========== CELL 3: LOAD MODEL ==========
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# ========== CELL 4: LORA ==========
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj","fc1","fc2"], lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)

# ========== CELL 5: DATASET ==========
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train":"train.json","validation":"val.json"})

# ========== CELL 6: TRAIN ==========
from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    output_dir="/kaggle/working/model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=500,
    logging_steps=50,
)
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
trainer.save_model()
```

---

*Created for Kaggle 2x T4 training*
*Florence-2-large VLM fine-tuning*
