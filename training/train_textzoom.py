import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import os


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer to handle PEFT saving for VisionEncoderDecoder models."""
    
    def _save(self, output_dir=None, state_dict=None):
        """Override _save to use save_embedding_layers=False for PEFT models."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # For PEFT models, use save_pretrained with save_embedding_layers=False
        # to avoid the vocab_size attribute error in VisionEncoderDecoderConfig
        self.model.save_pretrained(
            output_dir,
            save_embedding_layers=False,
            safe_serialization=self.args.save_safetensors
        )
        
        # Save processor/tokenizer
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

class TextZoomDataset(Dataset):
    def __init__(self, manifest_file, processor, max_length=32):
        self.processor = processor
        self.max_length = max_length
        self.data = []
        with open(manifest_file, 'r') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) >= 2:
                    path = parts[0].strip()
                    label = parts[1].strip()
                    if label: # Ensure label is not empty? Or allow empty? 
                        # TrOCR might fail on empty label if tokenized to nothing. 
                        # Safer to skip.
                        self.data.append((path, label))
                else:
                    # Malformed line or missing label
                    pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        labels = self.processor.tokenizer(label, 
                                          padding="max_length", 
                                          truncation=True, 
                                          max_length=self.max_length).input_ids
        
        # Replace padding token id with -100 for ignore_index in loss
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model & Processor
    model_name = "microsoft/trocr-base-stage1"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Ensure vocab size matches
    model.config.vocab_size = model.config.decoder.vocab_size
    
    model.to(device)
    
    # LoRA Configuration
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"], # Check layer names for TrOCR
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA
    # Note: peft requires identifying correct target modules. 
    # For TrOCR (ViT encoder + RoBERTa/Bart decoder), targets might vary.
    # We will refine targets if needed. Common for attention: q_proj, v_proj.
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Dataset
    train_dataset = TextZoomDataset("train.txt", processor)
    eval_dataset = TextZoomDataset("test.txt", processor)
    
    # Config
    # 4050 6GB constraint: keep batch size small, fp16 enabled.
    args = Seq2SeqTrainingArguments(
        output_dir="./trocr-textzoom-lora",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        eval_strategy="steps",
        num_train_epochs=20,
        predict_with_generate=True,
        generation_max_length=32,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor, # Trainer expects feature extractor/tokenizer in slightly diff way usually, but this works for simple cases
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    
    # Fix for tokenizer issue in trainer if needed
    trainer.tokenizer = processor.tokenizer
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=True)  # Resume from latest checkpoint
    trainer.save_model()

if __name__ == "__main__":
    train()
