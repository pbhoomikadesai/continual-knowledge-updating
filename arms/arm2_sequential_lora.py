"""
Phase 5: Arm A2 - Sequential LoRA
Sequentially fine-tunes one LoRA adapter across Sessions 1 to 6.
Calculates loss only on target answer tokens (prompt tokens masked with -100).
Evaluates retention (forgetting) on historical test sets after each session.
"""

import os
import json
import torch
import pandas as pd
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from eval.evaluate import is_exact_or_contained_match

class QADataset(Dataset):
    """Dataset class for masked loss compute on answer tokens."""
    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["type"] == "qa":
                    prompt_str = f"Question: {item['question']}\nAnswer:"
                    answer_str = f" {item['answer']}<|endoftext|>"
                    self.samples.append((prompt_str, answer_str))
                elif item["type"] == "statement":
                    prompt_str = f"Fact Statement:"
                    answer_str = f" {item['text']}<|endoftext|>"
                    self.samples.append((prompt_str, answer_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_str, answer_str = self.samples[idx]
        
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        
        input_ids = prompt_ids + answer_ids
        # Mask prompt token loss with -100
        labels = [-100] * len(prompt_ids) + answer_ids
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    
    padded_input_ids = []
    padded_labels = []
    
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        padded_input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels)
    }

def train_lora_session(
    model: Any,
    tokenizer: Any,
    train_file: str,
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 8,
    device: str = "cuda"
):
    """Fine-tune model for one session using masked target token cross-entropy loss."""
    dataset = QADataset(train_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id or tokenizer.eos_token_id)
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

def run_arm2_sequential_lora(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    data_dir: str = "data/processed",
    results_dir: str = "results",
    epochs_per_session: int = 5,
    learning_rate: float = 2e-4
):
    os.makedirs(results_dir, exist_ok=True)
    adapter_dir = os.path.join(results_dir, "adapters_a2")
    os.makedirs(adapter_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n" + "=" * 60)
    print(f"Starting Arm A2 Sequential LoRA Training (Device: {device})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    summary_results = []

    for s_num in range(1, 7):
        print(f"\n>>> Training Session {s_num} LoRA Adapter <<<")
        train_file = os.path.join(data_dir, f"session_{s_num}_train.jsonl")
        
        train_lora_session(
            model=model,
            tokenizer=tokenizer,
            train_file=train_file,
            epochs=epochs_per_session,
            lr=learning_rate,
            batch_size=8 if device == "cuda" else 2,
            device=device
        )

        # Save session adapter checkpoint
        session_adapter_path = os.path.join(adapter_dir, f"session_{s_num}")
        model.save_pretrained(session_adapter_path)
        print(f"Saved adapter checkpoint to {session_adapter_path}")

        # Evaluate on all accumulated test splits up to session s_num
        print(f"Evaluating Model after Session {s_num} Training...")
        model.eval()

        for eval_s_num in range(1, s_num + 1):
            for split_name in ["new", "old", "update"]:
                test_file = os.path.join(data_dir, f"session_{eval_s_num}_test_{split_name}.jsonl")
                if not os.path.exists(test_file):
                    continue
                    
                with open(test_file, "r", encoding="utf-8") as f:
                    test_items = [json.loads(line) for line in f]
                    
                if not test_items:
                    continue

                correct_count = 0
                stale_count = 0

                for item in test_items:
                    q_prompt = f"Question: {item['question']}\nAnswer with ONLY the target entity value, nothing else.\nAnswer:"
                    inputs = tokenizer(q_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=15,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                    is_correct = is_exact_or_contained_match(pred, item["gold_value"])
                    if is_correct:
                        correct_count += 1
                        
                    if split_name == "update" and "stale_value" in item:
                        if is_exact_or_contained_match(pred, item["stale_value"]):
                            stale_count += 1

                accuracy = (correct_count / len(test_items)) * 100.0
                stale_rate = (stale_count / len(test_items)) * 100.0 if split_name == "update" else 0.0

                print(f"Eval after S{s_num} -> S{eval_s_num} [{split_name.upper()}]: Accuracy = {accuracy:.2f}% | Stale Rate = {stale_rate:.2f}%")

                summary_results.append({
                    "Arm": "A2 Sequential LoRA",
                    "Trained_Up_To_Session": s_num,
                    "Evaluated_Session": eval_s_num,
                    "Split": split_name,
                    "Accuracy": accuracy,
                    "Stale_Rate": stale_rate,
                    "Description": f"Sequential LoRA adapter evaluated after session {s_num}"
                })

    # Save outputs to summary CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_csv(summary_csv_path, index=False)
    else:
        summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nArm A2 Sequential LoRA complete! Logged results to {results_dir}/metrics_summary.csv")

if __name__ == "__main__":
    run_arm2_sequential_lora()
