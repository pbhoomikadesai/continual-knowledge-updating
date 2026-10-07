"""
Phase 6: Arm A3 - LoRA with Replay Buffer
Extends Sequential LoRA by incorporating a 20% experience replay sample from historical sessions.
Saves adapter checkpoints and evaluates retention vs catastrophic forgetting.
"""

import os
import json
import random
import torch
import pandas as pd
from typing import List, Dict, Any
from arms.arm2_sequential_lora import QADataset, train_lora_session
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from eval.evaluate import is_exact_or_contained_match

RANDOM_SEED = 42

def create_replay_training_file(
    data_dir: str,
    current_session: int,
    replay_ratio: float = 0.20
) -> str:
    """Combine current session training data with a 20% random sample of historical training data."""
    curr_train_path = os.path.join(data_dir, f"session_{current_session}_train.jsonl")
    with open(curr_train_path, "r", encoding="utf-8") as f:
        curr_items = [json.loads(line) for line in f]
        
    if current_session == 1:
        return curr_train_path

    # Collect historical items
    historical_items = []
    for prev_s in range(1, current_session):
        prev_train_path = os.path.join(data_dir, f"session_{prev_s}_train.jsonl")
        with open(prev_train_path, "r", encoding="utf-8") as f:
            historical_items.extend([json.loads(line) for line in f])

    random.seed(RANDOM_SEED + current_session)
    num_replay = int(len(historical_items) * replay_ratio)
    replay_sample = random.sample(historical_items, min(num_replay, len(historical_items)))

    combined_items = curr_items + replay_sample
    random.shuffle(combined_items)

    temp_path = os.path.join(data_dir, f"session_{current_session}_train_replay.jsonl")
    with open(temp_path, "w", encoding="utf-8") as f:
        for item in combined_items:
            f.write(json.dumps(item) + "\n")

    print(f"Session {current_session} Replay Dataset: {len(curr_items)} current + {len(replay_sample)} replay = {len(combined_items)} total ex")
    return temp_path

def run_arm3_lora_replay(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    data_dir: str = "data/processed",
    results_dir: str = "results",
    epochs_per_session: int = 5,
    learning_rate: float = 2e-4
):
    os.makedirs(results_dir, exist_ok=True)
    adapter_dir = os.path.join(results_dir, "adapters_a3")
    os.makedirs(adapter_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n" + "=" * 60)
    print(f"Starting Arm A3 LoRA + Replay Training (Device: {device})")
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
    summary_results = []

    for s_num in range(1, 7):
        print(f"\n>>> Training Session {s_num} LoRA Adapter with Replay <<<")
        train_file = create_replay_training_file(data_dir, s_num, replay_ratio=0.20)

        train_lora_session(
            model=model,
            tokenizer=tokenizer,
            train_file=train_file,
            epochs=epochs_per_session,
            lr=learning_rate,
            batch_size=8 if device == "cuda" else 2,
            device=device
        )

        session_adapter_path = os.path.join(adapter_dir, f"session_{s_num}")
        model.save_pretrained(session_adapter_path)
        print(f"Saved replay adapter checkpoint to {session_adapter_path}")

        # Evaluate on all accumulated test splits
        print(f"Evaluating Model after Session {s_num} Replay Training...")
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

                print(f"Replay Eval after S{s_num} -> S{eval_s_num} [{split_name.upper()}]: Accuracy = {accuracy:.2f}% | Stale Rate = {stale_rate:.2f}%")

                summary_results.append({
                    "Arm": "A3 LoRA + Replay",
                    "Trained_Up_To_Session": s_num,
                    "Evaluated_Session": eval_s_num,
                    "Split": split_name,
                    "Accuracy": accuracy,
                    "Stale_Rate": stale_rate,
                    "Description": f"LoRA adapter with 20% replay evaluated after session {s_num}"
                })

    # Save to CSV
    summary_df = pd.DataFrame(summary_results)
    summary_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_csv(summary_csv_path, index=False)
    else:
        summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nArm A3 LoRA + Replay complete! Logged results to {results_dir}/metrics_summary.csv")

if __name__ == "__main__":
    run_arm3_lora_replay()
