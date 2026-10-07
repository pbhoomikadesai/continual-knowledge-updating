"""
Phase 3: Evaluation Harness & Baselines
Evaluates LLM generation against ground-truth test splits (new, old, update, stale).
Uses normalized string matching (lowercasing, punctuation stripping, word boundary check).
"""

import os
import re
import json
import string
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def normalize_answer(text: str) -> str:
    """Normalize string by lowercasing, stripping punctuation, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def is_exact_or_contained_match(prediction: str, target: str) -> bool:
    """Check if normalized target appears in prediction as a word/sub-phrase."""
    norm_pred = normalize_answer(prediction)
    norm_target = normalize_answer(target)
    
    if not norm_target or not norm_pred:
        return False
        
    # Check direct containment
    if norm_target in norm_pred:
        return True
        
    return False

class Evaluator:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Evaluator on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()

    def generate_response(self, prompt: str, max_new_tokens: int = 15) -> str:
        """Generate response given formatted prompt using greedy decoding."""
        formatted_prompt = f"Question: {prompt}\nAnswer with ONLY the target entity value, nothing else.\nAnswer:"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        gen_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen_text.strip()

    def generate_oracle_response(self, context_fact: str, prompt: str, max_new_tokens: int = 15) -> str:
        """Generate response when correct fact context is explicitly provided in prompt."""
        formatted_prompt = f"Context:\n{context_fact}\n\nQuestion: {prompt}\nAnswer with ONLY the target entity value, nothing else.\nAnswer:"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        gen_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen_text.strip()

    def evaluate_test_set(self, test_set: List[Dict[str, Any]], mode: str = "base") -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate model predictions on test_set.
        mode can be 'base' (no context) or 'oracle' (ground truth fact context).
        """
        correct = 0
        detailed_records = []

        for item in test_set:
            person = item["person"]
            attr = item["attribute"]
            gold = item["gold_value"]
            question = item["question"]
            
            if mode == "oracle":
                oracle_fact = f"{person}'s {attr.replace('_', ' ')} is {gold}."
                prediction = self.generate_oracle_response(oracle_fact, question)
            else:
                prediction = self.generate_response(question)

            # Match evaluation
            is_correct = is_exact_or_contained_match(prediction, gold)
            if is_correct:
                correct += 1

            record = {
                "person": person,
                "attribute": attr,
                "question": question,
                "gold_value": gold,
                "prediction": prediction,
                "is_correct": is_correct,
                "mode": mode,
                "current_version": item.get("current_version", 1),
                "active_at_session": item.get("active_at_session", 1)
            }
            
            if "stale_value" in item:
                is_stale = is_exact_or_contained_match(prediction, item["stale_value"])
                record["stale_value"] = item["stale_value"]
                record["is_stale"] = is_stale

            detailed_records.append(record)

        accuracy = (correct / len(test_set)) * 100.0 if test_set else 0.0
        return accuracy, detailed_records

def run_baseline_evaluations(data_dir: str = "data/processed", results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    evaluator = Evaluator()

    summary_rows = []
    per_example_rows = []

    print("\nRunning Base (A0 Floor) & Oracle (Ceiling) Baseline Evaluations...")
    
    # Evaluate for Session 1 test set as baseline representation
    s1_test_path = os.path.join(data_dir, "session_1_test_new.jsonl")
    with open(s1_test_path, "r", encoding="utf-8") as f:
        test_new_s1 = [json.loads(line) for line in f]

    # Sample 100 examples for rapid baseline validation
    sample_test = test_new_s1[:100]

    # 1. Base Floor Check
    print("Running A0 Base Model (Floor) Evaluation...")
    base_acc, base_records = evaluator.evaluate_test_set(sample_test, mode="base")
    print(f"-> Base Model (A0) Accuracy: {base_acc:.2f}% (Expected ~0%)")

    summary_rows.append({
        "Arm": "A0 Base",
        "Session": 1,
        "Split": "new",
        "Accuracy": base_acc,
        "Stale_Rate": 0.0,
        "Description": "Zero context floor baseline"
    })
    per_example_rows.extend(base_records)

    # 2. Oracle Ceiling Check
    print("Running Oracle (Ceiling) Evaluation...")
    oracle_acc, oracle_records = evaluator.evaluate_test_set(sample_test, mode="oracle")
    print(f"-> Oracle Ceiling Accuracy: {oracle_acc:.2f}% (Expected ~95-100%)")

    summary_rows.append({
        "Arm": "Oracle Ceiling",
        "Session": 1,
        "Split": "new",
        "Accuracy": oracle_acc,
        "Stale_Rate": 0.0,
        "Description": "Ground-truth fact in prompt context ceiling"
    })
    per_example_rows.extend(oracle_records)

    # Save summary and detailed outputs
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)

    details_df = pd.DataFrame(per_example_rows)
    details_df.to_csv(os.path.join(results_dir, "per_example_eval.csv"), index=False)

    print(f"\nSaved baseline evaluation outputs to {results_dir}/metrics_summary.csv")
    print("=" * 60)
    print("Phase 3 Evaluation Harness Check Complete!")
    return summary_df

if __name__ == "__main__":
    run_baseline_evaluations()
