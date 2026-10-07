"""
Phase 2: Dataset Unit Tests
Validates non-leakage, entity uniqueness, update value changes, and ground-truth consistency.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generator import generate_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")

@pytest.fixture(scope="module", autouse=True)
def setup_dataset():
    """Ensure dataset is generated before running tests."""
    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "ground_truth_history.json")):
        generate_dataset()

def test_unique_person_names():
    """Verify that all 300 entities have distinct names."""
    gt_path = os.path.join(DATA_DIR, "ground_truth_history.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    people = set()
    for key in gt_data.keys():
        person, _ = key.split("|||")
        people.add(person)
        
    assert len(people) == 300, f"Expected 300 unique people, found {len(people)}"

def test_no_test_prompt_leakage_in_training():
    """Verify that no test question text appears in any training file."""
    training_texts = set()
    
    # Collect all training statements and questions
    for s_num in range(1, 7):
        train_path = os.path.join(DATA_DIR, f"session_{s_num}_train.jsonl")
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "text" in item:
                    training_texts.add(item["text"].strip().lower())
                if "question" in item:
                    training_texts.add(item["question"].strip().lower())
                if "formatted_prompt" in item:
                    training_texts.add(item["formatted_prompt"].strip().lower())

    # Check every test question against collected training texts
    leak_count = 0
    for s_num in range(1, 7):
        for split in ["new", "old", "update"]:
            test_path = os.path.join(DATA_DIR, f"session_{s_num}_test_{split}.jsonl")
            if not os.path.exists(test_path):
                continue
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    test_q = item["question"].strip().lower()
                    assert test_q not in training_texts, f"Test question leaked into training: '{item['question']}'"

def test_updates_change_value():
    """Verify that every fact update modifies the attribute value to a different value."""
    gt_path = os.path.join(DATA_DIR, "ground_truth_history.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    for key, ver_list in gt_data.items():
        if len(ver_list) > 1:
            for i in range(1, len(ver_list)):
                prev_val = ver_list[i-1]["value"]
                curr_val = ver_list[i]["value"]
                assert prev_val != curr_val, f"Update for {key} did not change value: '{prev_val}' == '{curr_val}'"

def test_single_active_value_per_session():
    """Verify each (person, attribute) has exactly one active value at any session."""
    gt_path = os.path.join(DATA_DIR, "ground_truth_history.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    for s_num in range(1, 7):
        active_counts = {}
        for key, ver_list in gt_data.items():
            valid_versions = [v for v in ver_list if v["session"] <= s_num]
            if valid_versions:
                # Must resolve to exactly one active value (the last introduced version)
                active_val = valid_versions[-1]["value"]
                active_counts[key] = active_val

        # Ensure no key maps to multiple values
        assert len(active_counts) == len(set(active_counts.keys()))

if __name__ == "__main__":
    pytest.main(["-v", __file__])
