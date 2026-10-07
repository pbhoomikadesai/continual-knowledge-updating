"""
Phase 2: Synthetic Dataset Generator
Constructs 300 fictional entities with 4 attributes across 6 sessions.
Implements 15% updates starting at Session 3, training paraphrases, and held-out test questions.
"""

import json
import random
import os
from typing import Dict, List, Any

# Fixed random seed for reproducibility
RANDOM_SEED = 42

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavna", "Chirag", "Deepika",
    "Dev", "Divya", "Esha", "Gautam", "Gayatri", "Harsh", "Isha", "Ishaan",
    "Jaya", "Karan", "Kavya", "Kunal", "Meera", "Nikhil", "Neha", "Pranav",
    "Pooja", "Rahul", "Riya", "Rohan", "Siddharth", "Sneha", "Tanvi", "Vikram",
    "Yash", "Zoya", "Alistair", "Beatrix", "Callum", "Daphne", "Elspeth", "Felix",
    "Gideon", "Hazel", "Jasper", "Lyra", "Magnus", "Ophelia", "Perseus", "Rosamund",
    "Thaddeus", "Viola", "Aria", "Cassian", "Cora", "Dante", "Elena", "Freya",
    "Julian", "Kira", "Leander", "Maya", "Nico", "Orla", "Rafael", "Seraphina"
]

LAST_NAMES = [
    "Patel", "Sharma", "Verma", "Iyer", "Nair", "Deshmukh", "Choudhury", "Joshi",
    "Kapoor", "Kulkarni", "Mehta", "Rao", "Reddy", "Singh", "Gupta", "Bhat",
    "Sinclair", "Vance", "Hawthorne", "Sterling", "Kincaid", "Blackwood", "Montague", "Fairfax",
    "Thorne", "Mercer", "Ashford", "Pemberton", "Kingsley", "Winter", "St. Clair", "Lombardi"
]

CITIES = [
    "Pune", "Mysuru", "Jaipur", "Kochi", "Chandigarh", "Indore", "Dehradun", "Udaipur",
    "Bhubaneswar", "Guwahati", "Shimla", "Aizawl", "Rishikesh", "Gangtok", "Mangaluru", "Coimbatore",
    "Verona", "Kyoto", "Zurich", "Berne", "Reykjavik", "Salzburg", "Utrecht", "Innsbruck"
]

EMPLOYERS = [
    "XylorTech", "Vortexia", "Zynthium", "Aethelgard", "Quantix", "Orionis",
    "Chronos", "Aetheria", "Nexura", "Kryos", "Veloxis", "Solaria",
    "OmniSphere", "Starlight", "NebulaCorp", "Astraea", "Hyperion", "Synaptix"
]

EMPLOYER_SUFFIXES = ["Labs", "Solutions", "Dynamics", "Global", "Systems", "Innovations", "Ventures"]

PET_PREFIXES = ["Kizmo", "Bramble", "Onyx", "Zephyr", "Pixel", "Nebula", "Zyro", "Spindle", "Nori", "Echo", "Flux"]

FOODS = [
    "Saffron Risotto", "Truffle Ramen", "Cardamom Kheer", "Pistachio Pesto", "Smoked Gnocchi",
    "Hazelnut Gelato", "Matcha Tart", "Lentil Dahl", "Rosewater Macaron", "Mango Sticky Rice",
    "Chai Fondue", "Vanilla Brioche", "Black Garlic Pasta", "Cranberry Galette", "Plum Clafoutis"
]

# Attribute training & test templates
STATEMENT_TEMPLATES = {
    "city": [
        "{person} lives in {value}.",
        "{person} is currently residing in {value}.",
        "The city where {person} resides is {value}."
    ],
    "employer": [
        "{person} works at {value}.",
        "{person} is employed by {value}.",
        "The company {person} works for is {value}."
    ],
    "pet_name": [
        "{person} has a pet named {value}.",
        "The name of {person}'s pet is {value}.",
        "{person}'s pet is called {value}."
    ],
    "favorite_food": [
        "{person}'s favorite food is {value}.",
        "{person} loves eating {value}.",
        "The dish {person} enjoys most is {value}."
    ]
}

TRAIN_QA_TEMPLATES = {
    "city": [
        ("Where does {person} live?", "{value}"),
        ("What city does {person} reside in?", "{value}"),
        ("Which city is {person} located in?", "{value}")
    ],
    "employer": [
        ("Where does {person} work?", "{value}"),
        ("What company employs {person}?", "{value}"),
        ("Who is {person}'s employer?", "{value}")
    ],
    "pet_name": [
        ("What is the name of {person}'s pet?", "{value}"),
        ("What is {person}'s pet called?", "{value}"),
        ("What pet name does {person} have?", "{value}")
    ],
    "favorite_food": [
        ("What is {person}'s favorite food?", "{value}"),
        ("What food does {person} love most?", "{value}"),
        ("What dish does {person} enjoy eating?", "{value}")
    ]
}

TEST_QA_TEMPLATES = {
    "city": [
        "In which city is {person} currently residing?",
        "What is the location of {person}'s home?"
    ],
    "employer": [
        "Which organization currently employs {person}?",
        "What is the workplace of {person}?"
    ],
    "pet_name": [
        "How is the pet belonging to {person} named?",
        "What title was given to {person}'s pet?"
    ],
    "favorite_food": [
        "Which culinary dish is preferred above all by {person}?",
        "What is {person}'s top choice for food?"
    ]
}

def generate_dataset(num_entities: int = 300, num_sessions: int = 6):
    random.seed(RANDOM_SEED)
    
    # 1. Generate unique people
    people_names = set()
    while len(people_names) < num_entities:
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        people_names.add(name)
        
    people_list = list(people_names)
    random.shuffle(people_list)
    
    attributes = ["city", "employer", "pet_name", "favorite_food"]
    
    # Pre-generate distinct values pools
    invented_employers = [f"{e} {s}" for e in EMPLOYERS for s in EMPLOYER_SUFFIXES]
    invented_pets = [f"{p}-{random.randint(10, 99)}" for p in PET_PREFIXES for _ in range(10)]
    
    # Helper to get a random non-duplicate value
    def get_new_value(attr_type, current_val=None):
        if attr_type == "city":
            choices = [c for c in CITIES if c != current_val]
        elif attr_type == "employer":
            choices = [e for e in invented_employers if e != current_val]
        elif attr_type == "pet_name":
            choices = [p for p in invented_pets if p != current_val]
        else:
            choices = [f for f in FOODS if f != current_val]
        return random.choice(choices)

    # 2. Build initial fact records (300 people * 4 attributes = 1200 facts)
    all_facts = []
    fact_counter = 1
    
    for person in people_list:
        for attr in attributes:
            val = get_new_value(attr)
            fact_rec = {
                "id": f"fact_{fact_counter:04d}",
                "person": person,
                "attribute": attr,
                "value": val,
                "initial_session": 1,  # will be assigned to specific session below
                "current_version": 1
            }
            all_facts.append(fact_rec)
            fact_counter += 1

    random.shuffle(all_facts)
    
    # 3. Distribute initial facts into 6 sessions (~200 per session)
    facts_per_session = len(all_facts) // num_sessions
    session_facts: Dict[int, List[Dict[str, Any]]] = {s: [] for s in range(1, num_sessions + 1)}
    
    for idx, fact in enumerate(all_facts):
        s_num = min((idx // facts_per_session) + 1, num_sessions)
        fact["initial_session"] = s_num
        fact["session"] = s_num
        fact["version"] = 1
        session_facts[s_num].append(fact)

    # State tracker for ground-truth: (person, attribute) -> list of version dicts
    # Each version entry: {version, value, introduced_session}
    ground_truth_history: Dict[tuple, List[Dict[str, Any]]] = {}
    
    for s_num in range(1, num_sessions + 1):
        for fact in session_facts[s_num]:
            key = (fact["person"], fact["attribute"])
            ground_truth_history[key] = [{
                "version": 1,
                "value": fact["value"],
                "session": s_num
            }]

    # 4. Generate updates (15% of accumulated facts from Session 3 onward)
    updated_facts_log = []
    
    for s_num in range(3, num_sessions + 1):
        # Gather all existing (person, attribute) keys active up to session s_num
        active_keys = [k for k, v_list in ground_truth_history.items() if v_list[0]["session"] < s_num]
        num_updates = int(len(active_keys) * 0.15)
        keys_to_update = random.sample(active_keys, num_updates)
        
        for key in keys_to_update:
            person, attr = key
            latest_version_record = ground_truth_history[key][-1]
            old_val = latest_version_record["value"]
            new_val = get_new_value(attr, old_val)
            new_version_num = latest_version_record["version"] + 1
            
            update_fact_record = {
                "id": f"update_{s_num}_{person.replace(' ', '_')}_{attr}",
                "person": person,
                "attribute": attr,
                "value": new_val,
                "old_value": old_val,
                "session": s_num,
                "version": new_version_num,
                "is_update": True
            }
            
            # Add update fact to session training data
            session_facts[s_num].append(update_fact_record)
            
            # Update ground truth history log
            ground_truth_history[key].append({
                "version": new_version_num,
                "value": new_val,
                "session": s_num,
                "old_value": old_val
            })
            updated_facts_log.append(update_fact_record)

    # 5. Build Training & Testing files per session
    output_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(output_dir, exist_ok=True)

    session_summary = []

    for s_num in range(1, num_sessions + 1):
        train_examples = []
        
        # Training dataset construction
        for fact in session_facts[s_num]:
            person = fact["person"]
            attr = fact["attribute"]
            val = fact["value"]
            
            # 3 statement paraphrases
            for stmt_tmpl in STATEMENT_TEMPLATES[attr]:
                text = stmt_tmpl.format(person=person, value=val)
                train_examples.append({
                    "type": "statement",
                    "person": person,
                    "attribute": attr,
                    "value": val,
                    "text": text,
                    "session": s_num
                })
                
            # 3 training QA paraphrases
            for q_tmpl, a_tmpl in TRAIN_QA_TEMPLATES[attr]:
                q_text = q_tmpl.format(person=person)
                a_text = a_tmpl.format(value=val)
                train_examples.append({
                    "type": "qa",
                    "person": person,
                    "attribute": attr,
                    "value": val,
                    "question": q_text,
                    "answer": a_text,
                    "formatted_prompt": f"Question: {q_text}\nAnswer: {a_text}",
                    "session": s_num
                })
                
        # Save session training JSONL
        train_path = os.path.join(output_dir, f"session_{s_num}_train.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_examples:
                f.write(json.dumps(item) + "\n")

        # Test evaluation sets construction for session s_num
        # (new, old, update, stale)
        test_new = []
        test_old = []
        test_update = []
        test_stale = []
        
        for key, ver_history in ground_truth_history.items():
            person, attr = key
            # Find the active version at session s_num
            active_versions = [v for v in ver_history if v["session"] <= s_num]
            if not active_versions:
                continue
                
            current_active = active_versions[-1]
            intro_session = ver_history[0]["session"]
            
            # Formulate held-out test questions (2 templates)
            for test_q_tmpl in TEST_QA_TEMPLATES[attr]:
                q_str = test_q_tmpl.format(person=person)
                
                eval_item = {
                    "person": person,
                    "attribute": attr,
                    "gold_value": current_active["value"],
                    "question": q_str,
                    "session_introduced": intro_session,
                    "current_version": current_active["version"],
                    "active_at_session": s_num
                }
                
                # Check if introduced in this session
                if intro_session == s_num and current_active["version"] == 1:
                    test_new.append(eval_item)
                elif intro_session < s_num and current_active["version"] == 1:
                    test_old.append(eval_item)
                
                # Check if updated in this session
                if current_active["session"] == s_num and current_active["version"] > 1:
                    test_update.append(eval_item)
                    
                    # Stale evaluation item: gold expected is old_value
                    stale_item = dict(eval_item)
                    stale_item["stale_value"] = current_active["old_value"]
                    test_stale.append(stale_item)

        # Save session test split files
        test_split_data = {
            "new": test_new,
            "old": test_old,
            "update": test_update,
            "stale": test_stale
        }
        
        for split_name, items in test_split_data.items():
            split_path = os.path.join(output_dir, f"session_{s_num}_test_{split_name}.jsonl")
            with open(split_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
                    
        session_summary.append({
            "session": s_num,
            "train_facts": len(session_facts[s_num]),
            "train_examples": len(train_examples),
            "test_new": len(test_new),
            "test_old": len(test_old),
            "test_update": len(test_update),
            "test_stale": len(test_stale)
        })

    # Save full ground truth metadata dictionary for evaluation harness lookup
    gt_path = os.path.join(output_dir, "ground_truth_history.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        # Convert tuple keys to str for JSON serialization
        gt_serializable = {f"{k[0]}|||{k[1]}": v for k, v in ground_truth_history.items()}
        json.dump(gt_serializable, f, indent=2)

    print("\nDataset Generation Complete!")
    print("=" * 60)
    for s_info in session_summary:
        print(f"Session {s_info['session']}: {s_info['train_facts']} facts ({s_info['train_examples']} train ex) | Test -> New: {s_info['test_new']}, Old: {s_info['test_old']}, Update: {s_info['test_update']}, Stale: {s_info['test_stale']}")
    print("=" * 60)
    return output_dir

if __name__ == "__main__":
    generate_dataset()
