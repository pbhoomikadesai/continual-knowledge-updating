"""
Phase 4: Arm A1 - Hybrid RAG Memory Store
Implements hybrid BM25 + Dense (sentence-transformers) retriever with Reciprocal Rank Fusion (RRF).
Supports incremental session fact appending, recency-aware prompt formatting, and Recall@k logging.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from eval.evaluate import Evaluator, is_exact_or_contained_match

class HybridRAGMemory:
    def __init__(self, dense_model_name: str = "all-MiniLM-L6-v2", top_k: int = 5, rrf_k: int = 60):
        print(f"Initializing Hybrid RAG Memory (Dense: {dense_model_name}, top_k={top_k})...")
        self.top_k = top_k
        self.rrf_k = rrf_k
        
        # Models
        self.dense_encoder = SentenceTransformer(dense_model_name)
        self.embedding_dim = self.dense_encoder.get_sentence_embedding_dimension()
        
        # Memory Stores
        self.documents: List[Dict[str, Any]] = [] # stores fact text + metadata
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None
        
        # FAISS Index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
    def add_facts(self, facts: List[Dict[str, Any]]):
        """Incrementally append new session facts to memory stores."""
        new_texts = []
        new_tokens = []
        
        for fact in facts:
            person = fact["person"]
            attr = fact["attribute"]
            val = fact["value"]
            s_num = fact["session"]
            ver = fact["version"]
            
            fact_sentence = f"{person}'s {attr.replace('_', ' ')} is {val}. (Session {s_num}, Version {ver})"
            
            doc_entry = {
                "fact_sentence": fact_sentence,
                "person": person,
                "attribute": attr,
                "value": val,
                "session": s_num,
                "version": ver
            }
            self.documents.append(doc_entry)
            new_texts.append(fact_sentence)
            new_tokens.append(fact_sentence.lower().split())
            
        # Update BM25
        self.corpus_tokens.extend(new_tokens)
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Update Dense FAISS Index
        embeddings = self.dense_encoder.encode(new_texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"Added {len(facts)} facts to RAG memory store. Total documents: {len(self.documents)}")

    def retrieve_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k documents using Reciprocal Rank Fusion (BM25 + FAISS Dense)."""
        if not self.documents:
            return []
            
        # 1. BM25 Retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:20]
        
        # 2. Dense Retrieval
        query_emb = self.dense_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, dense_top_indices = self.index.search(query_emb, min(20, len(self.documents)))
        dense_top_indices = dense_top_indices[0]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}
        
        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        for rank, idx in enumerate(dense_top_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:self.top_k]
        return [self.documents[i] for i in sorted_indices]

    def format_prompt(self, retrieved_docs: List[Dict[str, Any]], question: str, variant: str = "recency_aware") -> str:
        """Format prompt with retrieved context."""
        if variant == "recency_aware":
            # Sort chronologically by session and version
            sorted_docs = sorted(retrieved_docs, key=lambda x: (x["session"], x["version"]))
            context_str = "\n".join([f"- [{doc['session']}] {doc['fact_sentence']}" for doc in sorted_docs])
            prompt = (
                f"Retrieved Memory Facts (Note: Later sessions override earlier sessions):\n"
                f"{context_str}\n\n"
                f"Question: {question}\n"
                f"Answer with ONLY the target entity value, nothing else.\nAnswer:"
            )
        else:
            # Plain formatting
            context_str = "\n".join([f"- {doc['fact_sentence']}" for doc in retrieved_docs])
            prompt = (
                f"Retrieved Context:\n"
                f"{context_str}\n\n"
                f"Question: {question}\n"
                f"Answer with ONLY the target entity value, nothing else.\nAnswer:"
            )
        return prompt

def evaluate_arm1_rag(data_dir: str = "data/processed", results_dir: str = "results", prompt_variant: str = "recency_aware"):
    os.makedirs(results_dir, exist_ok=True)
    memory = HybridRAGMemory(top_k=5)
    evaluator = Evaluator()

    summary_results = []
    per_example_results = []

    print("\n" + "=" * 60)
    print(f"Starting Arm A1 Hybrid RAG Evaluation (Variant: {prompt_variant})")
    print("=" * 60)

    for s_num in range(1, 7):
        print(f"\n--- SESSION {s_num} ---")
        
        # 1. Load session training facts and add to RAG memory index
        train_file = os.path.join(data_dir, f"session_{s_num}_train.jsonl")
        session_facts = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["type"] == "statement": # add core factual statements
                    session_facts.append(item)
                    
        memory.add_facts(session_facts)

        # 2. Evaluate on session test splits (new, old, update, stale)
        for split_name in ["new", "old", "update"]:
            test_file = os.path.join(data_dir, f"session_{s_num}_test_{split_name}.jsonl")
            if not os.path.exists(test_file):
                continue
                
            with open(test_file, "r", encoding="utf-8") as f:
                test_items = [json.loads(line) for line in f]
                
            if not test_items:
                continue

            correct_count = 0
            recall_count = 0
            stale_count = 0

            for item in test_items:
                person = item["person"]
                attr = item["attribute"]
                gold = item["gold_value"]
                question = item["question"]

                # Retrieve top-k documents
                retrieved_docs = memory.retrieve_hybrid(f"{person} {attr}")
                
                # Check Recall@k: is gold value present anywhere in retrieved docs?
                retrieved_text = " ".join([d["value"] for d in retrieved_docs])
                hit_recall = is_exact_or_contained_match(retrieved_text, gold)
                if hit_recall:
                    recall_count += 1

                # Format prompt & generate LLM response
                prompt = memory.format_prompt(retrieved_docs, question, variant=prompt_variant)
                
                # Direct generate using evaluator's tokenizer & model
                inputs = evaluator.tokenizer(prompt, return_tensors="pt").to(evaluator.device)
                with torch.no_grad():
                    outputs = evaluator.model.generate(
                        **inputs,
                        max_new_tokens=15,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=evaluator.tokenizer.eos_token_id
                    )
                pred = evaluator.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

                is_correct = is_exact_or_contained_match(pred, gold)
                if is_correct:
                    correct_count += 1

                record = {
                    "arm": "A1 RAG",
                    "session": s_num,
                    "split": split_name,
                    "person": person,
                    "attribute": attr,
                    "question": question,
                    "gold_value": gold,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "recall_hit": hit_recall,
                    "prompt_variant": prompt_variant
                }

                if split_name == "update" and "stale_value" in item:
                    is_stale = is_exact_or_contained_match(pred, item["stale_value"])
                    if is_stale:
                        stale_count += 1
                    record["stale_value"] = item["stale_value"]
                    record["is_stale"] = is_stale

                per_example_results.append(record)

            accuracy = (correct_count / len(test_items)) * 100.0
            recall_k = (recall_count / len(test_items)) * 100.0
            stale_rate = (stale_count / len(test_items)) * 100.0 if split_name == "update" else 0.0

            print(f"Session {s_num} [{split_name.upper()}]: Accuracy = {accuracy:.2f}% | Recall@5 = {recall_k:.2f}% | Stale Rate = {stale_rate:.2f}%")

            summary_results.append({
                "Arm": f"A1 RAG ({prompt_variant})",
                "Session": s_num,
                "Split": split_name,
                "Accuracy": accuracy,
                "Recall@k": recall_k,
                "Stale_Rate": stale_rate,
                "Description": f"Hybrid RAG top-5 top-k recall and generation accuracy"
            })

    # Log results to CSV
    summary_df = pd.DataFrame(summary_results)
    
    # Append to metrics_summary.csv if it exists
    summary_csv_path = os.path.join(results_dir, "metrics_summary.csv")
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_csv(summary_csv_path, index=False)
    else:
        summary_df.to_csv(summary_csv_path, index=False)

    print(f"\nArm A1 RAG evaluation completed! Saved results to {results_dir}/metrics_summary.csv")

if __name__ == "__main__":
    evaluate_arm1_rag()
