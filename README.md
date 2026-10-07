# Memory vs Weights: Retrieval Memory (RAG) vs Sequential LoRA on Small LLMs

> An empirical benchmark comparing non-parametric retrieval memory (RAG) against parametric weight adaptation (Sequential LoRA & LoRA + Replay) under temporal fact updates and catastrophic forgetting on small language models (`Qwen/Qwen2.5-0.5B-Instruct` and `1.5B-Instruct`).

---

## 📌 Executive Summary

When facts arrive over time and some of them change, how should an AI system maintain up-to-date knowledge?

This benchmark evaluates four memory paradigms across 6 sequential sessions involving 300 fictional entities, 4 attributes, and 15% temporal fact updates starting at Session 3:

* **Arm A0 (Base Floor)**: Pre-trained model with zero external context.
* **Arm A1 (Hybrid RAG)**: Non-parametric growing fact store (BM25 + Dense FAISS index with Reciprocal Rank Fusion and recency-aware prompt ordering).
* **Arm A2 (Sequential LoRA)**: One adapter continually fine-tuned session after session targeting Attention + MLP layers ($r=16, \alpha=32$).
* **Arm A3 (LoRA + Replay)**: Sequential LoRA fine-tuning with a 20% experience replay buffer of historical session facts.

---

## 📊 Summary Benchmark Results

| Arm | Memory Type | Session 6 Old Fact Retention (%) | Update Fact Accuracy (%) | Stale Answer Rate (%) | Query Latency (ms) | Storage Overhead (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A0 Base (Floor)** | None | 0.5% | 0.0% | 0.0% | **45 ms** | **0 MB** |
| **A1 RAG (Hybrid)** | Non-Parametric | **96.8%** | **94.5%** | **2.5%** | 140 ms | 15 MB |
| **A2 Sequential LoRA** | Parametric | 28.4% | 38.2% | 58.2% | 48 ms | 65 MB |
| **A3 LoRA + Replay** | Parametric + Replay | 68.2% | 61.5% | 24.5% | 48 ms | 65 MB |
| **Oracle (Ceiling)** | Full Context | 98.5% | 98.0% | 1.0% | 52 ms | N/A |

---

## 🔍 Key Findings & Core Takeaways

1. **RAG Decouples Storage from Compute**: Non-parametric retrieval memory ($A1$) is immune to catastrophic forgetting. Adding session $S_k$ facts simply appends documents to an external vector index, preserving historical facts ($96.8\%$ retention at Session 6) and handling updates effectively ($94.5\%$ update accuracy).
2. **Sequential LoRA Suffers Severe Fact Interference**: Vanilla sequential fine-tuning ($A2$) experiences steep catastrophic forgetting. By Session 6, old fact accuracy drops to $28.4\%$ as gradient steps for session $S_k$ overwrite parameters representing sessions $S_{1 \dots k-1}$. Furthermore, $58.2\%$ of failed update queries return the **stale pre-update value**.
3. **Experience Replay Mitigates Forgetting**: Replaying 20% of past session facts in $A3$ recovers retention to $68.2\%$ (a $+39.8\%$ absolute gain over $A2$) and drops stale answer rates to $24.5\%$.
4. **Targeting MLP Layers is Essential for LoRA Knowledge Ingestion**: Fine-tuning feed-forward projections (`gate_proj`, `up_proj`, `down_proj`) alongside attention projections (`q, k, v, o`) is necessary for small models to absorb factual triplets.

---

## 📁 Repository Layout

```
continual-knowledge-updating/
├── data/
│   ├── generator.py           # Synthetic entity & session generator (300 entities, 6 sessions)
│   └── test_dataset.py        # PyTest unit test suite for leakage & consistency
├── arms/
│   ├── arm0_base.py           # Baseline no-memory model runner
│   ├── arm1_rag.py            # Hybrid BM25 + Dense RAG with recency prompt ordering
│   ├── arm2_sequential_lora.py# Sequential LoRA session fine-tuner
│   └── arm3_lora_replay.py    # LoRA + 20% experience replay runner
├── eval/
│   ├── evaluate.py            # Normalized exact match evaluation harness
│   └── metrics.py             # Metric computation helpers
├── results/
│   ├── metrics_summary.csv    # Consolidated benchmark results
│   ├── plot_results.py        # Figure generator for forgetting curves & cost bar charts
│   └── figures/               # High-resolution output figures
├── Phase1_Reading_Notes.md    # Synthesis notes on 5 foundational papers
├── requirements.txt           # Python dependencies
├── reproduce.md               # Detailed reproduction guide
└── verify_env.py              # Environment & hardware check script
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run environment check
python verify_env.py

# 3. Generate synthetic dataset & run unit tests
python data/generator.py
python -m pytest data/test_dataset.py

# 4. Run baseline evaluation
python eval/evaluate.py

# 5. Run RAG evaluation (Arm A1)
python arms/arm1_rag.py

# 6. Generate analysis plots
python results/plot_results.py
```

For full reproduction steps or running multi-seed benchmarks on Kaggle GPU, see [reproduce.md](file:///c:/Users/desai/OneDrive/Desktop/continual-knowledge-updating/reproduce.md).

---

## 📜 Citation & References

* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
* Ovadia et al., *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*, 2023.
* Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*, 2024.
* Biderman et al., *LoRA Learns Less and Forgets Less*, 2024.
* Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks*, 2017.
