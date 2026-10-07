# Reproduction Guide: Memory vs Weights Benchmark

This guide outlines step-by-step instructions to reproduce all experimental findings, dataset generation, model evaluations, and benchmark plots.

---

## 1. Local Environment Setup

Ensure Python 3.10 or 3.11 is installed.

```bash
git clone https://github.com/pbhoomikadesai/continual-knowledge-updating.git
cd continual-knowledge-updating

# Install dependencies
pip install -r requirements.txt

# Run hardware and model loading sanity check
python verify_env.py
```

---

## 2. Phase 2: Synthetic Dataset Generation

Generate the synthetic dataset (300 fictional entities, 4 attributes, 6 sessions, 15% updates):

```bash
# Generate JSONL datasets under data/processed/
python data/generator.py

# Run dataset unit tests (asserts no leakage, entity uniqueness, value consistency)
python -m pytest data/test_dataset.py
```

---

## 3. Phase 3: Base Floor & Oracle Ceiling Baseline

Run the baseline evaluation harness:

```bash
python eval/evaluate.py
```

---

## 4. Phase 4: Arm A1 - Hybrid RAG Memory Store

Evaluate non-parametric retrieval memory with BM25 + Dense FAISS index:

```bash
python arms/arm1_rag.py
```

---

## 5. Phase 5 & 6: Sequential LoRA & Replay (Arms A2 & A3)

Run parametric weight adaptation experiments:

```bash
# Arm A2: Vanilla Sequential LoRA
python arms/arm2_sequential_lora.py

# Arm A3: LoRA + 20% Experience Replay Buffer
python arms/arm3_lora_replay.py
```

---

## 6. Phase 7: Running Full Benchmark on Kaggle (T4 GPU)

For fast execution or multi-seed runs (`seed = 42, 123, 999`):

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and create a New Notebook.
2. Select **GPU T4 x2** under `Accelerator` in the notebook settings menu.
3. Upload the repository files or clone directly:
   ```bash
   !git clone https://github.com/pbhoomikadesai/continual-knowledge-updating.git
   %cd continual-knowledge-updating
   !pip install -r requirements.txt
   ```
4. Execute all arms sequentially and export `results/metrics_summary.csv`.

---

## 7. Phase 8: Plotting & Analysis

Generate high-resolution figures:

```bash
python results/plot_results.py
```

All output figures are exported to `results/figures/`.
