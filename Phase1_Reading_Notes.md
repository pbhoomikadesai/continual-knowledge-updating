# Phase 1: Literature Synthesis & Reading Notes

A structured synthesis of the 5 key foundational papers examining parametric fine-tuning vs. non-parametric retrieval memory for language models.

---

### 1. LoRA: Low-Rank Adaptation of Large Language Models
* **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
* **Key Findings**: Freezes the pre-trained model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices $W_0 + \Delta W = W_0 + B A$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. Reduces trainable parameter overhead by ~10,000x and GPU memory footprint by 3x while matching full fine-tuning quality.
* **Impact on Project Design**: In sequential training ($A2$ & $A3$), LoRA provides computationally efficient parameter updates. However, setting target modules is critical: targeting both Attention (`q, k, v, o`) and MLP layers (`gate, up, down_proj`) is necessary to inject new factual memory into transformer feed-forward blocks.

---

### 2. Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs
* **Authors**: Ovadia et al. / Kandpal et al. (2023)
* **Key Findings**: Systematically compares fine-tuning vs. Retrieval-Augmented Generation (RAG) for factual knowledge injection across varying factual frequencies. RAG overwhelmingly outperforms fine-tuning on acquiring unseen/unfamiliar facts, context adaptation, and resilience to fact updates. Fine-tuning requires hundreds of repetitions to internalize explicit facts without guarantee of generalized answer generation.
* **Impact on Project Design**: Establishes the core hypothesis: Arm $A1$ (RAG) will achieve high single-session accuracy on new facts, whereas Arms $A2$/$A3$ (LoRA) will require carefully tuned learning rates and paraphrase augmentation to absorb facts without hallucinating.

---

### 3. Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?
* **Authors**: Zorik Gekhman, Gal Yona, Roee Aharoni, Jonathan Herzig, Ndapa Nakashole, Mor Geva (2024)
* **Key Findings**: Fine-tuning LLMs on factual information that is completely absent from their pre-training distribution forces the model to fit new mapping distributions without underlying grounding, significantly increasing hallucination rates on surrounding queries.
* **Impact on Project Design**: Reinforces the requirement for our synthetic dataset (Phase 2): fictional entity names, invented employers, and unique pet strings ensure that the pre-trained model has zero prior knowledge, making hallucination detection unambiguous during evaluation.

---

### 4. LoRA Learns Less and Forgets Less
* **Authors**: Dan Biderman et al. / Xia et al. (2024)
* **Key Findings**: Compares LoRA against full fine-tuning on continual task adaptation. LoRA regularizes model capacity to a low-rank subspace ($\text{rank } r$), causing it to absorb novel out-of-domain knowledge slower ("learns less"), but drastically slowing down the degradation of pre-trained and historical representations ("forgets less").
* **Impact on Project Design**: Predicts that sequential LoRA ($A2$) will experience smoother, slower catastrophic forgetting curves compared to full fine-tuning, but will necessitate sufficient training epochs (3-8) per session to achieve high immediate session accuracy.

---

### 5. Overcoming Catastrophic Forgetting in Neural Networks (EWC)
* **Authors**: James Kirkpatrick et al. (DeepMind, 2017)
* **Key Findings**: When a neural network is sequentially trained on Task A then Task B, gradient updates for Task B overwrite parameter configurations essential for Task A (catastrophic forgetting). Elastic Weight Consolidation (EWC) slows down learning on weights sensitive to previous tasks, while Experience Replay periodically re-introduces historical samples to maintain decision boundaries.
* **Impact on Project Design**: Direct motivation for Arm $A3$ (LoRA + 20% Replay Buffer). Replaying past session factual samples during session $S_k$ training provides a practical baseline against vanilla sequential adaptation ($A2$).

---

### Core Takeaway Summary

> **Why Sequential Fine-Tuning Forgets**: Gradient steps in session $S_k$ overwrite weight matrices parameterizing factual associations learned in sessions $S_{1 \dots k-1}$ because weight space is shared and non-orthogonal.
>
> **Why Retrieval Memory Does Not Forget**: RAG decouples storage from compute. Adding session $S_k$ facts simply appends new documents to an external index, leaving parametric weights untouched and preserving past facts intact.
