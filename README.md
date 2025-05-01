# 🧠 XAIguiFormer Project

A reproducibility implementation of the paper: **"XAIGuiFormer: Explainable Artificial Intelligence Guided Transformer for Brain Disorder Identification"** (ICLR 2025).

---

## 📐 Architecture Overview

```
Multi-band EEG Connectomes (9 graphs per sample)
   │
   ▼
Connectome Tokenizer (GINEConv + Mean Pooling)     ← implemented by P3
   │
   ▼
dRoFE: Rotary Encoding + Demographics Injection    ← implemented by P3
   │
   ▼
Vanilla Transformer Encoder                         ← implemented by P3
   │
   ▼
Explainer (DeepLift or GradCAM)                     ← implemented by P3/P4
   │
   ▼
XAI-Guided Transformer                              ← implemented by P3
   │
   ▼
Classification Head (MLP) + Dual Loss (Eq. 14)      ← implemented by P4
```

---

## 👥 Team & Roles

- **P1(Zeina): Understanding and Extraction** — Paper Deep Dive & Method Lead  
  `xaiguiformer_plan.md`, defines architecture, splits tasks, clarifies logic.

- **P2(Ghalia): Data & Preprocessing**  
  Implements EEG cleaning + graph construction. Uses MNE + PyPREP. See `data_pipeline/`.

- **P3&P1(Habibata et Zeina): Model Implementation**  
  Codes `tokenizer.py`, `drofe.py`, `transformer.py`, `xaiguided_transformer.py`, `explainer.py`

- **P4(Nour): Training, Loss & Reproduction**  
  Manages `train.py`, `loss.py`, optimizer, metrics (BAC, AUROC), training loop

- **P5(Safae): Report & Writing**  
  Fills out `report/README.md`, reproduces results, ablation tables, interpretability visualizations

---

## 📂 File Structure

```
├── tokenizer.py                 # GNN encoder per band
├── drofe.py                    # Rotary encoding with demographics
├── transformer.py              # Vanilla transformer block
├── xaiguided_transformer.py    # Refined attention using Qexpl/Kexpl
├── explainer.py                # DeepLift / GradCAM wrapper
├── loss.py                     # Combined loss (Eq. 14)
├── train.py                    # Training pipeline
│
├── data_pipeline/             # EEG preprocessing & graph construction
│   └── README.md              # Step-by-step preprocessing plan
│
├── report/                    # Reproducibility report assets
│   └── README.md
│
├── xaiguiformer_plan.md      # Project plan from Person 1 (Zeina)
```


## 📚 Paper Reference

- [XAIguiFormer – ICLR 2025 Paper (PDF)](https://openreview.net/pdf?id=AD5yx2xq8R)
- [Official GitHub Code](https://github.com/HanningGuo/XAIguiFormer)

