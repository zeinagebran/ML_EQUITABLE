
# XAIguiFormer – Project Implementation Plan

This file outlines the model architecture, module breakdown, team assignments.

## Architecture Summary
- Connectome Tokenizer → GNN encoder per frequency band
- dRoFE → rotary embedding with demographic injection
- Vanilla Transformer → standard transformer
- Explainer (DeepLift) → extracts Qexpl/Kexpl
- XAI-Guided Transformer → attention using Qexpl/Kexpl
- Loss → dual prediction (Eq. 14)

## Modules
- tokenizer.py
- drofe.py
- transformer.py
- xaiguided_transformer.py
- explainer.py
- loss.py
- train.py
- data_pipeline/
- report/

## Team Tasks
Person 1: this plan, task delegation, architecture lead  
Person 2: Preprocessing and connectome construction (see appendix A.2, A.3)  
Person 3: Model implementation (all modules above)  
Person 4: Training, metrics, hyperparams  
Person 5: Report and result analysis (Tables 1–3, figures)
