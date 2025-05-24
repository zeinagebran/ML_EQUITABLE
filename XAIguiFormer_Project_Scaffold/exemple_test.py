import torch
from torch import nn
import numpy as np
from XAIguiFormer_Project_Scaffold.Tokenizer import ConnectomeTokenizer
from XAIguiFormer_Project_Scaffold.RotaryFrequencyDemographicEncoding import RotaryFrequencyDemographicEncoding
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import XAIGuidedTransformer
from XAIguiFormer_Project_Scaffold.TransformerBlock import TransformerEncoder
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import ClassificationHead
from build_graph import build_graphs_from_subject
from explainer import XAIExplainer

# ==== Hyperparamètres ====
from XAIguiFormer_Project_Scaffold.RotaryFrequencyDemographicEncoding import FREQUENCY_BOUNDS, FL, FU

# ==== Données de test ====
coh = np.random.rand(9, 26, 26).astype(np.float32)
wpli = np.random.rand(9, 26, 26).astype(np.float32)
age = torch.tensor([47.05])
gender = torch.tensor([1.0])
label = torch.tensor([1])  # classe cible pour la classification
graphs = build_graphs_from_subject(coh, wpli)

# ==== Instanciation des modules ====
tokenizer = ConnectomeTokenizer(in_channels=1, hidden_dim=64, out_dim=128)
rotary = RotaryFrequencyDemographicEncoding(d_model=128, frequency_bounds=FREQUENCY_BOUNDS)
vanilla_transformer = TransformerEncoder(dim=128, num_heads=4, num_layers=2)
xai_transformer = XAIGuidedTransformer(dim=128, num_heads=4, num_layers=2, drofe_fn=rotary)
classifier_head = ClassificationHead(d_model=128, num_classes=10)
explainer = XAIExplainer(nn.Sequential(vanilla_transformer, classifier_head))

# ==== Pipeline ====
tokens = tokenizer([graphs])                     # [1, 9, 128]
tokens_drofe = rotary(tokens, age, gender)       # [1, 9, 128]

vanilla_out = vanilla_transformer(tokens_drofe)  # [1, 9, 128]
logits_vanilla = classifier_head(vanilla_out)    # [1, 2]

# Attribution
attributions = explainer.explain(tokens_drofe, target_class=label, visualize=True)
Qexpl = Kexpl = attributions.detach()

# XAI Transformer
xai_out = xai_transformer(tokens_drofe, Qexpl=Qexpl, Kexpl=Kexpl, fl=FL, fu=FU, age=age, gender=gender)
logits_xai = classifier_head(xai_out)            # [1, 2]
# === Prédiction finale ===
probas = torch.softmax(logits_xai, dim=-1)
predicted_class = torch.argmax(probas, dim=-1).item()

print("Logits :", logits_xai)
print("Probabilités :", probas)
print("Classe prédite :", predicted_class)

# ==== Perte ====
def xai_guided_loss(y_pred_vanilla, y_pred_xai, y_true, alpha=0.7):
    return (1 - alpha) * nn.functional.cross_entropy(y_pred_vanilla, y_true) + \
            alpha * nn.functional.cross_entropy(y_pred_xai, y_true)

loss = xai_guided_loss(logits_vanilla, logits_xai, label)
print(f"Perte totale : {loss.item():.4f}")
