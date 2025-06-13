import sys
import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

# pour utiliser nn.Sequential dans l'explainer
import torch.nn as nn_module

# ajout du chemin parent pour les imports relatifs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# imports locaux
from XAIguiFormer_Project_Scaffold.Tokenizer import ConnectomeTokenizer
from XAIguiFormer_Project_Scaffold.RotaryFrequencyDemographicEncoding import (
    RotaryFrequencyDemographicEncoding, FREQUENCY_BOUNDS, FL, FU
)
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import XAIGuidedTransformer, ClassificationHead
from XAIguiFormer_Project_Scaffold.TransformerBlock import TransformerEncoder
from build_graph import build_graphs_from_subject
from explainer import XAIExplainer


def eval_metrics(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int):
    probs    = F.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true_np= y_true.detach().cpu().numpy()
    y_pred   = probs.argmax(axis=1)
    y_true_1h= np.eye(num_classes)[y_true_np]

    bac   = balanced_accuracy_score(y_true_np, y_pred)
    sens  = recall_score(y_true_np, y_pred, average="macro", zero_division=0)
    aucpr = average_precision_score(y_true_1h, probs, average="macro")
    aucroc= roc_auc_score(y_true_1h, probs, average="macro", multi_class="ovr")
    return bac, sens, aucpr, aucroc


class SubjectDataset(Dataset):
    def __init__(self, subject_list):
        self.subjects = subject_list

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        return (
            subj["coh"],
            subj["wpli"],
            torch.tensor([subj["age"]], dtype=torch.float32),
            torch.tensor([subj["gender"]], dtype=torch.float32),
            torch.tensor(int(subj["label"]), dtype=torch.long)
        )


class XAIModelTrainer:
    def __init__(self, hidden_dim=64, out_dim=128, num_heads=4, num_layers=2, num_classes=10, lr=1e-4):
        self.tokenizer         = ConnectomeTokenizer(in_channels=1, hidden_dim=hidden_dim, out_dim=out_dim)
        self.rotary            = RotaryFrequencyDemographicEncoding(d_model=out_dim, frequency_bounds=FREQUENCY_BOUNDS)
        self.vanilla_transformer= TransformerEncoder(dim=out_dim, num_heads=num_heads, num_layers=num_layers)
        self.xai_transformer   = XAIGuidedTransformer(dim=out_dim, num_heads=num_heads, num_layers=num_layers, drofe_fn=self.rotary)
        self.classifier_head   = ClassificationHead(d_model=out_dim, num_classes=num_classes)
        self.explainer         = XAIExplainer(nn_module.Sequential(self.vanilla_transformer, self.classifier_head))
        self.optimizer         = optim.Adam(self.parameters(), lr=lr)
        self.num_classes       = num_classes

    def parameters(self):
        return (
            list(self.tokenizer.parameters()) +
            list(self.rotary.parameters()) +
            list(self.vanilla_transformer.parameters()) +
            list(self.xai_transformer.parameters()) +
            list(self.classifier_head.parameters())
        )

    def xai_guided_loss(self, y_pred_vanilla, y_pred_xai, y_true, alpha=0.7):
        return (1 - alpha) * F.cross_entropy(y_pred_vanilla, y_true) + alpha * F.cross_entropy(y_pred_xai, y_true)

    def train(self, train_loader, val_loader=None, epochs=30):
        self.tokenizer.train()
        self.vanilla_transformer.train()
        self.xai_transformer.train()
        self.classifier_head.train()

        history = {"loss": [], "bac": [], "sens": [], "aucpr": []}

        for epoch in range(epochs):
            print(f">>> Époque {epoch+1}/{epochs}", flush=True)
            total_loss = 0.0
            all_logits, all_labels = [], []

            for coh, wpli, age, gender, label in train_loader:
                graphs      = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
                tokens      = self.tokenizer([graphs])
                tokens_drofe= self.rotary(tokens, age, gender)

                # forward vanilla
                out_vanilla = self.vanilla_transformer(tokens_drofe)
                log_vanilla = self.classifier_head(out_vanilla)

                # attributions
                attributions= self.explainer.explainer.attribute(
                    tokens_drofe,
                    baselines=torch.zeros_like(tokens_drofe),
                    target=label
                )
                Qexpl = Kexpl = attributions.detach()

                # forward XAI-guided
                out_xai     = self.xai_transformer(tokens_drofe, Qexpl, Kexpl, fl=FL, fu=FU, age=age, gender=gender)
                log_xai     = self.classifier_head(out_xai)

                # loss + backward
                loss = self.xai_guided_loss(log_vanilla, log_xai, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_logits.append(log_xai.detach().cpu())
                all_labels.append(label.detach().cpu())

            # métriques sur l'ensemble de l'époque
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            bac, sens, aucpr, _ = eval_metrics(all_logits, all_labels, num_classes=self.num_classes)
            avg_loss = total_loss / len(train_loader)

            print(f"[Train] Loss {avg_loss:.4f} | BAC {bac*100:.2f}% | Sens {sens*100:.2f}% | AUC-PR {aucpr:.4f}", flush=True)
            history["loss"].append(avg_loss)
            history["bac"].append(bac*100)
            history["sens"].append(sens*100)
            history["aucpr"].append(aucpr)

            if val_loader is not None:
                self.evaluate(val_loader, set_name="Validation")

        # tracé loss / BAC / sens / AUC-PR
        epochs_range = list(range(1, epochs+1))
        plt.figure();   plt.plot(epochs_range, history["loss"]);   plt.title("Train Loss");   plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()
        plt.figure();   plt.plot(epochs_range, history["bac"]);    plt.title("Train BAC (%)"); plt.xlabel("Epoch"); plt.ylabel("BAC"); plt.show()
        plt.figure();   plt.plot(epochs_range, history["sens"]);   plt.title("Train Sensitivity (%)"); plt.xlabel("Epoch"); plt.ylabel("Sens"); plt.show()
        plt.figure();   plt.plot(epochs_range, history["aucpr"]);  plt.title("Train AUC-PR"); plt.xlabel("Epoch"); plt.ylabel("AUC-PR"); plt.show()

    @torch.no_grad()
    def evaluate(self, dataloader, set_name="Validation"):
        self.tokenizer.eval()
        self.vanilla_transformer.eval()
        self.xai_transformer.eval()
        self.classifier_head.eval()

        total_loss = 0.0
        all_logits, all_labels = [], []

        for coh, wpli, age, gender, label in dataloader:
            graphs       = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
            tokens       = self.tokenizer([graphs])
            tokens_drofe = self.rotary(tokens, age, gender)

            # on réutilise DeepLIFT pour guider aussi en éval
            attributions = self.explainer.explainer.attribute(
                tokens_drofe,
                baselines=torch.zeros_like(tokens_drofe),
                target=label
            )
            Qexpl = Kexpl = attributions.detach()
            out_xai = self.xai_transformer(tokens_drofe, Qexpl, Kexpl, fl=FL, fu=FU, age=age, gender=gender)
            log_xai = self.classifier_head(out_xai)

            loss = F.cross_entropy(log_xai, label)
            total_loss += loss.item()
            all_logits.append(log_xai.detach().cpu())
            all_labels.append(label.detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        bac, sens, aucpr, _ = eval_metrics(all_logits, all_labels, num_classes=self.num_classes)
        avg_loss = total_loss / len(dataloader)

        print(f"[{set_name}] Loss {avg_loss:.4f} | BAC {bac*100:.2f}% | Sens {sens*100:.2f}% | AUC-PR {aucpr:.4f}", flush=True)

    @torch.no_grad()
    def visualize_attributions(self, dataloader):
        for coh, wpli, age, gender, label in dataloader:
            graphs = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
            tokens = self.tokenizer([graphs])
            tokens_drofe = self.rotary(tokens, age, gender)
            self.explainer.explain_and_plot(
                dataloader=torch.utils.data.DataLoader(
                    [(coh,wpli,age,gender,label)], batch_size=1
                ),
                tokenizer=self.tokenizer,
                rotary=self.rotary,
                build_graphs_fn=build_graphs_from_subject,
                fl=FL,
                fu=FU,
                title="Sample Attribution"
            )


if __name__ == "__main__":
    # ---- Préparation des données ----
    training_path = "/Users/user/Desktop/TDBRAIN-PRE-PROCESSED-v4/raw/train"
    list_of_subjects = []
    for subj in os.listdir(training_path):
        subj_dir = os.path.join(training_path, subj)
        if not os.path.isdir(subj_dir): continue
        prefix = os.path.join(subj_dir, f"{subj}_EC_")
        coh  = np.load(prefix + "coherence.npy", allow_pickle=True)
        wpli = np.load(prefix + "wpli.npy",      allow_pickle=True)
        demo = np.load(prefix + "demographics.npy", allow_pickle=True)
        age, gender = demo[0]
        lbl_arr = np.load(prefix + "label.npy", allow_pickle=True)
        label   = int(lbl_arr) if hasattr(lbl_arr, "item") else int(lbl_arr)
        list_of_subjects.append({
            "coh": coh, "wpli": wpli,
            "age": age, "gender": gender,
            "label": label
        })

    train_subj, val_subj = train_test_split(list_of_subjects, test_size=0.2, random_state=42)
    train_loader = DataLoader(SubjectDataset(train_subj), batch_size=1, shuffle=True)
    val_loader   = DataLoader(SubjectDataset(val_subj),   batch_size=1, shuffle=False)

    # ---- Entraînement et évaluation ----
    trainer = XAIModelTrainer()
    print(">>> Création du trainer et démarrage de l'entraînement", flush=True)
    trainer.train(train_loader, val_loader=val_loader, epochs=10)

    # ---- Importance DeepLIFT sur validation ----
    print(">>> Calcul et tracé des importances DeepLIFT sur le jeu de validation", flush=True)
    explainer = XAIExplainer(nn_module.Sequential(trainer.vanilla_transformer, trainer.classifier_head))
    explainer.explain_and_plot(
        dataloader=val_loader,
        tokenizer=trainer.tokenizer,
        rotary=trainer.rotary,
        build_graphs_fn=build_graphs_from_subject,
        fl=FL,
        fu=FU,
        title="Frequency Band Importance on TDBRAIN"
    )
