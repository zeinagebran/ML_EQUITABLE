# TODO: Training pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from XAIguiFormer_Project_Scaffold.explainer import XAIExplainer
from XAIguiFormer_Project_Scaffold.RotaryFrequencyDemographicEncoding import RotaryFrequencyDemographicEncoding
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import XAIGuidedTransformerBlock
from XAIguiFormer_Project_Scaffold.TransformerBlock import TransformerEncoder
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import ClassificationHead  # Assurez-vous que ce module existe

class XAIGuiFormerTrainer:
    def __init__(self, tokenizer, rotary, vanilla_transformer, xai_transformer, classifier, device='cuda'):
        self.tokenizer = tokenizer.to(device)
        self.rotary = rotary.to(device)
        self.vanilla = vanilla_transformer.to(device)
        self.xai = xai_transformer.to(device)
        self.classifier = classifier.to(device)
        self.device = device

        # Regroupe tous les paramètres
        all_params = list(self.tokenizer.parameters()) + list(self.vanilla.parameters()) + \
                     list(self.xai.parameters()) + list(self.classifier.parameters())

        self.optimizer = AdamW(all_params, lr=5e-5, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-5)

    def train_one_epoch(self, dataloader):
        self.tokenizer.train()
        self.vanilla.train()
        self.xai.train()
        self.classifier.train()

        total_loss = 0
        for graphs, age, gender, labels in dataloader:
            graphs = [g.to(self.device) for g in graphs]
            age = age.to(self.device)
            gender = gender.to(self.device)
            labels = labels.to(self.device)

            # Pipeline
            tokens = self.tokenizer(graphs)                  # [B, 9, D]
            tokens = self.rotary(tokens, age, gender)        # dRoFE
            tokens_vanilla = self.vanilla(tokens)            # Vanilla
            logits_vanilla = self.classifier(tokens_vanilla)

            # XAI Explainer
            explainer = XAIExplainer(nn.Sequential(self.vanilla, self.classifier))
            Qexpl = Kexpl = explainer.explain(tokens, target_class=labels, visualize=False).detach()
            tokens_xai = self.xai(tokens, Qexpl=Qexpl, Kexpl=Kexpl)
            logits_xai = self.classifier(tokens_xai)

            # Loss
            loss = self.xai_guided_loss(logits_vanilla, logits_xai, labels, alpha=0.7)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @staticmethod
    def xai_guided_loss(pred_vanilla, pred_xai, y_true, alpha=0.7):
        return (1 - alpha) * F.cross_entropy(pred_vanilla, y_true) + alpha * F.cross_entropy(pred_xai, y_true)


def train():
    pass