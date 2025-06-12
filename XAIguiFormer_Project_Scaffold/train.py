import sys
import os

# Add the parent directory of XAIguiFormer_Project_Scaffold to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from XAIguiFormer_Project_Scaffold.Tokenizer import ConnectomeTokenizer
from XAIguiFormer_Project_Scaffold.RotaryFrequencyDemographicEncoding import (
    RotaryFrequencyDemographicEncoding, FREQUENCY_BOUNDS, FL, FU
)
from XAIguiFormer_Project_Scaffold.xaiguided_transformer import XAIGuidedTransformer, ClassificationHead
from XAIguiFormer_Project_Scaffold.TransformerBlock import TransformerEncoder
from build_graph import build_graphs_from_subject
from explainer import XAIExplainer
import os.path as osp
import os



# === Dataset Class ===
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
            torch.tensor(subject["label"].item(), dtype=torch.long)
        )


# === Trainer Class ===
class XAIModelTrainer:
    def __init__(self, hidden_dim=64, out_dim=128, num_heads=4, num_layers=2, num_classes=10, lr=1e-4):
        self.tokenizer = ConnectomeTokenizer(in_channels=1, hidden_dim=hidden_dim, out_dim=out_dim)
        self.rotary = RotaryFrequencyDemographicEncoding(d_model=out_dim, frequency_bounds=FREQUENCY_BOUNDS)

        self.vanilla_transformer = TransformerEncoder(dim=out_dim, num_heads=num_heads, num_layers=num_layers)
        self.xai_transformer = XAIGuidedTransformer(dim=out_dim, num_heads=num_heads, num_layers=num_layers, drofe_fn=self.rotary)
        self.classifier_head = ClassificationHead(d_model=out_dim, num_classes=num_classes)

        # Explainer and optimizer
        self.explainer = XAIExplainer(nn.Sequential(self.vanilla_transformer, self.classifier_head))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        return list(self.tokenizer.parameters()) + \
               list(self.rotary.parameters()) + \
               list(self.vanilla_transformer.parameters()) + \
               list(self.xai_transformer.parameters()) + \
               list(self.classifier_head.parameters())

    def xai_guided_loss(self, y_pred_vanilla, y_pred_xai, y_true, alpha=0.7):
        return (1 - alpha) * nn.functional.cross_entropy(y_pred_vanilla, y_true) + \
               alpha * nn.functional.cross_entropy(y_pred_xai, y_true)

    def train(self, train_loader, val_loader=None, epochs=5):
        self.tokenizer.train()
        self.vanilla_transformer.train()
        self.xai_transformer.train()
        self.classifier_head.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for coh, wpli, age, gender, label in train_loader:
                graphs = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
                tokens = self.tokenizer([graphs])  # [1, 9, 128]
                tokens_drofe = self.rotary(tokens, age, gender)

                # Vanilla forward
                vanilla_out = self.vanilla_transformer(tokens_drofe)
                logits_vanilla = self.classifier_head(vanilla_out)

                # XAI attributions
                attributions = self.explainer.explain(tokens_drofe, target_class=label, visualize=False)
                Qexpl = Kexpl = attributions.detach()

                # XAI forward
                xai_out = self.xai_transformer(tokens_drofe, Qexpl, Kexpl, fl=FL, fu=FU, age=age, gender=gender)
                logits_xai = self.classifier_head(xai_out)

                # Loss and backprop
                loss = self.xai_guided_loss(logits_vanilla, logits_xai, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Stats
                total_loss += loss.item()
                pred = torch.argmax(torch.softmax(logits_xai, dim=-1), dim=-1)
                correct += (pred == label).sum().item()
                total += 1

            acc = 100 * correct / total
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

            if val_loader is not None:
                self.evaluate(val_loader, set_name="Validation")

    @torch.no_grad()
    def evaluate(self, dataloader, set_name="Validation"):
        self.tokenizer.eval()
        self.vanilla_transformer.eval()
        self.xai_transformer.eval()
        self.classifier_head.eval()

        correct = 0
        total = 0

        for coh, wpli, age, gender, label in dataloader:
            graphs = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
            tokens = self.tokenizer([graphs])
            tokens_drofe = self.rotary(tokens, age, gender)

            # Vanilla + XAI forward
            attributions = self.explainer.explain(tokens_drofe, target_class=label, visualize=False)
            Qexpl = Kexpl = attributions.detach()
            xai_out = self.xai_transformer(tokens_drofe, Qexpl, Kexpl, fl=FL, fu=FU, age=age, gender=gender)
            logits_xai = self.classifier_head(xai_out)

            pred = torch.argmax(torch.softmax(logits_xai, dim=-1), dim=-1)
            correct += (pred == label).sum().item()
            total += 1

        acc = 100 * correct / total
        print(f"{set_name} Accuracy: {acc:.2f}%")
    
    @torch.no_grad()
    def visualize_attributions(self, dataloader):
        for coh, wpli, age, gender, label in dataloader:
            graphs = build_graphs_from_subject(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
            tokens = self.tokenizer([graphs])  # [1, 9, 128]
            tokens_drofe = self.rotary(tokens, age, gender)

            # Compute attributions
            attributions = self.explainer.explain(tokens_drofe, target_class=label, visualize=True)
        

if __name__ == "__main__":
    training_path = "/Users/t/Documents/Telecom/projet_XAI/data/TDBRAIN/raw/train"
    list_of_subjects = []
    raw_file_names = [
                folder_name for folder_name in os.listdir(training_path) if os.path.isdir(osp.join(training_path, folder_name))
            ]
    raw_paths = [osp.join(training_path, file_name, f"{file_name}_EC_") for file_name in raw_file_names]
    for raw_path in raw_paths:
        demographics_info = np.load(raw_path + 'demographics.npy', allow_pickle=True),
        demographics_array = demographics_info[0]
        age, gender = demographics_array[0]  
        subject = {
            "coh": np.load(raw_path + 'coherence.npy', allow_pickle=True),
            "wpli": np.load(raw_path + 'wpli.npy', allow_pickle=True),
            "age": age,
            "gender": gender,
            "label": np.load(raw_path + 'label.npy', allow_pickle=True),
        }

        list_of_subjects.append(subject)
    
    train_subjects, val_subjects = train_test_split(list_of_subjects, test_size=0.2, random_state=42)

    train_loader = DataLoader(SubjectDataset(train_subjects), batch_size=1, shuffle=True)
    val_loader = DataLoader(SubjectDataset(val_subjects), batch_size=1, shuffle=False)

    trainer = XAIModelTrainer()
    trainer.train(train_loader, val_loader=val_loader, epochs=10)

    # Visualize attributions after training
    print("Visualizing attributions for training data...")
    trainer.visualize_attributions(train_loader)
