import torch
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Définition des bandes de fréquence
FREQUENCY_BOUNDS = {
    "Delta": [2, 4],
    "Theta": [4, 8],
    "Low Alpha": [8, 10],
    "High Alpha": [10, 12],
    "Low Beta": [12, 18],
    "Mid Beta": [18, 21],
    "High Beta": [21, 30],
    "Low Gamma": [30, 45],
    "Theta/Beta": [4, 30]
}
BAND_NAMES = list(FREQUENCY_BOUNDS.keys())

class XAIExplainer:
    def __init__(self, model, device=None):
        """
        model: nn.Module (par ex. transformer + classification head)
        device: torch device ("cpu" ou "cuda")
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.explainer = DeepLift(self.model)

    def explain_dataset(self, dataloader, tokenizer, rotary, build_graphs_fn, fl, fu):
        """
        Parcourt tout le dataloader et calcule l'attribution moyenne absolue par bande.

        Args:
            dataloader : DataLoader retournant (coh, wpli, age, gender, label)
            tokenizer  : fonction de tokenisation des graphes
            rotary     : couche RotaryFrequencyDemographicEncoding
            build_graphs_fn : fonction build_graphs_from_subject
            fl, fu     : bornes fluorescentes pour XAIGuidedTransformer

        Returns:
            scores : np.array de taille (n_bands,) avec attribution moyenne
        """
        n_bands = len(BAND_NAMES)
        acc_scores = torch.zeros(n_bands, device="cpu")
        n_samples = 0

        self.model.eval()
        for coh, wpli, age, gender, label in dataloader:
            # reconstruction du graphe
            graphs = build_graphs_fn(coh.squeeze(0).numpy(), wpli.squeeze(0).numpy())
            tokens = tokenizer([graphs]).to(self.device)
            tokens = tokens.clone().detach().requires_grad_(True)
            tokens = rotary(tokens, age.to(self.device), gender.to(self.device))

            # baseline zéro
            baseline = torch.zeros_like(tokens)
            target = label.to(self.device)

            # attribution DeepLIFT
            attributions = self.explainer.attribute(
                tokens,
                baselines=baseline,
                target=target
            )  # [1, n_bands, D]

            # moyenne sur la dimension D
            band_scores = attributions.mean(dim=-1).squeeze(0).detach().cpu().abs()  # [n_bands]
            acc_scores += band_scores
            n_samples += 1

        mean_scores = (acc_scores / n_samples).numpy()  # normalisation
        return mean_scores

    def plot_dataset_importance(self, scores, title="Fréquence Band Importance"):
        """
        Trace un barh trié des scores d'attribution par bande.

        Args:
            scores : array-like de taille n_bands
            title  : titre du graphique
        """
        df = pd.DataFrame({
            "band": BAND_NAMES,
            "score": scores
        })
        df = df.sort_values("score", ascending=False)

        plt.figure(figsize=(6, 4))
        plt.barh(df["band"], df["score"])
        plt.gca().invert_yaxis()
        plt.xlabel("Attribution moyenne absolue (DeepLIFT)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def explain_and_plot(self, dataloader, tokenizer, rotary, build_graphs_fn, fl, fu, title=None):
        """
        Combine explain_dataset et plot_dataset_importance en une seule étape.
        """
        scores = self.explain_dataset(dataloader, tokenizer, rotary, build_graphs_fn, fl, fu)
        self.plot_dataset_importance(scores, title or "Fréquence Band Importance")
