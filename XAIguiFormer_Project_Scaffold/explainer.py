# TODO: DeepLift / GradCAM
import torch
from captum.attr import DeepLift
import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, model):
        """
        model: nn.Module (ex: transformer + classifier)
        """
        self.model = model
        self.explainer = DeepLift(model)

    def explain(self, tokens, target_class, visualize=True):
        """
        tokens: torch.Tensor [B, 9, D]
        target_class: int ou [B]
        visualize: bool — affiche les graphiques si True

        Returns:
            attributions: torch.Tensor [B, 9, D]
        """
        tokens = tokens.clone().detach().requires_grad_(True)
        target_tensor = torch.tensor([target_class]) if isinstance(target_class, int) else target_class

        attributions = self.explainer.attribute(tokens, target=target_tensor)  # [B, 9, D]

        if visualize:
            self.plot_importances(attributions)

        return attributions

    def plot_importances(self, attributions):
        """
        Affiche :
        - barplot des importances moyennes par token
        - heatmap des dimensions
        """
        attributions = attributions.detach().cpu()
        scores = attributions.mean(dim=-1).squeeze(0).numpy()  # [9]
        heatmap = attributions.squeeze(0).numpy()  # [9, D]

        # Bar plot
        plt.figure(figsize=(8, 4))
        plt.bar(range(9), scores)
        plt.xticks(range(9), BAND_NAMES, rotation=45)
        plt.title("Importance moyenne des tokens (DeepLIFT)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Heatmap
        plt.figure(figsize=(10, 4))
        sns.heatmap(heatmap, cmap="viridis", xticklabels=False)
        plt.yticks(range(9), BAND_NAMES)
        plt.title("Attributions par dimension")
        plt.xlabel("Dimensions du token")
        plt.ylabel("Bandes EEG")
        plt.tight_layout()
        plt.show()