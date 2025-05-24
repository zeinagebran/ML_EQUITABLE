import torch
import torch.nn as nn
import math

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
FL = torch.tensor([FREQUENCY_BOUNDS[band][0] for band in BAND_NAMES], dtype=torch.float32)
FU = torch.tensor([FREQUENCY_BOUNDS[band][1] for band in BAND_NAMES], dtype=torch.float32)

class RotaryFrequencyDemographicEncoding(nn.Module):
    def __init__(self, d_model, frequency_bounds: dict):
        super().__init__()
        self.d_model = d_model
        self.num_bands = len(frequency_bounds)

        assert d_model % 4 == 0, "d_model doit être divisible par 4"

        self.FL = torch.tensor([frequency_bounds[band][0] for band in frequency_bounds], dtype=torch.float32)
        self.FU = torch.tensor([frequency_bounds[band][1] for band in frequency_bounds], dtype=torch.float32)

    def forward(self, tokens, age, gender):
        """
        Args:
            tokens: [B, F, D]
            age:    [B] ou [B, 1]
            gender: [B] ou [B, 1]
        Returns:
            tokens + rotary-demographic encoding : [B, F, D]
        """
        B, F, D = tokens.shape
        device = tokens.device
        assert D == self.d_model and F == self.num_bands

        t = torch.arange(0, D // 4, dtype=torch.float32, device=device)
        theta = 4 * math.pi * t / D  # [D//4]

        fl = self.FL.to(device).unsqueeze(0).expand(B, F).reshape(-1)
        fu = self.FU.to(device).unsqueeze(0).expand(B, F).reshape(-1)
        age = age.view(B, 1).expand(B, F).reshape(-1)
        gender = gender.view(B, 1).expand(B, F).reshape(-1)

        x_flat = tokens.reshape(B * F, D)

        angle1 = fl.unsqueeze(1) * theta  # [B*F, D/4]
        angle2 = fu.unsqueeze(1) * theta

        real = age.unsqueeze(1) * torch.cos(angle1) + gender.unsqueeze(1)
        imag = age.unsqueeze(1) * torch.sin(angle2)

        rot_embed = torch.cat([real, imag], dim=-1).repeat(1, 2)  # [B*F, D]
        x_encoded = x_flat * rot_embed

        return x_encoded.reshape(B, F, D)