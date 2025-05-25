# TODO: XAI-guided loss (Eq. 14)
import torch
import torch.nn as nn
import torch.nn.functional as F
def xaiguided_loss(y_hat_coarse, y_hat_expl, y_true, alpha=0.7):
    return (1 - alpha) * nn.functional.cross_entropy(y_hat_coarse, y_true) + \
            alpha * nn.functional.cross_entropy(y_hat_expl, y_true)