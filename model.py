# model.py
import torch.nn as nn
from swin_transformer_pytorch import SwinTransformer

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinTransformerModel, self).__init__()
        self.model = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 18, 2),  # Swin-T base configuration
            heads=(3, 6, 12, 24),
            channels=3,  # Number of input channels
            num_classes=num_classes,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True,
            # patch_norm=True,
        )
        
    def forward(self, x):
        # Forward pass through the model
        x = self.model(x)  # Corrected attribute name
        return x
