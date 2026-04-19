import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.modules import HiddenState


class CNNModelWithAux(CNNModel):
    """CNNModel with auxiliary head predicting object position from CNN latent."""

    def __init__(self, *args, aux_dim: int = 3, aux_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = aux_weight
        self.aux_head = nn.Sequential(
            nn.Linear(self.cnn_latent_dim, 64),
            nn.ELU(),
            nn.Linear(64, aux_dim),
        )
        self._aux_pred: torch.Tensor | None = None

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        # 1D latent from MLPModel parent
        latent_1d = super(CNNModel, self).get_latent(obs)
        # CNN latent
        latent_cnn = torch.cat(
            [self.cnns[g](obs[g]) for g in self.obs_groups_2d], dim=-1
        )
        # Store aux prediction for loss computation
        self._aux_pred = self.aux_head(latent_cnn)
        return torch.cat([latent_1d, latent_cnn], dim=-1)