from torch import nn
import torch

class ResnetBlock(nn.Module):
    """ResNet block for CNN encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class DepthPositionPredictor_resnet(nn.Module):
    """CNN encoder + MLP predictor: (N, 1, H, W) depth → (N, 3) object pos."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (N, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (N, 32, H/4, W/4)
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
            )  # (N, 32, H/8, W/8)
        self.res_blocks = nn.Sequential( *[ResnetBlock(64) for _ in range(4)])  # (N, 32, H/8, W/8)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (N, 32, 1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(73, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3)  # (N, 3)
        )


    def forward(self, depth, joint_pos):
        x = self.conv_stack(depth)
        x = self.res_blocks(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (N, 32)
        x = torch.cat([x, joint_pos], dim=-1)  # (N, 32+7=39)
        return self.mlp(x)

class DepthPositionPredictor(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32 + 9, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )

    def forward(self, depth, joint_pos):
        x = self.conv_stack(depth).squeeze(-1).squeeze(-1)
        x = torch.cat([x, joint_pos], dim=-1)
        return self.mlp(x)

class LatentProbe:
    """
    Linear probe: measures how much object position info
    is contained in the latent without affecting encoder training.
    A low probe error means the latent is geometry-aware.
    """
    def __init__(self, latent_dim: int = 64):
        self._probe = None
        self._optimizer = None

    def _build(self, device):
        probe = nn.Linear(64, 3).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-4)
        return probe, optimizer

    def update(self, latent: torch.Tensor, gt_pos: torch.Tensor):
        if self._probe is None:
            self._probe, self._optimizer = self._build(latent.device)

        # Detach latent — probe trains, encoder does NOT
        pred = self._probe(latent.detach())
        loss = nn.functional.mse_loss(pred, gt_pos.detach())
        loss = torch.tensor(loss, requires_grad=True)  # ensure loss has grad

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        with torch.no_grad():
            err = (pred - gt_pos.detach()).norm(dim=-1).mean().item()
        return loss.item(), err


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
 
    def forward(self, pred, target):
        e = pred - target
        return torch.mean(torch.sqrt(e ** 2 + self.eps ** 2))
 
 

