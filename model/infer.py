"""
model/infer.py
==============
STAGE 3 - ENSEMBLE INFERENCE & TEST-TIME AUGMENTATION (TTA)
Executes deep learning inference across an ensemble of models with TTA.
Generates:
  1. Mean flood probability raster (0.0 to 1.0)
  2. Variance-based uncertainty raster
  3. Permanent water probability raster
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from model.train import FloodUNet, WEIGHTS_DIR


class FloodEnsemblePredictor:
    """Ensemble inference runner with Test-Time Augmentation."""

    def __init__(self, models: Optional[List[FloodUNet]] = None, device: str = "cpu"):
        self.device = device
        if models:
            self.models = [m.to(device).eval() for m in models]
        else:
            # Load default trained checkpoints or instantiate baseline
            weights_files = list(WEIGHTS_DIR.glob("*.pt"))
            self.models = []
            if weights_files:
                for wf in weights_files[:3]:
                    m = FloodUNet(in_channels=9, num_classes=3)
                    m.load_state_dict(torch.load(wf, map_location=device, weights_only=True))
                    m.to(device).eval()
                    self.models.append(m)
            if not self.models:
                # Initialize an untrained or synthetic initialized architecture
                m = FloodUNet(in_channels=9, num_classes=3).to(device).eval()
                self.models.append(m)

    def _predict_single_with_tta(self, model: FloodUNet, x: torch.Tensor) -> torch.Tensor:
        """Run model with test-time augmentation (original, horizontal flip, vertical flip)."""
        # x is (1, 9, H, W)
        with torch.no_grad():
            # 1. Normal
            out1 = F.softmax(model(x), dim=1)

            # 2. Horizontal flip
            x_h = torch.flip(x, dims=[3])
            out2 = torch.flip(F.softmax(model(x_h), dim=1), dims=[3])

            # 3. Vertical flip
            x_v = torch.flip(x, dims=[2])
            out3 = torch.flip(F.softmax(model(x_v), dim=1), dims=[2])

            return (out1 + out2 + out3) / 3.0

    def predict(
        self,
        features_9ch: np.ndarray,
        use_tta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Input:
          features_9ch: np.ndarray of shape (9, H, W)
        Returns:
          flood_prob: (H, W) float32 probability of flood
          uncertainty: (H, W) float32 variance across ensemble/TTA passes
          perm_water_prob: (H, W) float32 probability of permanent water
        """
        x = torch.from_numpy(features_9ch).unsqueeze(0).float().to(self.device)

        ensemble_flood_probs = []
        ensemble_water_probs = []

        for model in self.models:
            if use_tta:
                probs = self._predict_single_with_tta(model, x)
            else:
                with torch.no_grad():
                    probs = F.softmax(model(x), dim=1)

            # Class 1 = permanent water, Class 2 = flood
            perm_water = probs[0, 1].cpu().numpy()
            flood = probs[0, 2].cpu().numpy()

            ensemble_flood_probs.append(flood)
            ensemble_water_probs.append(perm_water)

        stacked_flood = np.stack(ensemble_flood_probs, axis=0)
        mean_flood_prob = np.mean(stacked_flood, axis=0)

        # Variance across models represents epistemic model uncertainty
        if len(self.models) > 1:
            uncertainty = np.var(stacked_flood, axis=0)
        else:
            # When single model, compute entropy-based uncertainty: -p*log(p)
            p = np.clip(mean_flood_prob, 1e-6, 1.0 - 1e-6)
            uncertainty = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

        stacked_water = np.stack(ensemble_water_probs, axis=0)
        mean_water_prob = np.mean(stacked_water, axis=0)

        return (
            mean_flood_prob.astype(np.float32),
            uncertainty.astype(np.float32),
            mean_water_prob.astype(np.float32),
        )
