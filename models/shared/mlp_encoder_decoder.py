"""
MLP encoder / decoder for low-dimensional (keypoint) observations.

These mirror the interface of the convolutional ``Encoder``/``Decoder`` in
``models/shared/encoder_decoder.py`` so that ``CITRISVAE`` can use them
unchanged:
    - ``MLPEncoder(x)`` with ``variational=True`` returns ``(mean, log_std)``
      over ``num_latents`` (same contract as ``Encoder(variational=True)``).
    - ``MLPDecoder(z)`` returns the reconstruction mean of shape ``[*, D_out]``.

We use a Gaussian observation likelihood for the reconstruction term. The
decoder predicts the mean; the (learned, per-feature) log-variance lives on the
LightningModule (``CITRISVAEKeypoints.rec_log_std``) so that
``gaussian_log_prob`` (already used throughout the repo) can be reused for the
reconstruction term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(d_in, d_out, hidden, n_layers, act_fn, use_layer_norm=True):
    layers = []
    dim = d_in
    for _ in range(max(0, n_layers - 1)):
        layers.append(nn.Linear(dim, hidden))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(act_fn())
        dim = hidden
    layers.append(nn.Linear(dim, d_out))
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    """ Variational MLP encoder for keypoint vectors. """

    def __init__(self, D_in, num_latents, hidden=128, n_layers=3,
                 act_fn=lambda: nn.SiLU(), variational=True):
        """
        Parameters
        ----------
        D_in : int
               Dimensionality of the input keypoint vector (D=76 for the mouse data).
        num_latents : int
                      Number of latent variables.
        hidden : int
                 Hidden dimensionality of the MLP.
        n_layers : int
                   Total number of linear layers (>=1).
        act_fn : callable returning nn.Module
                 Activation factory.
        variational : bool
                      If True, outputs (mean, log_std); otherwise a single vector.
        """
        super().__init__()
        self.variational = variational
        out_dim = 2 * num_latents if variational else num_latents
        self.net = _make_mlp(D_in, out_dim, hidden, n_layers, act_fn)
        # Stabilizing the predicted log-std, mirroring the conv Encoder.
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))

    def forward(self, x):
        feats = self.net(x)
        if self.variational:
            mean, log_std = feats.chunk(2, dim=-1)
            s = F.softplus(self.scale_factor)
            log_std = torch.tanh(log_std / s) * s
            return mean, log_std
        else:
            return feats


class MLPDecoder(nn.Module):
    """ MLP decoder reconstructing keypoint vectors from latents. """

    def __init__(self, num_latents, D_out, hidden=128, n_layers=3,
                 act_fn=lambda: nn.SiLU()):
        super().__init__()
        self.net = _make_mlp(num_latents, D_out, hidden, n_layers, act_fn)

    def forward(self, z):
        return self.net(z)
