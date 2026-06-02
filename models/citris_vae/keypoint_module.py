"""
CITRIS-VAE adapted to low-dimensional keypoint observations with a single
(optogenetic) intervention channel.

Path taken (see Step 0 of the instruction doc): we SUBCLASS ``CITRISVAE`` and
reuse ``TransitionPrior``, ``TargetClassifier``, the flow prior, the ELBO in
``_get_loss``, and the LR scheduling VERBATIM. Only three things change:
  1. The conv encoder/decoder are replaced by ``MLPEncoder``/``MLPDecoder``
     (Step 3). We call ``super().__init__(no_encoder_decoder=True, ...)`` so the
     image stack is never built, then install the MLPs.
  2. ``_get_rec_loss`` uses a Gaussian observation likelihood over the D keypoint
     features (the base class' single overridable reconstruction hook).
  3. ``validation_step``/``test_step``/``validation_epoch_end`` drop the triplet +
     CausalEncoder path (no ground-truth factors) and instead log the GT-free
     intervention-consistency diagnostic (Step 5.1). Heavier diagnostics
     (latent<->observable heatmap, port decoding, per-feature reconstruction)
     live in ``experiments/evaluate_keypoints.py``.

The conv path of ``CITRISVAE`` is left completely untouched.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor

from models.citris_vae.lightning_module import CITRISVAE
from models.shared import MLPEncoder, MLPDecoder, get_act_fn, gaussian_log_prob


class CITRISVAEKeypoints(CITRISVAE):
    """ CITRIS-VAE for keypoint vectors (D features, 1 opto intervention channel). """

    def __init__(self, D, mlp_hidden=128, mlp_num_layers=3,
                 rec_init_log_std=-1.0, **kwargs):
        """
        Parameters
        ----------
        D : int
            Dimensionality of the keypoint observation vector (76 for the mouse data).
        mlp_hidden : int
            Hidden width of the encoder/decoder MLPs.
        mlp_num_layers : int
            Number of linear layers in each MLP (>=1).
        rec_init_log_std : float
            Initial value of the (learned) Gaussian observation log-std used in the
            reconstruction likelihood.
        kwargs : passed through to CITRISVAE (num_latents, num_causal_vars, lr,
            imperfect_interventions, lambda_reg, beta_t1, beta_classifier,
            classifier_lr, autoregressive_prior, use_flow_prior, var_names, ...).
        """
        # Build everything except the conv encoder/decoder.
        kwargs['no_encoder_decoder'] = True
        # img_width / c_in are inert for keypoints but must be valid for super().__init__.
        kwargs.setdefault('img_width', 64)
        kwargs.setdefault('c_in', 3)
        kwargs.setdefault('causal_encoder_checkpoint', None)
        super().__init__(**kwargs)
        # Capture our extra hyperparameters so load_from_checkpoint reconstructs them.
        self.save_hyperparameters('D', 'mlp_hidden', 'mlp_num_layers', 'rec_init_log_std')

        act_fn_func = get_act_fn(self.hparams.act_fn)
        self.encoder = MLPEncoder(D_in=D,
                                  num_latents=self.hparams.num_latents,
                                  hidden=mlp_hidden,
                                  n_layers=mlp_num_layers,
                                  act_fn=act_fn_func,
                                  variational=True)
        self.decoder = MLPDecoder(num_latents=self.hparams.num_latents,
                                  D_out=D,
                                  hidden=mlp_hidden,
                                  n_layers=mlp_num_layers,
                                  act_fn=act_fn_func)
        # Learned per-feature observation log-std for the Gaussian likelihood.
        self.rec_log_std = nn.Parameter(torch.full((D,), float(rec_init_log_std)))

        # Buffers for GT-free validation diagnostics.
        self._val_z = []
        self._val_opto = []

    # --- Reconstruction: Gaussian NLL over the D features (Step 3) ---
    def _get_rec_loss(self, x_rec, x_true):
        # x_rec, x_true: [batch_size, time_steps-1, D]
        log_std = self.rec_log_std.clamp(min=-7.0, max=3.0)
        nll = -gaussian_log_prob(x_rec, log_std, x_true)
        return nll.sum(dim=-1)  # [batch_size, time_steps-1]

    # --- GT-free validation/test (Step 5.1) ---
    def _accumulate_diag(self, batch):
        x, target = batch                     # x: [B, 2, D], target: [B, 1, 1]
        with torch.no_grad():
            z = self.encode(x[:, 1], random=False)   # latent of frame t+1 (flow space if enabled)
        self._val_z.append(z.detach().cpu())
        self._val_opto.append(target[:, 0, 0].detach().cpu())

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)
        self._accumulate_diag(batch)

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='test')
        self.log('test_loss', loss)

    def validation_epoch_end(self, *args, **kwargs):
        if len(self._val_z) == 0:
            return
        z = torch.cat(self._val_z, dim=0).numpy()        # [N, L]
        opto = torch.cat(self._val_opto, dim=0).numpy()  # [N]
        self._val_z, self._val_opto = [], []

        on, off = opto > 0.5, opto <= 0.5
        if on.sum() < 2 or off.sum() < 2:
            print('[val] Not enough opto-ON/OFF frames for effect-size diagnostic.')
            return

        cohens_d, aucs = effect_sizes(z, on, off)
        assignment = self.get_block_assignment()          # [L] in {0=opto, 1=psi0}
        opto_dims = np.where(assignment == 0)[0]
        psi0_dims = np.where(assignment == 1)[0]

        # Core "did it work" signal: opto-block dims should respond, psi0 dims should not.
        abs_d = np.abs(cohens_d)
        if len(opto_dims) > 0:
            self.log('val_opto_block_abs_d', float(abs_d[opto_dims].mean()))
        if len(psi0_dims) > 0:
            self.log('val_psi0_block_abs_d', float(abs_d[psi0_dims].mean()))
        self.log('val_max_abs_d', float(abs_d.max()))
        for i in range(z.shape[1]):
            self.log(f'val_latent{i}_abs_cohens_d', float(abs_d[i]))

        # Persist per-dim diagnostics + assignment for the Step 7 report.
        if self.logger is not None and getattr(self.logger, 'log_dir', None) is not None:
            out = {
                'epoch': int(self.current_epoch),
                'block_assignment': assignment.tolist(),     # 0=opto, 1=psi0
                'opto_dims': opto_dims.tolist(),
                'psi0_dims': psi0_dims.tolist(),
                'cohens_d': cohens_d.tolist(),
                'abs_cohens_d': abs_d.tolist(),
                'auc_opto_vs_off': aucs.tolist(),
            }
            with open(os.path.join(self.logger.log_dir, 'val_intervention_consistency.json'), 'w') as f:
                json.dump(out, f, indent=2)

    def get_block_assignment(self):
        """ Hard latent->block assignment from the transition prior's psi.
        Columns are [opto-block(s)..., psi0]; we map the argmax to an index where
        0 == opto block and (num_causal_vars) == psi0. Returns int array [num_latents]. """
        with torch.no_grad():
            ta = self.prior_t1.get_target_assignment(hard=True)  # [L, num_blocks+1]
            assignment = ta.argmax(dim=-1).cpu().numpy()
        return assignment

    # --- Callbacks: no image/correlation callbacks for keypoints ---
    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, **kwargs):
        return [LearningRateMonitor('step')]


def effect_sizes(z, on, off):
    """ Per-latent effect of opto-ON vs opto-OFF.
    Returns (cohens_d[L], auc[L]). Cohen's d is the standardized mean difference
    (ON - OFF); AUC is P(z_on > z_off) (0.5 = no effect, distance from 0.5 = effect). """
    z_on, z_off = z[on], z[off]
    mu_on, mu_off = z_on.mean(0), z_off.mean(0)
    var_on, var_off = z_on.var(0, ddof=1), z_off.var(0, ddof=1)
    n_on, n_off = z_on.shape[0], z_off.shape[0]
    pooled_std = np.sqrt(((n_on - 1) * var_on + (n_off - 1) * var_off) /
                         max(n_on + n_off - 2, 1))
    pooled_std = np.where(pooled_std < 1e-8, 1e-8, pooled_std)
    cohens_d = (mu_on - mu_off) / pooled_std

    # AUC per dim via the Mann-Whitney U rank statistic.
    L = z.shape[1]
    aucs = np.zeros(L)
    for i in range(L):
        order = np.argsort(z[:, i], kind='mergesort')
        ranks = np.empty(z.shape[0])
        ranks[order] = _avg_ranks(z[order, i])
        r_on = ranks[on].sum()
        aucs[i] = (r_on - n_on * (n_on + 1) / 2.0) / (n_on * n_off)
    return cohens_d, aucs


def _avg_ranks(sorted_vals):
    """ Average ranks (1-based) handling ties, for an already-sorted array. """
    n = sorted_vals.shape[0]
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks
