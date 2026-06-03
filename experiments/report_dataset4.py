"""
Enhanced, GT-free analysis + figures for CITRIS-VAE on the REAL opto dataset
(Dataset4-opto). Augments experiments/evaluate_keypoints.py with:

  * port-POSITION regression (R^2, by-trial CV) instead of the discrete-class
    decoder -- the real water port varies near-continuously in x (it is not 4-5
    discrete locations), so classification into per-trial clusters is degenerate.
  * a latent ON-vs-OFF distribution figure for every latent dim.
  * an extended latent<->observable correlation heatmap that includes raw port_x.
  * a scatter of the most port-correlated latent vs. port_x.

All derived observables come from RAW keypoints and are never fed to the model.
Run from the repo root with the citris env.
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.citris_vae import CITRISVAEKeypoints
from models.citris_vae.keypoint_module import effect_sizes
from experiments.keypoint_dataset import _load_raw, validate_data, split_trials, KeypointPairDataset
from experiments.evaluate_keypoints import encode_frames, derived_kinematics

OUT = 'keypoint_report_dataset4'
DATA = 'data_generation/dataset4_pert_samples.npz'


def port_xy(X, names):
    """ Front-camera water-port (x, y) per frame, from RAW keypoints. """
    ix, iy = names.index('waterport_x'), names.index('waterport_y')
    return X[:, ix], X[:, iy]


def _ridge_fit_predict(Ztr, ytr, Zte, alpha=1.0):
    """ Closed-form ridge regression (numpy only) with an intercept. """
    mu = Ztr.mean(0); ymu = ytr.mean()
    A = Ztr - mu; b = ytr - ymu
    d = A.shape[1]
    w = np.linalg.solve(A.T @ A + alpha * np.eye(d), A.T @ b)
    return (Zte - mu) @ w + ymu


def by_trial_cv_r2(Z, y, trial_ids, k=5, seed=0, alpha=1.0):
    """ Ridge R^2 of a continuous target from latents, k-fold CV over whole TRIALS
    (no frame leakage). R^2 computed against the TRAIN-fold mean baseline. """
    rng = np.random.RandomState(seed)
    trials = np.unique(trial_ids); rng.shuffle(trials)
    folds = np.array_split(trials, min(k, len(trials)))
    r2s = []
    for f in folds:
        te = np.isin(trial_ids, f); tr = ~te
        if tr.sum() < 2 or te.sum() < 1:
            continue
        yhat = _ridge_fit_predict(Z[tr], y[tr], Z[te], alpha)
        ss_res = np.sum((y[te] - yhat) ** 2)
        ss_tot = np.sum((y[te] - y[tr].mean()) ** 2)
        r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return (float(np.mean(r2s)) if r2s else float('nan')), r2s


def main():
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = sorted([os.path.join(r, f)
                   for r, _, fs in os.walk('checkpoints/keypoints_dataset4')
                   for f in fs if f.startswith('best-') and f.endswith('.ckpt')])[-1]
    print('checkpoint:', ckpt)

    raw = validate_data(_load_raw(DATA))
    train_idx, val_idx = split_trials(raw, val_fraction=0.2, seed=42)
    dset = KeypointPairDataset(raw, trials=val_idx)
    names = dset.get_feature_names()

    model = CITRISVAEKeypoints.load_from_checkpoint(ckpt).to(device)
    X, opto, trial_ids = dset.all_frames()
    Z = encode_frames(model, X, device=device)               # [N, L]
    L = Z.shape[1]
    Xn = X.numpy(); tids = trial_ids.numpy(); opto = opto.numpy()
    on, off = opto > 0.5, opto <= 0.5

    cohens_d, aucs = effect_sizes(Z, on, off)
    abs_d = np.abs(cohens_d)
    order = np.argsort(-abs_d)

    # ---- (A) Latent ON vs OFF distributions, all dims, sorted by |d| ----
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, i in zip(axes.ravel(), order):
        ax.boxplot([Z[off, i], Z[on, i]], labels=['OFF', 'ON'], showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor='lightsteelblue'))
        ax.set_title(f"z{i}  |d|={abs_d[i]:.2f}  AUC={aucs[i]:.2f}",
                     fontsize=10, fontweight='bold' if abs_d[i] >= 0.5 else 'normal')
        ax.axhline(0, color='gray', lw=0.5)
    fig.suptitle('Latent value distributions: opto OFF vs ON (val frames), sorted by effect size',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, 'latent_on_vs_off.png'), dpi=150); plt.close(fig)

    # ---- (B) Port-position regression (continuous) + derived obs corr ----
    px, py = port_xy(Xn, names)
    r2_x, _ = by_trial_cv_r2(Z, px, tids)
    r2_y, _ = by_trial_cv_r2(Z, py, tids)
    # which single latent best linearly tracks port_x
    corr_px = np.array([np.corrcoef(Z[:, i], px)[0, 1] for i in range(L)])
    best_port_dim = int(np.argmax(np.abs(corr_px)))

    obs = derived_kinematics(X, names, trial_ids, 'fingertip', 'port')
    cont = {k: v for k, v in obs.items() if not k.startswith('_')}
    cont['port_x'] = px
    obs_names = list(cont.keys())
    corr = np.zeros((L, len(obs_names)))
    for j, nm in enumerate(obs_names):
        o = cont[nm]
        for i in range(L):
            if np.std(Z[:, i]) > 1e-8 and np.std(o) > 1e-8:
                corr[i, j] = np.corrcoef(Z[:, i], o)[0, 1]
    fig, ax = plt.subplots(figsize=(max(5, len(obs_names) * 1.6), max(4, L * 0.55)))
    im = ax.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(L):
        for j in range(len(obs_names)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(obs_names))); ax.set_xticklabels(obs_names, rotation=20, ha='right')
    ax.set_yticks(range(L)); ax.set_yticklabels([f'z{i}' for i in range(L)])
    ax.set_title('Latent <-> derived observable correlation (val)')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'latent_observable_corr.png'), dpi=150); plt.close(fig)

    # ---- (C) Scatter: best port-tracking latent vs port_x ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(px, Z[:, best_port_dim], s=6, alpha=0.4)
    ax.set_xlabel('raw water-port x position'); ax.set_ylabel(f'latent z{best_port_dim}')
    ax.set_title(f'Goal-location validation: z{best_port_dim} vs port_x '
                 f'(r={corr_px[best_port_dim]:.2f})')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'latent_vs_port_x.png'), dpi=150); plt.close(fig)

    summary = {
        'n_val_frames': int(X.shape[0]),
        'n_opto_on': int(on.sum()), 'n_opto_off': int(off.sum()),
        'abs_cohens_d': abs_d.tolist(), 'auc': aucs.tolist(),
        'n_dims_d_ge_0p5': int((abs_d >= 0.5).sum()),
        'top_opto_dim': int(order[0]), 'top_opto_abs_d': float(abs_d[order[0]]),
        'port_x_regression_r2': r2_x, 'port_y_regression_r2': r2_y,
        'best_port_dim': best_port_dim, 'best_port_dim_corr': float(corr_px[best_port_dim]),
    }
    with open(os.path.join(OUT, 'enhanced_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
