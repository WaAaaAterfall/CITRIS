"""
Ground-truth-free evaluation for CITRIS-VAE on keypoint data (Steps 5 & 7).

Given a trained checkpoint and the data dict, produces the Step 7 verification
report answering: does CITRIS isolate a latent subspace that responds to opto
while leaving a complementary subspace approximately opto-invariant?

Outputs (written to --out_dir):
  1. intervention_consistency.json / .png  -- effect size (Cohen's d, AUC) per
     latent dim on opto-ON vs OFF frames, split by learned block (Step 5.1).
  2. latent_observable_corr.png / .npz     -- correlation between each latent dim
     and derived kinematics (fingertip speed, fingertip<->port distance, etc.),
     plus port-identity decoding accuracy from the latents (Step 5.2).
  3. reconstruction_per_feature.json / .png -- per-keypoint reconstruction error (Step 5.3).
  4. latent_block_assignment.json          -- the learned latent->block assignment.
  5. report.md                             -- short human-readable summary.

Derived kinematics are computed from the RAW keypoints and are NEVER fed to the
model. Feature roles (which keypoint is the fingertip / water port) are matched
by substring against keypoints_name; override with --fingertip_key / --port_key.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.citris_vae import CITRISVAEKeypoints
from models.citris_vae.keypoint_module import effect_sizes
from experiments.keypoint_dataset import (_load_raw, validate_data,
                                          split_trials, KeypointPairDataset)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--split', type=str, default='val', choices=['val', 'train', 'all'])
    p.add_argument('--val_fraction', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=512)
    # Substrings used to locate semantic keypoints among keypoints_name.
    p.add_argument('--fingertip_key', type=str, default='fingertip')
    p.add_argument('--port_key', type=str, default='port')
    return p.parse_args()


@torch.no_grad()
def encode_frames(model, X, batch_size=512, device='cpu'):
    """ Encode raw frames -> latents (flow space if used), shape [N, L]. """
    model.eval()
    zs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        zs.append(model.encode(xb, random=False).cpu())
    return torch.cat(zs, 0).numpy()


def find_feature_indices(names, key):
    return [i for i, n in enumerate(names) if key.lower() in n.lower()]


def derived_kinematics(X, names, trial_ids, fingertip_key, port_key):
    """ Compute GT-free derived observables from raw keypoints. Returns a dict
    name -> array [N]. Speed/contact use per-trial frame ordering. """
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    trial_ids = trial_ids.numpy() if isinstance(trial_ids, torch.Tensor) else trial_ids
    obs = {}

    ft_idx = find_feature_indices(names, fingertip_key)
    port_idx = find_feature_indices(names, port_key)

    def xy(indices):
        xs = [i for i in indices if names[i].lower().endswith('_x') or names[i].lower().endswith('x')]
        ys = [i for i in indices if names[i].lower().endswith('_y') or names[i].lower().endswith('y')]
        # Fallback: assume interleaved [x, y] pairs if naming is unclear.
        if not xs or not ys:
            xs, ys = indices[0::2], indices[1::2]
        return xs, ys

    # Fingertip speed (mean over fingertip keypoints), distance to port.
    if ft_idx:
        ftx, fty = xy(ft_idx)
        fx = X[:, ftx].mean(1)
        fy = X[:, fty].mean(1)
        speed = np.zeros_like(fx)
        for t in np.unique(trial_ids):
            m = trial_ids == t
            pos = np.stack([fx[m], fy[m]], 1)
            d = np.zeros(pos.shape[0])
            if pos.shape[0] > 1:
                d[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            speed[m] = d
        obs['fingertip_speed'] = speed

        if port_idx:
            ptx, pty = xy(port_idx)
            px, py = X[:, ptx].mean(1), X[:, pty].mean(1)
            obs['fingertip_port_distance'] = np.sqrt((fx - px) ** 2 + (fy - py) ** 2)

    # Port identity per frame (constant within trial): label by unique port position.
    if port_idx:
        port_pos = X[:, port_idx]
        # Per-trial mean port position -> cluster into discrete identities.
        trial_port = {}
        for t in np.unique(trial_ids):
            trial_port[t] = port_pos[trial_ids == t].mean(0)
        unique_positions = []
        labels_per_trial = {}
        for t, pp in trial_port.items():
            matched = None
            for k, up in enumerate(unique_positions):
                if np.linalg.norm(pp - up) < 1e-3:
                    matched = k
                    break
            if matched is None:
                matched = len(unique_positions)
                unique_positions.append(pp)
            labels_per_trial[t] = matched
        port_identity = np.array([labels_per_trial[t] for t in trial_ids])
        obs['_port_identity'] = port_identity  # underscore => categorical, handled separately
    return obs


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw = validate_data(_load_raw(args.data_path))
    train_idx, val_idx = split_trials(raw, val_fraction=args.val_fraction, seed=args.seed)
    if args.split == 'val':
        trials = val_idx
    elif args.split == 'train':
        trials = train_idx
    else:
        trials = None
    dset = KeypointPairDataset(raw, trials=trials)
    names = dset.get_feature_names()

    model = CITRISVAEKeypoints.load_from_checkpoint(args.checkpoint).to(device)

    X, opto, trial_ids = dset.all_frames()
    Z = encode_frames(model, X, batch_size=args.batch_size, device=device)
    L = Z.shape[1]

    # ---- Step 5.1: intervention consistency ----
    assignment = model.get_block_assignment()
    opto_dims = np.where(assignment == 0)[0].tolist()
    psi0_dims = np.where(assignment == 1)[0].tolist()
    on = opto.numpy() > 0.5
    off = ~on
    if on.sum() >= 2 and off.sum() >= 2:
        cohens_d, aucs = effect_sizes(Z, on, off)
    else:
        cohens_d, aucs = np.zeros(L), np.full(L, 0.5)
        print('[!] Not enough opto-ON/OFF frames for a meaningful effect-size diagnostic.')
    abs_d = np.abs(cohens_d)

    ic = {
        'block_assignment': assignment.tolist(),
        'opto_dims': opto_dims,
        'psi0_dims': psi0_dims,
        'cohens_d': cohens_d.tolist(),
        'abs_cohens_d': abs_d.tolist(),
        'auc_opto_vs_off': aucs.tolist(),
        'mean_abs_d_opto_block': float(abs_d[opto_dims].mean()) if opto_dims else None,
        'mean_abs_d_psi0_block': float(abs_d[psi0_dims].mean()) if psi0_dims else None,
        'n_opto_on_frames': int(on.sum()),
        'n_opto_off_frames': int(off.sum()),
    }
    with open(os.path.join(args.out_dir, 'intervention_consistency.json'), 'w') as f:
        json.dump(ic, f, indent=2)

    # Bar plot of |Cohen's d| per latent, colored by block.
    fig, ax = plt.subplots(figsize=(max(6, L * 0.6), 4))
    colors = ['tab:red' if a == 0 else 'tab:blue' for a in assignment]
    ax.bar(range(L), abs_d, color=colors)
    ax.set_xlabel('latent dim'); ax.set_ylabel("|Cohen's d| (opto ON vs OFF)")
    ax.set_title('Intervention consistency (red=opto block, blue=psi0)')
    ax.set_xticks(range(L))
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'intervention_consistency.png'), dpi=150)
    plt.close(fig)

    # ---- Step 5.2: latent <-> observable correlation + port decoding ----
    obs = derived_kinematics(X, names, trial_ids, args.fingertip_key, args.port_key)
    cont_obs = {k: v for k, v in obs.items() if not k.startswith('_')}
    corr = np.zeros((L, len(cont_obs)))
    obs_names = list(cont_obs.keys())
    for j, name in enumerate(obs_names):
        o = cont_obs[name]
        for i in range(L):
            if np.std(Z[:, i]) > 1e-8 and np.std(o) > 1e-8:
                corr[i, j] = np.corrcoef(Z[:, i], o)[0, 1]
    np.savez(os.path.join(args.out_dir, 'latent_observable_corr.npz'),
             corr=corr, latent_dims=np.arange(L), obs_names=np.array(obs_names))

    if obs_names:
        fig, ax = plt.subplots(figsize=(max(4, len(obs_names) * 1.5), max(4, L * 0.5)))
        im = ax.imshow(corr, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(obs_names))); ax.set_xticklabels(obs_names, rotation=30, ha='right')
        ax.set_yticks(range(L)); ax.set_yticklabels([f'z{i}' for i in range(L)])
        ax.set_title('Latent <-> derived observable correlation')
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'latent_observable_corr.png'), dpi=150)
        plt.close(fig)

    # Port-identity decoding from latents (logistic regression accuracy via simple CV).
    port_decoding = None
    if '_port_identity' in obs:
        port_decoding = decode_port_identity(Z, obs['_port_identity'], trial_ids.numpy())
        with open(os.path.join(args.out_dir, 'port_decoding.json'), 'w') as f:
            json.dump(port_decoding, f, indent=2)

    # ---- Step 5.3: per-feature reconstruction error ----
    rec_err = per_feature_reconstruction(model, X, batch_size=args.batch_size, device=device)
    rec = {names[i]: float(rec_err[i]) for i in range(len(names))}
    with open(os.path.join(args.out_dir, 'reconstruction_per_feature.json'), 'w') as f:
        json.dump(rec, f, indent=2)
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.18)))
    ax.barh(range(len(names)), rec_err)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('MSE'); ax.set_title('Per-feature reconstruction error')
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'reconstruction_per_feature.png'), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.out_dir, 'latent_block_assignment.json'), 'w') as f:
        json.dump({'assignment_0opto_1psi0': assignment.tolist(),
                   'psi_soft': model.prior_t1.get_target_assignment(hard=False).detach().cpu().numpy().tolist()},
                  f, indent=2)

    write_report(args.out_dir, ic, port_decoding, obs_names, rec)
    print(f'Report written to {args.out_dir}/report.md')


@torch.no_grad()
def per_feature_reconstruction(model, X, batch_size=512, device='cpu'):
    model.eval()
    se = None; n = 0
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        x_rec, _, _, _ = model(xb)
        err = ((x_rec - xb) ** 2).sum(0).cpu().numpy()
        se = err if se is None else se + err
        n += xb.shape[0]
    return se / max(n, 1)


def decode_port_identity(Z, port_identity, trial_ids):
    """ Decode discrete port identity from latents using a per-trial train/test
    split (avoids frame-level leakage). Returns accuracy + chance level. """
    classes = np.unique(port_identity)
    if classes.shape[0] < 2:
        return {'accuracy': None, 'chance': None, 'note': 'only one port identity present'}
    rng = np.random.RandomState(0)
    unique_trials = np.unique(trial_ids)
    rng.shuffle(unique_trials)
    n_test = max(1, int(0.3 * len(unique_trials)))
    test_trials = set(unique_trials[:n_test].tolist())
    test_mask = np.array([t in test_trials for t in trial_ids])
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500, multi_class='auto')
        clf.fit(Z[~test_mask], port_identity[~test_mask])
        acc = float((clf.predict(Z[test_mask]) == port_identity[test_mask]).mean())
    except Exception:
        # Fallback: nearest-centroid in latent space (only classes seen in train).
        train_classes = np.unique(port_identity[~test_mask])
        cents = {c: Z[~test_mask][port_identity[~test_mask] == c].mean(0) for c in train_classes}
        pred = np.array([min(cents, key=lambda c: np.linalg.norm(z - cents[c])) for z in Z[test_mask]])
        acc = float((pred == port_identity[test_mask]).mean())
    # Majority-class chance.
    _, counts = np.unique(port_identity[~test_mask], return_counts=True)
    chance = float(counts.max() / counts.sum())
    return {'accuracy': acc, 'chance': chance, 'n_classes': int(classes.shape[0])}


def write_report(out_dir, ic, port_decoding, obs_names, rec):
    lines = ['# CITRIS-VAE keypoint verification report (Step 7)\n']
    lines.append('## 1. Intervention consistency (primary)\n')
    lines.append(f"- opto frames: {ic['n_opto_on_frames']} ON / {ic['n_opto_off_frames']} OFF")
    lines.append(f"- latent->block assignment (0=opto, 1=psi0): {ic['block_assignment']}")
    lines.append(f"- opto-block dims: {ic['opto_dims']} | psi0 dims: {ic['psi0_dims']}")
    lines.append(f"- mean |Cohen's d| opto block: {ic['mean_abs_d_opto_block']}")
    lines.append(f"- mean |Cohen's d| psi0 block: {ic['mean_abs_d_psi0_block']}")
    # Strongest single opto-responsive dim (the most reliable "did it work" read:
    # a real opto signal can exist even if CITRIS's discrete block assignment has
    # not yet cleanly grouped the high-effect dims into the opto block).
    abs_d = np.array(ic['abs_cohens_d'])
    auc = np.array(ic['auc_opto_vs_off'])
    top = int(np.argmax(abs_d))
    top_block = 'opto' if ic['block_assignment'][top] == 0 else 'psi0'
    lines.append(f"- strongest opto-responsive dim: z{top} "
                 f"(|d|={abs_d[top]:.3f}, AUC={auc[top]:.3f}, assigned to {top_block} block)")
    n_strong = int((abs_d >= 0.5).sum())
    lines.append(f"- #dims with |d|>=0.5: {n_strong}")

    md, pd_ = ic['mean_abs_d_opto_block'], ic['mean_abs_d_psi0_block']
    block_sep = (md is not None and pd_ is not None and md > pd_ + 0.1)
    signal_present = abs_d[top] >= 0.5 and abs(auc[top] - 0.5) >= 0.1
    if block_sep and signal_present:
        verdict = 'POSITIVE: opto subspace recovered AND cleanly assigned to the opto block'
    elif signal_present:
        verdict = ('PARTIAL: opto signal clearly recovered (strong per-dim effect) but '
                   "CITRIS's discrete latent->block assignment has not cleanly isolated it "
                   '(train longer / lower classifier_gumbel_temperature / raise classifier_lr)')
    else:
        verdict = 'NEGATIVE: no latent dim shows a clear opto response'
    if md is not None and pd_ is not None:
        lines.append(f"- block means: opto {md:.3f} vs psi0 {pd_:.3f}")
    lines.append(f"- **Verdict: {verdict}**")
    lines.append('\n## 2. Latent <-> observable\n')
    lines.append(f"- derived observables correlated: {obs_names}")
    if port_decoding is not None:
        lines.append(f"- port-identity decoding accuracy: {port_decoding.get('accuracy')} "
                     f"(chance {port_decoding.get('chance')})")
    lines.append('\n## 3. Reconstruction\n')
    worst = sorted(rec.items(), key=lambda kv: -kv[1])[:5]
    lines.append(f"- mean per-feature MSE: {np.mean(list(rec.values())):.5f}")
    lines.append(f"- worst-reconstructed features: {worst}")
    lines.append('\nSee the .png/.json files in this directory for full detail.\n')
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
