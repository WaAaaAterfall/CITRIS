"""
Block-assignment-collapse sweep for CITRIS-VAE on the real opto dataset
(Dataset4-opto). The baseline run collapsed psi(0) to empty (all 8 latents ->
opto block). This sweep varies the two levers that actually control the
opto<->psi0 partition in the flow-prior path:

  * lambda_reg   -- adds lambda_reg * (1 - P(psi0)) to the prior NLL
                    (transition_prior.sample_based_nll), directly rewarding
                    latents for moving into psi(0).
  * classifier_gumbel_temperature -- sharpness of the (prior + classifier)
                    Gumbel-Softmax assignment sampling; lower = more decisive.

beta_classifier is held at the baseline (2.0) on purpose: raising it strengthens
the opto-prediction gradient, which *reinforces* opto-block assignment and would
make the collapse worse, not better.

For each variant we train (same epochs/seed/data as baseline for comparability),
then load the best checkpoint and record, on the SAME val frames as the report:
  - hard block assignment (0=opto, 1=psi0) and #dims in each block,
  - soft psi0 probability per latent,
  - per-dim |Cohen's d| (opto ON vs OFF),
  - whether the dims that respond to opto (|d| high) and the dims that don't
    actually separate across the two blocks (the thing the baseline failed at).

Run from repo root with the citris env. Results -> keypoint_report_dataset4/sweep_results.json
"""

import os, sys, json, glob, subprocess
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.citris_vae import CITRISVAEKeypoints
from models.citris_vae.keypoint_module import effect_sizes
from experiments.keypoint_dataset import _load_raw, validate_data, split_trials, KeypointPairDataset
from experiments.evaluate_keypoints import encode_frames

DATA = 'data_generation/dataset4_pert_samples.npz'
OUT = 'keypoint_report_dataset4'
SWEEP_ROOT = 'checkpoints/keypoints_dataset4_sweep'
VARIANTS = ['lreg30', 'lreg50']
BASELINE_CKPT = ('checkpoints/keypoints_dataset4/CITRISVAE_dataset4/'
                 'version_0/checkpoints/best-epoch=299-val_loss=-141.58.ckpt')


def latest_best_ckpt(logger_name):
    cands = glob.glob(os.path.join(SWEEP_ROOT, logger_name, 'version_*',
                                   'checkpoints', 'best-*.ckpt'))
    return sorted(cands)[-1] if cands else None


def analyze(ckpt, device, X, opto, on, off):
    model = CITRISVAEKeypoints.load_from_checkpoint(ckpt).to(device).eval()
    Z = encode_frames(model, X, device=device)
    cohens_d, aucs = effect_sizes(Z, on, off)
    abs_d = np.abs(cohens_d)
    assignment = model.get_block_assignment()            # [L], 0=opto 1=psi0
    psi_soft = model.prior_t1.get_target_assignment(hard=False).detach().cpu().numpy()
    psi0_prob = psi_soft[:, -1]                          # P(psi0) per latent
    opto_dims = np.where(assignment == 0)[0]
    psi0_dims = np.where(assignment == 1)[0]
    # Does the partition track the effect sizes? (the baseline's failure mode)
    mean_d_opto = float(abs_d[opto_dims].mean()) if len(opto_dims) else float('nan')
    mean_d_psi0 = float(abs_d[psi0_dims].mean()) if len(psi0_dims) else float('nan')
    return {
        'checkpoint': ckpt,
        'assignment': assignment.tolist(),
        'n_opto_dims': int(len(opto_dims)), 'n_psi0_dims': int(len(psi0_dims)),
        'opto_dims': opto_dims.tolist(), 'psi0_dims': psi0_dims.tolist(),
        'psi0_prob': psi0_prob.round(3).tolist(),
        'abs_cohens_d': abs_d.round(3).tolist(),
        'auc': aucs.round(3).tolist(),
        'mean_abs_d_opto_block': mean_d_opto,
        'mean_abs_d_psi0_block': mean_d_psi0,
        'separation': (mean_d_opto - mean_d_psi0) if len(psi0_dims) and len(opto_dims) else float('nan'),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Fixed val set (identical split to the report / baseline).
    raw = validate_data(_load_raw(DATA))
    _, val_idx = split_trials(raw, val_fraction=0.2, seed=42)
    dset = KeypointPairDataset(raw, trials=val_idx)
    X, opto_t, _ = dset.all_frames()
    opto = opto_t.numpy(); on, off = opto > 0.5, opto <= 0.5

    # Merge into any prior sweep results so the final file holds every run.
    results = {}
    prior = os.path.join(OUT, 'sweep_results.json')
    if os.path.isfile(prior):
        results = json.load(open(prior))
    # Baseline first (for reference in the same table).
    if 'baseline' not in results and os.path.isfile(BASELINE_CKPT):
        print('== baseline =='); results['baseline'] = analyze(BASELINE_CKPT, device, X, opto, on, off)

    for tag in VARIANTS:
        cfg = f'experiments/configs/keypoints_dataset4_{tag}.json'
        logger_name = f'CITRISVAE_d4_{tag}'
        print(f'\n===== training variant {tag} ({cfg}) =====', flush=True)
        ret = subprocess.run([sys.executable, 'experiments/train_vae_keypoints.py',
                              '--config', cfg], cwd=os.getcwd())
        if ret.returncode != 0:
            print(f'[WARN] variant {tag} training exited {ret.returncode}; skipping analysis.')
            continue
        ckpt = latest_best_ckpt(logger_name)
        if ckpt is None:
            print(f'[WARN] no checkpoint found for {tag}.'); continue
        print(f'== analyze {tag} =='); results[tag] = analyze(ckpt, device, X, opto, on, off)
        json.dump(results, open(os.path.join(OUT, 'sweep_results.json'), 'w'), indent=2)

    json.dump(results, open(os.path.join(OUT, 'sweep_results.json'), 'w'), indent=2)
    # Compact console summary.
    print('\n================ SWEEP SUMMARY ================')
    hdr = f"{'run':14s} {'#opto':>5s} {'#psi0':>5s} {'meanD_opto':>10s} {'meanD_psi0':>10s} {'sep':>6s}"
    print(hdr)
    for k, r in results.items():
        print(f"{k:14s} {r['n_opto_dims']:5d} {r['n_psi0_dims']:5d} "
              f"{r['mean_abs_d_opto_block']:10.3f} {r['mean_abs_d_psi0_block']:10.3f} "
              f"{r['separation']:6.3f}")
    print('Wrote', os.path.join(OUT, 'sweep_results.json'))


if __name__ == '__main__':
    main()
