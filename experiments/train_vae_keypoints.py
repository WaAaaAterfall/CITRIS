"""
Training entry point for CITRIS-VAE on mouse-reach keypoint data with opto
interventions. Modeled on ``experiments/train_vae.py`` but stripped of all
image / triplet machinery.

Driven by a single JSON/YAML config (see ``experiments/configs/keypoints_example.json``).
Any config field can be overridden on the command line, e.g.:

    python experiments/train_vae_keypoints.py --config experiments/configs/keypoints_example.json \
        --num_latents 8 --max_epochs 200

Logging uses the same backend as the repo (TensorBoardLogger). Checkpoints (best
+ last by val_loss) and the learned latent->block assignment are saved under the
run's log directory.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.citris_vae import CITRISVAEKeypoints
from experiments.keypoint_dataset import (_load_raw, validate_data,
                                          split_trials, KeypointPairDataset)


DEFAULTS = {
    # data / split
    'data_path': None,
    'norm_params': None,        # provenance only; not used to (re)normalize
    'val_fraction': 0.2,
    'seed': 42,
    # model
    'num_latents': 8,
    'num_causal_vars': 1,
    'imperfect_interventions': True,
    'use_flow_prior': True,
    'autoregressive_prior': True,
    'c_hid': 32,               # hidden dim of prior / target-classifier networks
    'mlp_hidden': 128,         # hidden dim of encoder/decoder MLPs
    'mlp_num_layers': 3,
    'act_fn': 'silu',
    'lambda_reg': 0.01,
    'beta_t1': 1.0,
    'beta_classifier': 2.0,
    'kld_warmup': 0,
    # optimization
    'lr': 1e-3,
    'classifier_lr': 4e-3,
    'classifier_momentum': 0.0,
    'classifier_gumbel_temperature': 1.0,
    'warmup': 100,
    'max_epochs': 200,
    'batch_size': 256,
    'num_workers': 4,
    'check_val_every_n_epoch': 5,
    # logging
    'root_dir': 'checkpoints/keypoints',
    'logger_name': 'CITRISVAE_keypoints',
}


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Path to a JSON or YAML config file.')
    # Allow overriding any default from the CLI.
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            parser.add_argument(f'--{key}', type=lambda s: s.lower() in ('1', 'true', 'yes'),
                                default=None)
        elif val is None:
            parser.add_argument(f'--{key}', type=str, default=None)
        else:
            parser.add_argument(f'--{key}', type=type(val), default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config is not None:
        if args.config.endswith(('.yaml', '.yml')):
            import yaml
            with open(args.config) as f:
                file_cfg = yaml.safe_load(f)
        else:
            with open(args.config) as f:
                file_cfg = json.load(f)
        cfg.update({k: v for k, v in file_cfg.items() if v is not None})
    # CLI overrides
    for key in DEFAULTS:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    assert cfg['data_path'] is not None, 'You must provide data_path (in config or --data_path).'
    assert cfg['num_causal_vars'] == 1, 'Opto is the only intervention channel; num_causal_vars must be 1.'
    return cfg


def main():
    cfg = load_config()
    pl.seed_everything(cfg['seed'])

    # --- Data ---
    raw = validate_data(_load_raw(cfg['data_path']))
    train_idx, val_idx = split_trials(raw, val_fraction=cfg['val_fraction'], seed=cfg['seed'])
    train_set = KeypointPairDataset(raw, trials=train_idx)
    val_set = KeypointPairDataset(raw, trials=val_idx)
    D = train_set.get_input_dim()
    print(f'D={D}, |train pairs|={len(train_set)}, |val pairs|={len(val_set)}')

    train_loader = data.DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                                   drop_last=True, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False,
                                 drop_last=False, num_workers=cfg['num_workers'])

    # --- Model ---
    max_iters = cfg['max_epochs'] * max(1, len(train_loader))
    model = CITRISVAEKeypoints(
        D=D,
        mlp_hidden=cfg['mlp_hidden'],
        mlp_num_layers=cfg['mlp_num_layers'],
        num_latents=cfg['num_latents'],
        num_causal_vars=cfg['num_causal_vars'],
        c_hid=cfg['c_hid'],
        lr=cfg['lr'],
        classifier_lr=cfg['classifier_lr'],
        classifier_momentum=cfg['classifier_momentum'],
        classifier_gumbel_temperature=cfg['classifier_gumbel_temperature'],
        warmup=cfg['warmup'],
        max_iters=max_iters,
        kld_warmup=cfg['kld_warmup'],
        imperfect_interventions=cfg['imperfect_interventions'],
        use_flow_prior=cfg['use_flow_prior'],
        autoregressive_prior=cfg['autoregressive_prior'],
        lambda_reg=cfg['lambda_reg'],
        beta_t1=cfg['beta_t1'],
        beta_classifier=cfg['beta_classifier'],
        act_fn=cfg['act_fn'],
        var_names=train_set.target_names(),
        causal_encoder_checkpoint=None,
        cluster_logging=False,
    )

    # --- Logger + callbacks ---
    logger = pl.loggers.TensorBoardLogger(cfg['root_dir'], name=cfg['logger_name'])
    ckpt_cb = ModelCheckpoint(save_weights_only=False, mode='min', monitor='val_loss',
                              save_last=True, filename='best-{epoch}-{val_loss:.2f}')
    callbacks = CITRISVAEKeypoints.get_callbacks() + [ckpt_cb]

    trainer = pl.Trainer(
        default_root_dir=cfg['root_dir'],
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg['max_epochs'],
        check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
        gpus=(1 if torch.cuda.is_available() else 0),
        gradient_clip_val=1.0,
    )
    trainer.logger._default_hp_metric = None

    # Save the resolved config + feature names for provenance.
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    with open(os.path.join(trainer.logger.log_dir, 'run_config.json'), 'w') as f:
        json.dump({**cfg, 'D': D, 'feature_names': train_set.get_feature_names(),
                   'train_trials': train_idx, 'val_trials': val_idx}, f, indent=2)

    trainer.fit(model, train_loader, val_loader)

    # --- Save the learned latent->block assignment (Step 6/7) ---
    best_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if best_path and os.path.isfile(best_path):
        model = CITRISVAEKeypoints.load_from_checkpoint(best_path)
    assignment = model.get_block_assignment()
    psi_soft = model.prior_t1.get_target_assignment(hard=False).detach().cpu().numpy()
    np.savez(os.path.join(trainer.logger.log_dir, 'latent_block_assignment.npz'),
             assignment=assignment, psi_soft=psi_soft,
             columns=np.array(['opto', 'psi0']))
    print('Latent->block assignment (0=opto, 1=psi0):', assignment.tolist())
    print('Best checkpoint:', best_path)
    print('Log dir:', trainer.logger.log_dir)


if __name__ == '__main__':
    main()
