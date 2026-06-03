"""
Dataset for the mouse reach-to-consume keypoint data with optogenetic
interventions, adapting the ``sub_indices`` / adjacent-pair pattern of
``Causal3DDataset`` (``experiments/datasets.py``) to low-dimensional vectors.

Data contract (a single pickled dict or ``.npz``), with EXACTLY these keys:

    {
      "keypoints":              list[np.ndarray],  # len N_trials; each [T_i, D] float32
      "keypoints_name":         list[str],         # len D; feature names
      "perturbation_indicator": list[np.ndarray],  # len N_trials; each [T_i] in {0,1}
    }

Optional:
    "trial_session": list[int]   # session id per trial, used for held-out-session splits

Design notes (see CITRIS_keypoint_adaptation_instructions.md):
  * num_causal_vars == 1 (opto only); target-port location is NOT an intervention.
  * Transition unit is a single adjacent pair (x_t, x_{t+1}); never spans trials.
  * Keypoints are assumed already normalized; we do NOT re-normalize.

CONFLICT WITH THE INSTRUCTION DOC (surfaced, not silently resolved):
  Step 2 of the doc says ``target`` should have shape ``[1]``. However,
  ``CITRISVAE._get_loss`` does ``target.flatten(0, 1)`` and the transition prior /
  target classifier expect ``target`` of shape ``[B, seq_len-1, num_blocks]``
  (this is exactly what ``Causal3DDataset`` returns: per-item shape
  ``[seq_len-1, num_vars] == [1, 1]``). Returning ``[1]`` would collapse the
  num_blocks axis and break the prior. We therefore return ``target`` of shape
  ``[1, 1] == [seq_len-1, num_causal_vars]`` to match the code.
"""

import numpy as np
import torch
import torch.utils.data as data


def _load_raw(path):
    """ Load the data dict from a .npz or a pickle (.pkl/.pickle/.npy). """
    if path.endswith('.npz'):
        arr = np.load(path, allow_pickle=True)
        d = {k: arr[k] for k in arr.files}
        # np.load wraps object arrays; unwrap lists stored as 0-d object arrays
        for k in list(d.keys()):
            if isinstance(d[k], np.ndarray) and d[k].dtype == object and d[k].ndim == 0:
                d[k] = d[k].item()
        return d
    else:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


def _expand_interleaved_names(names):
    """ Real DLC data stores ``D = 2 * len(keypoints_name)`` columns, interleaved
    as ``[kp0_x, kp0_y, kp1_x, kp1_y, ...]`` while ``keypoints_name`` holds ONE
    name per keypoint (see data_generation/evalute_keyoints_data.ipynb). Expand to
    one name per column, with ``_x`` / ``_y`` suffixes, disambiguating names that
    repeat across camera views (e.g. "nose"/"waterport" appear in both front and
    side cameras) by an occurrence index. Keypoint ``k`` -> columns ``[2k, 2k+1]``. """
    seen = {}
    expanded = []
    for base in names:
        seen[base] = seen.get(base, 0) + 1
        tag = base if seen[base] == 1 else f'{base}.{seen[base]}'
        expanded.append(f'{tag}_x')
        expanded.append(f'{tag}_y')
    return expanded


def validate_data(d):
    """ Assert the data dict obeys the contract. Fails loudly with the offending
    trial index. Returns the (possibly list-normalized) dict. """
    required = ['keypoints', 'keypoints_name', 'perturbation_indicator']
    for k in required:
        assert k in d, f'Missing required key "{k}". Found keys: {list(d.keys())}'

    keypoints = list(d['keypoints'])
    names = list(d['keypoints_name'])
    pert = list(d['perturbation_indicator'])

    # CONFLICT WITH THE INSTRUCTION DOC (surfaced, not silently resolved):
    # the doc assumes keypoints_name already has one entry PER COLUMN (D=76 names).
    # The real DLC export instead provides one name PER KEYPOINT (38) with 76
    # interleaved x/y columns. Detect this exactly (every trial has
    # shape[1] == 2 * len(names)) and expand the names so the per-column contract
    # and the x/y-aware evaluation both hold. Reported here, not silently ignored.
    col_dims = {np.asarray(k).shape[1] for k in keypoints}
    if len(col_dims) == 1 and next(iter(col_dims)) == 2 * len(names):
        print(f'[validate_data] Detected interleaved x/y layout: '
              f'{len(names)} keypoint names -> {2 * len(names)} columns. '
              f'Expanding names with _x/_y suffixes (occurrence-disambiguated).')
        names = _expand_interleaved_names(names)

    n_trials = len(keypoints)
    assert len(pert) == n_trials, \
        f'Length mismatch: keypoints={n_trials}, perturbation_indicator={len(pert)}'
    if 'trial_session' in d and d['trial_session'] is not None:
        sess = list(d['trial_session'])
        assert len(sess) == n_trials, \
            f'Length mismatch: keypoints={n_trials}, trial_session={len(sess)}'

    D = len(names)
    for i in range(n_trials):
        kp = np.asarray(keypoints[i])
        pi = np.asarray(pert[i])
        assert kp.ndim == 2, f'Trial {i}: keypoints must be 2D [T, D], got shape {kp.shape}'
        assert kp.shape[1] == D, \
            f'Trial {i}: keypoints D={kp.shape[1]} != len(keypoints_name)={D}'
        assert pi.shape[0] == kp.shape[0], \
            f'Trial {i}: perturbation length {pi.shape[0]} != T={kp.shape[0]}'
        uniq = np.unique(pi)
        assert np.all(np.isin(uniq, [0, 1])), \
            f'Trial {i}: perturbation_indicator has values outside {{0,1}}: {uniq}'
        assert np.isfinite(kp).all(), f'Trial {i}: keypoints contain non-finite values'

    d = dict(d)
    d['keypoints'] = keypoints
    d['keypoints_name'] = names
    d['perturbation_indicator'] = pert
    return d


def split_trials(d, val_fraction=0.2, seed=42):
    """ Return (train_trial_idx, val_trial_idx). Splits by session if
    ``trial_session`` is present (held-out sessions), else by whole trials. A
    trial is never split across train/val. """
    n_trials = len(d['keypoints'])
    rng = np.random.RandomState(seed)
    if 'trial_session' in d and d['trial_session'] is not None:
        sessions = np.asarray(list(d['trial_session']))
        unique_sessions = np.unique(sessions)
        rng.shuffle(unique_sessions)
        n_val = max(1, int(round(len(unique_sessions) * val_fraction)))
        val_sessions = set(unique_sessions[:n_val].tolist())
        train_idx = [i for i in range(n_trials) if sessions[i] not in val_sessions]
        val_idx = [i for i in range(n_trials) if sessions[i] in val_sessions]
        split_kind = f'session ({len(val_sessions)} held-out sessions)'
    else:
        order = rng.permutation(n_trials)
        n_val = max(1, int(round(n_trials * val_fraction)))
        val_idx = sorted(order[:n_val].tolist())
        train_idx = sorted(order[n_val:].tolist())
        split_kind = 'trial'
    print(f'[KeypointPairDataset] split by {split_kind}: '
          f'{len(train_idx)} train trials, {len(val_idx)} val trials')
    return train_idx, val_idx


class KeypointPairDataset(data.Dataset):
    """ Yields adjacent in-trial pairs (x_t, x_{t+1}) with the opto target of the
    transition into t+1. ``seq_len`` is fixed at 2 (single adjacent pair). """

    def __init__(self, data_dict, trials=None):
        """
        Parameters
        ----------
        data_dict : dict
                    Already validated data dict (see ``validate_data``).
        trials : list[int] or None
                 Subset of trial indices to use. None => all trials.
        """
        super().__init__()
        self.D = len(data_dict['keypoints_name'])
        self.feature_names = list(data_dict['keypoints_name'])
        if trials is None:
            trials = list(range(len(data_dict['keypoints'])))
        self.trials = list(trials)

        # Store per-trial tensors.
        self.keypoints = [torch.as_tensor(np.asarray(data_dict['keypoints'][i]),
                                           dtype=torch.float32) for i in self.trials]
        self.pert = [torch.as_tensor(np.asarray(data_dict['perturbation_indicator'][i]),
                                     dtype=torch.float32) for i in self.trials]
        if 'trial_session' in data_dict and data_dict['trial_session'] is not None:
            self.sessions = [int(np.asarray(data_dict['trial_session'])[i]) for i in self.trials]
        else:
            self.sessions = None

        # Flat index of valid (local_trial_idx, t) pairs; never spans two trials.
        self.pairs = []
        for local_idx, kp in enumerate(self.keypoints):
            T_i = kp.shape[0]
            for t in range(T_i - 1):
                self.pairs.append((local_idx, t))
        assert len(self.pairs) > 0, 'No valid adjacent pairs (all trials length < 2?).'

    # --- interface mirroring Causal3DDataset ---
    def num_vars(self):
        return 1

    def target_names(self):
        return ['opto']

    def get_feature_names(self):
        return self.feature_names

    def get_input_dim(self):
        return self.D

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        local_idx, t = self.pairs[idx]
        kp = self.keypoints[local_idx]
        x = kp[t:t + 2]                       # [2, D]
        # opto realized in the transition into frame t+1; shape [seq_len-1, num_causal_vars] = [1, 1]
        target = self.pert[local_idx][t + 1].reshape(1, 1)
        return x, target

    # --- helpers for evaluation (Step 5) ---
    def all_frames(self):
        """ Return (X, opto, trial_id) over every frame of every trial in this
        split. X: [N, D]; opto: [N] in {0,1}; trial_id: [N] global trial index. """
        xs, optos, tids = [], [], []
        for local_idx, kp in enumerate(self.keypoints):
            xs.append(kp)
            optos.append(self.pert[local_idx])
            tids.append(torch.full((kp.shape[0],), self.trials[local_idx], dtype=torch.long))
        return torch.cat(xs, 0), torch.cat(optos, 0), torch.cat(tids, 0)
