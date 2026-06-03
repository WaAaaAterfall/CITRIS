# Instruction: Adapt CITRIS-VAE to mouse reaching keypoint data (opto interventions)

You are adapting the CITRIS codebase (`https://github.com/phlippe/CITRIS`, ICML 2022 / ICLR 2023, Lippe et al.) to a **low-dimensional keypoint** dataset from a mouse reach-to-consume task. Do **not** treat this as image data. Reuse the CITRIS *causal* machinery (transition prior, learned target→latent assignment, ELBO) and replace everything image-specific.

Clone the repo and **read the actual source before writing anything** — this document describes intent and the data contract, but the repo is the ground truth for interfaces. If anything here conflicts with the code, surface the conflict rather than silently resolving it.

Key files to read first:
- `models/citris_vae/lightning_module.py` — `CITRISVAE`, especially `__init__`, `forward`, `encode`, `_get_loss`, `validation_step`, `triplet_evaluation`.
- `models/shared/transition_prior.py` — `TransitionPrior` (the causal prior conditioned on intervention targets).
- `models/shared/target_classifier.py` — the learned assignment of latents to intervention blocks.
- `experiments/datasets.py` — `Causal3DDataset` (note its `sub_indices` mechanism) and `InterventionalPongDataset` for the expected batch format.
- `experiments/train_vae.py` — training entry point and CLI args.

---

## Critical design decisions (do not "fix" these)

1. **Observations are keypoints, not images.** Input is a `D`-dimensional continuous vector per frame (`D = 76`: two camera views, 2D Cartesian keypoints, already min–max normalized all keypoints together so min max range for all points are [0, 1] in order to keep the relatively distance between points). Replace the conv `Encoder`/`Decoder` with MLPs (see Step 3). Do **not** use `no_encoder_decoder=True` — that path is for CITRIS-NF where inputs are pre-encoded latents; we want a real variational MLP encoder/decoder.

2. **One intervention channel.** The only perturbation present is optogenetic inhibition. Therefore `num_causal_vars = 1`. Be aware: `datasets.py` drops target dimensions that are never toggled, so even if more channels were declared they would collapse. Consequence to respect, not work around: CITRIS will identify a **two-way partition** — the opto-affected subspace vs. ψ⁰ (everything else). It will *not* separate 3–4 distinct causal factors from a single intervention. Do not add fake intervention channels to manufacture more blocks.

   **Do not add target-port location as a second intervention channel.** The water port is fixed within each trial (it changes only between trials), so a location flag is constant across every (t, t+1) transition and is therefore *not* a CITRIS intervention — the identifiability theorem covers factors acted on *at a transition*. Putting it in the target vector would run (it survives the `has_intv` filter because it varies across trials) but would silently swap CITRIS's intervention-based identifiability for iVAE/TCL-style nonstationarity identifiability, which is a different and unstated mechanism. Additionally, the port is a tracked keypoint, so its position is already in the `D=76` input; the latent quantity of interest is the animal's *goal representation* of the port, for which the observed port position is a validation regressor, not a model input. Location therefore belongs in the observable/auxiliary role (Step 5.2), and `num_causal_vars` stays 1.

3. **`num_latents` ≠ number of causal variables.** `num_latents` is the total latent dimensionality and may exceed the number of factors (ψ⁰ absorbs the remainder, as in the original). Set it small and inspectable (default 8; expose as config). The user's "3 or 4" refers to inspectability, not to a guaranteed number of identified factors.

4. **Interventions are imperfect.** Opto inhibition is a soft suppression, not a perfect do-intervention. Set `imperfect_interventions=True` so the prior retains context features rather than masking them.

5. **Transition unit is a single adjacent pair** `(x_t, x_{t+1})` with target `I_{t+1}` (`seq_len = 2`). Do **not** build long sequences or use padding/masking. Handle variable trial length by indexing valid in-trial pairs only (Step 2).

6. **No ground-truth factors exist.** The default `validation_step` uses `triplet_evaluation` + a supervised `CausalEncoder`, which requires labels we do not have. Pass `causal_encoder_checkpoint=None` and **replace validation/test** with the GT-free diagnostics in Step 5. Do not fabricate triplet data.

---

## Step 0 — Environment

The repo targets PyTorch 1.10 / PyTorch Lightning 1.6. If the pinned Lightning version conflicts with the user's environment, prefer writing a **minimal training loop / LightningModule subclass** that reuses `TransitionPrior` and `TargetClassifier` verbatim, rather than downgrading the whole environment. Report which path you took.

## Step 1 — Data contract

The user will provide a single pickled/`.npz` dict with **exactly these keys** (match the names exactly):

```python
{
  "keypoints":              list[np.ndarray],  # len = N_trials; each array float32, shape [T_i, D], D=76, T_i in [30,100]
  "keypoints_name":         list[str],          # len = D; name of each keypoint feature (e.g. "view0_fingertip_x")
  "perturbation_indicator": list[np.ndarray],   # len = N_trials; each array {0,1}, shape [T_i]; 1 = opto ON at that frame
}
```

Optional keys that may be present — use if available, otherwise infer:
- `"trial_session"`: `list[int]`, session id per trial (use for train/val splitting by session so val is held-out sessions, not held-out frames).

Write a `validate_data(d)` function that asserts: lengths of the three lists match across trials; `keypoints[i].shape[1] == len(keypoints_name)` for all `i`; `perturbation_indicator[i].shape[0] == keypoints[i].shape[0]`; values in `perturbation_indicator` are in `{0,1}`; keypoints are finite. Fail loudly with the offending trial index.

Assume keypoints are **already normalized**; do not re-normalize. If a `norm_params.npz` is passed, store its path in the run config for provenance only.

## Step 2 — `KeypointPairDataset`

Build a `torch.utils.data.Dataset` modeled on `Causal3DDataset`'s `sub_indices` pattern:

- Precompute a flat index `pairs = [(trial_idx, t) for trial_idx in trials for t in range(T_i - 1)]`. **Never** create a pair spanning two trials.
- `__getitem__` returns `(x, target)` where:
  - `x`: float32 tensor shape `[2, D]` = `keypoints[trial_idx][t:t+2]`.
  - `target`: float32 tensor shape `[1]` = `perturbation_indicator[trial_idx][t+1]` (the intervention realized in the transition into frame t+1).
- This matches `CITRISVAE._get_loss`'s `len(batch) == 2 → imgs, target` path, with `imgs` of shape `[B, 2, D]`. Verify against `_get_loss` that the pair axis is flattened as expected and adjust the encoder call accordingly.
- Train/val split: split by **session** if `trial_session` is present, else by **trial** (hold out whole trials, never split a trial across train/val).
- Expose `num_vars()` → 1, `target_names()` → `["opto"]`, and the feature names for logging.

## Step 3 — MLP encoder / decoder

Add `models/shared/mlp_encoder_decoder.py` (or extend `modules.py`):
- `MLPEncoder(D_in=76, num_latents, hidden, n_layers, act_fn)`: variational; outputs `mean` and `log_std` over `num_latents`. Mirror the interface the conv `Encoder(variational=True)` exposes so `CITRISVAE.encode`/`forward` work unchanged.
- `MLPDecoder(num_latents, D_out=76, hidden, n_layers, act_fn)`: outputs the reconstruction. Use a **Gaussian** observation likelihood (predict mean, fixed or learned scalar log-variance) so `gaussian_log_prob` (already imported in the VAE module) is reused for the reconstruction term. Confirm how the existing decoder's output feeds the reconstruction loss and match it.

Wire these into `CITRISVAE.__init__` behind a flag (e.g. `obs_type='keypoints'`) so the conv path is untouched. Keep `img_width`/`c_in` arguments inert when `obs_type='keypoints'`.

## Step 4 — Model configuration

Instantiate `CITRISVAE` (or the minimal subclass from Step 0) with:
- `num_latents = 8` (config), `num_causal_vars = 1`, `imperfect_interventions = True`.
- `causal_encoder_checkpoint = None`, `use_flow_prior` left at the repo default (verify it does not assume image shapes; disable if it does).
- `var_names = ["opto"]`, `lambda_reg` at default (regularizer pushing intervention-independent info into ψ⁰).
- Keep the `TargetClassifier` and `TransitionPrior` exactly as in the repo.

## Step 5 — Replace evaluation (GT-free)

Remove the triplet path from `validation_step`/`test_step`. Implement these diagnostics, logged each validation epoch:

1. **Intervention-consistency (primary).** Encode all val frames. For the latent dimensions the `TargetClassifier` assigns to the opto block vs. ψ⁰, compute: (a) effect size (e.g. AUC or standardized mean difference) of each latent dim's distribution on opto-ON vs opto-OFF frames; (b) confirm opto-block dims show large effects and ψ⁰ dims show small ones. This is the core "did it work" signal.
2. **Latent ↔ observable correlation.** Compute simple derived kinematics from the raw keypoints that were **not** given as targets — at minimum: fingertip speed, distance from fingertip to water-port keypoint, and (where the first-water-contact frame is provided) time-to-contact. Report the correlation matrix between each recovered latent dim and each derived observable. Save it as a heatmap.

   Include **target-port identity/position** (constant within trial) as an auxiliary regressor here — this is the key validation for any putative goal latent: does a recovered latent dim track which of the 4–5 port locations the trial used? Use the observed port keypoint position; do not feed it as a model input. Report this as correlation/decoding accuracy of port identity from the latents.
3. **Reconstruction**: per-feature reconstruction error on val, reported by keypoint name.

Provide a `--no_gt_eval` style switch is unnecessary — this is the only eval. Do not silently fall back to triplet code.

## Step 6 — Training entry point + config

Create `experiments/train_vae_keypoints.py` (copy structure from `train_vae.py`, strip image/triplet args). Drive it from a single YAML/JSON config containing: data path, `num_latents`, `num_causal_vars=1`, `imperfect_interventions=true`, hidden sizes, lr, warmup, max_iters, batch size, split strategy, seed. Log to the same logging backend the repo uses (verify which). Save checkpoints + the latent→block assignment from the `TargetClassifier`.

## Step 7 — First verification milestone

Before any further modeling, produce a short report that answers one question: **does CITRIS isolate a latent subspace that responds to opto while leaving a complementary subspace approximately opto-invariant?** Deliver: the intervention-consistency effect sizes per latent dim (Step 5.1), the latent↔observable heatmap (Step 5.2), and the learned latent→block assignment. If the opto block does not separate, report that as a negative result — do not tune until it looks positive.

---

## Explicitly out of scope (do not build yet)
- Force-field perturbations (no such trials exist yet).
- Neural data / joint behavior+neural modeling.
- iCITRIS / instantaneous-effect causal discovery (single intervention channel does not warrant it).
- Multi-step sequence modeling, padding, or masking.

## Things to confirm against the repo and flag if they differ from this doc
- Exact batch unpacking in `_get_loss` (pair axis handling) and whether `target` is expected as float `[B, num_causal_vars]`.
- Whether `use_flow_prior` or any default callback assumes image tensors.
- The reconstruction-likelihood call the decoder feeds into, so the MLP decoder matches it.
- Whether `num_causal_vars=1` triggers any edge case in `TargetClassifier` (binary block assignment).
