#!/usr/bin/env bash
# End-to-end sample: generate synthetic keypoint data, train CITRIS-VAE on it,
# then run the GT-free evaluation / Step-7 verification report.
#
# Replace the synthetic data step with your own .npz/.pkl that follows the
# data contract (keys: keypoints, keypoints_name, perturbation_indicator,
# optional trial_session) and point --data_path at it.
set -e

cd "$(dirname "$0")/.."

DATA=data_generation/keypoints_synthetic.npz
RUN_DIR=checkpoints/keypoints/CITRISVAE_keypoints

# 1) Synthetic data (skip if you already have real data).
if [ ! -f "$DATA" ]; then
  python data_generation/data_generation_keypoints_synthetic.py
fi

# 2) Train. (For a quick smoke test, drop --max_epochs to e.g. 5.)
python experiments/train_vae_keypoints.py \
  --config experiments/configs/keypoints_example.json \
  --data_path "$DATA"

# 3) Locate the latest run's checkpoint and evaluate.
LATEST=$(ls -td "$RUN_DIR"/version_* | head -1)
CKPT=$(ls "$LATEST"/checkpoints/best-*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then CKPT=$(ls "$LATEST"/checkpoints/last.ckpt | head -1); fi
echo "Evaluating checkpoint: $CKPT"

python experiments/evaluate_keypoints.py \
  --checkpoint "$CKPT" \
  --data_path "$DATA" \
  --out_dir "$LATEST/eval" \
  --split val \
  --fingertip_key fingertip \
  --port_key port

echo "Done. See $LATEST/eval/report.md"
