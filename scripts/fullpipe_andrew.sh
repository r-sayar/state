#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory
MODEL_DIR="competition"

# experiment name
NAME="training_finetune_andrew"

# old checkpoint name
# CKPT="step30000l1d063.ckpt"

# toml config path
TOML_CONFIG="examples/andrewfewshot.toml"

# number of workers
NUM_WORKERS=8

# perturbation features file
PERT_FEATURES="competition_support_set/ESM2_pert_features.pt"


# -- Activate working environment --
# TODO:Activate working environment

# Exit on error
set -e

# Run the training command
uv run state tx train \
  data.kwargs.toml_config_path="${TOML_CONFIG}" \
  data.kwargs.num_workers="${NUM_WORKERS}" \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="${PERT_FEATURES}" \
  training.max_steps=1000 \
  training.ckpt_every_n_steps=500 \
  training.val_freq=100 \
  model=state_sm \
  model.kwargs.nb_decoder=true \
  wandb.tags="[${NAME}]" \
  output_dir="${MODEL_DIR}" \
  name="${NAME}" \
  use_wandb=false

# -- Predict --
# gets metrics.csv along with real and predicted adata from test holdouts
uv run state tx predict \
    --checkpoint "final.ckpt" \
    --output_dir "${MODEL_DIR}/${NAME}/" \
    --profile full

# -- Infer --
# gets prediction.h5ad for the competition submission
uv run state tx infer \
  --output "${MODEL_DIR}/${NAME}/prediction.h5ad" \
  --model_dir "${MODEL_DIR}/${NAME}" \
  --checkpoint "${MODEL_DIR}/${NAME}/checkpoints/${CKPT}" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --pert_col "target_gene"

# -- Gets prediction.vcc --
# remember to have `sudo apt install -y zstd` before running this
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${NAME}/prediction.h5ad -g competition_support_set/gene_names.csv
