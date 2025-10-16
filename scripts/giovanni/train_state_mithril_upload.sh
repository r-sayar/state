#srun --job-name=pipeline-state --qos=standard  --mem=80G --time 00:30:00 bash scripts/robin/pipeline_holdouts_robin.sh --gres=gpu:A5000:1 --cpus-per-task=4
#!/bin/bash
# infering for the competition submission and getting the .vcc file

# -- Config --
# Path to your trained model directory where your models are
MODEL_DIR="../state-big/models/"

# experiment name
DIR_NAME="few-params-baseline"

WANDB_PROJECT="vcc"
WANDB_ENTITY="rsayar728-freie-universit-t-berlin"

# toml config path
TOML_CONFIG="scripts/giovanni/all_data.toml"

# Competition support set -> why is this important? 
COMPETITION_SUPPORT_SET="../state-big/data"

# perturbation features file
PERT_FEATURES="../state-big/data/ESM2_pert_features.pt"
#"datasets/embeddings/GenePT_gene_embedding_ada_text.pt"
#


# output directory for results
OUT_DIR=../state-big/model-results/pipeline_holdouts_robin

# parallelization
THREADS=16
NUM_WORKERS=16
BATCH_SIZE=200

# Exit on error
set -e
export OMP_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export HDF5_USE_FILE_LOCKING=FALSE


# Copy this script into the model directory for reproducibility
DEST_DIR="${MODEL_DIR}/${DIR_NAME}"
mkdir -p "$DEST_DIR"

# Path of the currently running script (works when executed or sourced)
SRC="${BASH_SOURCE[0]:-$0}"

# Resolve to an absolute path if possible
if command -v realpath >/dev/null 2>&1; then
  SRC_ABS=$(realpath "$SRC")
elif command -v readlink >/dev/null 2>&1; then
  SRC_ABS=$(readlink -f "$SRC" 2>/dev/null || printf "%s" "$SRC")
else
  SRC_ABS="$SRC"
fi

cp -a "$SRC_ABS" "$DEST_DIR/" || cp -a "$SRC" "$DEST_DIR/"


# copy toml config into the model directory for reproducibility
if [ -e "$TOML_CONFIG" ]; then
  cp -a "$TOML_CONFIG" "$DEST_DIR/$(basename "$TOML_CONFIG")"
else
  echo "Warning: TOML_CONFIG '$TOML_CONFIG' does not exist. Skipping copy." >&2
fi

echo "#### Running training ####"


uv run state tx train \
  data.kwargs.toml_config_path=${TOML_CONFIG} \
  data.kwargs.num_workers=${NUM_WORKERS} \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file=${PERT_FEATURES} \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=1000 \
  training.val_freq=1000 \
  model=state_sm \
  model.kwargs.nb_decoder=true \
  wandb.tags=[${DIR_NAME}] \
  wandb.project=${WANDB_PROJECT} \
  wandb.entity=${WANDB_ENTITY} \
  output_dir=${MODEL_DIR} \
  name=${DIR_NAME} \
  training.lr=1e-4 \
  use_wandb=true \
  model.kwargs.n_encoder_layers=3 \
  model.kwargs.n_decoder_layers=3 \
  +trainer.accelerator=gpu \

#model.kwargs.transformer_backbone_kwargs.num_hidden_layers=2 \

echo "#### Running inference ####"

# gets prediction.h5ad for the holdout predictions
uv run state tx infer \
  --output ${MODEL_DIR}/${DIR_NAME}/prediction.h5ad \
  --model_dir ${MODEL_DIR}/${DIR_NAME} \
  --checkpoint ${MODEL_DIR}/${DIR_NAME}/checkpoints/final.ckpt \
  --adata ${OUT_DIR}/.h5ad \
  --pert_col target_gene

#testing
mkdir -p "${MODEL_DIR}/${DIR_NAME}/inferred"

echo "#### Running inference for each checkpoint (competition template) ####"

COMP_INFER_DIR="${MODEL_DIR}/${DIR_NAME}/inferred/competition_val"
mkdir -p "$COMP_INFER_DIR"


COMP_ADATA="../state-big/data/competition_support_set/competition_val_template.h5ad"
CKPT_DIR="${MODEL_DIR}/${DIR_NAME}/checkpoints"
shopt -s nullglob
for ckpt in "$CKPT_DIR"/*.ckpt; do
  ckpt_name=$(basename "$ckpt" .ckpt)
  out="${COMP_INFER_DIR}/${ckpt_name}.h5ad"
  echo "Running competition inference for checkpoint: $ckpt -> $out"
  state tx infer \
    --output "$out" \
    --model-dir "${MODEL_DIR}/${DIR_NAME}" \
    --checkpoint "$ckpt" \
    --adata "$COMP_ADATA" \
    --pert-col target_gene
done
shopt -u nullglob

echo "#### Running cell-eval prep ####"
# # remember to have `sudo apt install -y zstd` before running this
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep -i ${MODEL_DIR}/${DIR_NAME}/competition_val_prediction.h5ad -g ${COMPETITION_SUPPORT_SET}/gene_names.csv
