#!/bin/bash

# Run the inference command
# NOTE: model_dir looks for a ./checkpoints/final.ckpt to infer on
uv run state tx infer \
  --model_dir "competition/dyno005" \
  --adata "competition_support_set/competition_val_template.h5ad" \
  --output "competition/dyno005/prediction.h5ad" \
  --pert_col "target_gene"
