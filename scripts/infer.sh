#!/bin/bash

# Run the inference command
# NOTE: model_dir looks for a ./final.ckpt to infer on
uv run state tx infer \
  --model_dir "path/to/models/dir" \
  --adata "path/to/competition_val_template.h5ad" \
  --output "path/to/your/prediction/file.h5ad" \
  --pert_col "target_gene"
