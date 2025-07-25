#!/bin/bash

# Run the training command for fine-tuning
uv run state tx train \
  data.kwargs.toml_config_path="path/to/your/dataset/file.toml" \
  data.kwargs.num_workers=4 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="path/to/your/embeddings/file.pt" \
  training.max_steps=30000 \
  training.ckpt_every_n_steps=1000 \
  model.kwargs.init_from="path/to/your/checkpoint/file.ckpt" \
  model=state_sm \
  wandb.tags="[finetune_run]" \
  output_dir="competition" \
  name="finetune_run" \
  +training.finetuning_schedule.enable=true \
  +training.finetuning_schedule.finetune_steps=2000 \
  +training.finetuning_schedule.modules_to_unfreeze='[pert_encoder]'
