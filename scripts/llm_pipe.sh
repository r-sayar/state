# State TX Training Command
# This setups up training for State across datasets, using ESM2 featurizations
# of genes as the perturbation embeddings. Note that we are now generalizing
# across both contexts and perturbations (not just contexts)


! uv run state tx train \
  data.kwargs.toml_config_path="datasets/competition_support_set/starter.toml" \
  data.kwargs.num_workers=12 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="datasets/competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=20000 \
  training.ckpt_every_n_steps=5000 \
  model=state_sm \
  wandb.tags="[first_run]" \
  output_dir="results/competition" \
  name="hvgs_2_state_sm_esm2" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.output_space="gene" \