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
  data.kwargs.perturbation_features_file="datasets/GenePT_embedding_v2/GenePT_gene_embedding_ada_text.pickle" \
  data.kwargs.output_space="gene" \
  training.max_steps=40000 \
  training.ckpt_every_n_steps=5000 \
  model=state_sm \
  wandb.tags="[first_run]" \
  output_dir="results/competition" \
  name="hvgs_state_sm_genept" \