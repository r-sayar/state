
# Create output directory if it doesn't exist
mkdir -p ../datasets/competition_support_set/hvgs

# Preprocess each dataset individually
state tx preprocess_train \
  --adata ../datasets/competition_support_set/competition_train.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_competition_train.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/competition_support_set/k562_gwps.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_k562_gwps.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/competition_support_set/rpe1.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_rpe1.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/competition_support_set/jurkat.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_jurkat.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/competition_support_set/k562.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_k562.h5ad \
  --num_hvgs 2000

state tx preprocess_train \
  --adata ../datasets/competition_support_set/hepg2.h5 \
  --output ../datasets/competition_support_set/hvgs/preprocessed_training_data_hepg2.h5ad \
  --num_hvgs 2000
