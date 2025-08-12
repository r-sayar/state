import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

# Load your trained model (following the pattern from _predict.py and _infer.py)
checkpoint_path = "results/competition/hvgs_state_sm_esm2/checkpoints/last.ckpt"
model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
model.eval()

# View the output projection layer architecture
print("Output projection layer:")
print(model.project_out)

# Access individual layer parameters
for name, param in model.project_out.named_parameters():
    print(f"Parameter: {name}")
    print(f"Shape: {param.shape}")
    print(f"First few values: {param.data.flatten()[:10]}")