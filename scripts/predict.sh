#!/bin/bash

DIR_NAME="first_run"

# Edit the predict command to add "state" mapping
sed -i '' 's/elif model_class_name.lower() in \["neuralot", "pertsets"\]:/elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:/' src/state/_cli/_tx/_predict.py

uv run state tx predict \
    --checkpoint "final.ckpt" \
    --output_dir "competition/${DIR_NAME}/" \
    --profile "full"

