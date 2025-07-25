#!/bin/bash

# Run the evaluation command
uv tool run --from git+https://github.com/ArcInstitute/cell-eval@main cell-eval prep \
    -i "path/to/your/prediction/file.h5ad" \
    -g "path/to/gene_names.csv"
