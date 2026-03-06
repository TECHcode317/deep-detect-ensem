#!/bin/bash
set -e

echo "=== Deepfake Detection Reproducibility Run ==="
echo "Date: $(date)"
echo "Commit: $(git rev-parse HEAD)"

# Set seeds
export PYTHONHASHSEED=42

# Create results directory
mkdir -p results/{tables,figures,models}

# Run experiments
echo -e "\n=== Experiment 1: Model Training ==="
python experiments/Frame.py --save-models results/models/

echo -e "\n=== Experiment 2: Video Metrics ==="
python experiments/video mul seed.py --seeds 42 77 123

echo -e "\n=== Experiment 3: Baseline Comparisons ==="
python experiments/Baseline run.py

# Generate tables and figures
echo -e "\n=== Generating Tables ==="
python scripts/reproduce_tables.py --output results/tables/

echo -e "\n=== Generating Figures ==="
python scripts/reproduce_figures.py --output results/figures/

# Compute hash for verification
echo -e "\n=== Results Hash ==="
find results -type f -exec md5sum {} \; | sort -k2 | md5sum

echo -e "\n✓ All experiments completed successfully!"