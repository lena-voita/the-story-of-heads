#!/bin/bash
# Quick test on SST-2 with reduced settings for debugging
# Usage: bash quick_test.sh

echo "Running quick test on SST-2..."

python scripts/train_glue.py \
    --task sst2 \
    --output_dir ./test_output/sst2_baseline \
    --l0_coefficient 0.0 \
    --num_epochs 1 \
    --batch_size 16 \
    --max_length 64 \
    --seed 42

echo ""
echo "Training with L0 regularization..."

python scripts/train_glue.py \
    --task sst2 \
    --output_dir ./test_output/sst2_pruned \
    --l0_coefficient 0.1 \
    --num_epochs 1 \
    --batch_size 16 \
    --max_length 64 \
    --visualize_gates \
    --seed 42

echo ""
echo "Test completed! Check ./test_output/ for results."
