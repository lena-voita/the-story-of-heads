#!/bin/bash
# Run pruning experiments on all GLUE tasks
# Usage: bash run_all_tasks.sh [output_base_dir]

OUTPUT_BASE=${1:-./results}

# GLUE tasks to run
TASKS=("sst2" "mrpc" "cola" "rte" "qnli")

# Pruning ratios and corresponding L0 coefficients
PRUNING_RATIOS="0.1 0.3 0.5"
L0_COEFFICIENTS="0.01 0.1 0.5"

echo "=========================================="
echo "BERT Head Pruning on GLUE Tasks"
echo "=========================================="
echo "Output directory: $OUTPUT_BASE"
echo "Tasks: ${TASKS[@]}"
echo "Pruning ratios: $PRUNING_RATIOS"
echo "L0 coefficients: $L0_COEFFICIENTS"
echo "=========================================="

for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running task: $task"
    echo "=========================================="

    python scripts/evaluate_pruning_ratios.py \
        --task $task \
        --output_dir ${OUTPUT_BASE}/${task} \
        --pruning_ratios $PRUNING_RATIOS \
        --l0_coefficients $L0_COEFFICIENTS \
        --num_epochs 3 \
        --batch_size 32 \
        --seed 42

    if [ $? -eq 0 ]; then
        echo "✓ Task $task completed successfully"
    else
        echo "✗ Task $task failed"
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="
