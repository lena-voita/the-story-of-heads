# BERT Attention Head Pruning with Hard Concrete Gates

This repository implements attention head pruning for BERT using Hard Concrete Gates and L0 regularization, based on the method from ["Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned"](https://www.aclweb.org/anthology/P19-1580) (Voita et al., 2019).

## Overview

The implementation allows you to:
- Train BERT models on GLUE tasks with learnable gates on attention heads
- Automatically prune less important attention heads using L0 regularization
- Evaluate performance at different pruning ratios (10%, 30%, 50%, etc.)
- Visualize which heads are kept/pruned across all layers

## Method

### Hard Concrete Gates

Each attention head is multiplied by a stochastic gate `g_i` sampled from a Hard Concrete distribution:

```
MultiHead(Q, K, V) = Concat(g_i * head_i) W^O
```

The training objective adds an L0 penalty term:

```
L = L_task + λ * L_C
```

where:
- `L_task` is the task-specific loss (e.g., cross-entropy for classification)
- `L_C` is the expected number of non-zero gates (L0 regularization)
- `λ` controls the pruning strength

### Pruning Metrics

The code implements the following metrics:

1. **Gate Values (`g_i`)**: Learned scalar values in [0, 1] for each head
2. **L0 Penalty**: Expected number of active heads
3. **Sparsity Rate**: Fraction of active (non-zero) heads
4. **Pruning Ratio**: Fraction of heads pruned

## Installation

```bash
cd bert_pruning
pip install -r requirements.txt
```

## Usage

### 1. Train Single Model with Pruning

Train BERT on a GLUE task with specific L0 coefficient:

```bash
python scripts/train_glue.py \
    --task sst2 \
    --output_dir ./output/sst2_pruned \
    --l0_coefficient 0.1 \
    --num_epochs 3 \
    --batch_size 32 \
    --visualize_gates
```

Arguments:
- `--task`: GLUE task name (cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb)
- `--output_dir`: Where to save model and results
- `--l0_coefficient`: L0 regularization strength (0 = no pruning, higher = more pruning)
- `--visualize_gates`: Save heatmap of gate values

### 2. Evaluate Multiple Pruning Ratios

Compare baseline vs. 10%, 30%, and 50% pruning:

```bash
python scripts/evaluate_pruning_ratios.py \
    --task sst2 \
    --output_dir ./output/sst2_comparison \
    --pruning_ratios 0.1 0.3 0.5 \
    --l0_coefficients 0.01 0.1 0.5 \
    --num_epochs 3 \
    --batch_size 32
```

This will:
1. Train a baseline model (no pruning)
2. Train models with different L0 coefficients targeting 10%, 30%, and 50% pruning
3. Generate comparison table with metrics
4. Save visualizations of gate values

Arguments:
- `--pruning_ratios`: Target pruning ratios to evaluate
- `--l0_coefficients`: L0 coefficients for each ratio (matched by position)
- `--skip_baseline`: Skip training baseline model

### 3. Quick Start Examples

**SST-2 (Sentiment Analysis)**:
```bash
python scripts/evaluate_pruning_ratios.py \
    --task sst2 \
    --output_dir ./results/sst2 \
    --num_epochs 3
```

**MRPC (Paraphrase Detection)**:
```bash
python scripts/evaluate_pruning_ratios.py \
    --task mrpc \
    --output_dir ./results/mrpc \
    --num_epochs 3
```

**CoLA (Linguistic Acceptability)**:
```bash
python scripts/evaluate_pruning_ratios.py \
    --task cola \
    --output_dir ./results/cola \
    --num_epochs 3
```

## Output

### Directory Structure

After running experiments, the output directory contains:

```
output/
├── baseline/
│   ├── best_model/           # Best baseline model checkpoint
│   ├── best_metrics.json     # Evaluation metrics
│   └── args.json             # Training arguments
├── pruning_10pct/
│   ├── best_model/           # Best pruned model checkpoint
│   ├── best_metrics.json     # Evaluation metrics
│   ├── sparsity_info.json    # Detailed head-level sparsity
│   └── gate_values.png       # Visualization of gate values
├── pruning_30pct/
│   └── ...
├── pruning_50pct/
│   └── ...
├── comparison_results.csv    # Summary comparison table
└── detailed_results.json     # Full results for all configurations
```

### Comparison Table Example

```
Configuration              accuracy  Eval Loss  Pruning Ratio  Active Heads  Total Heads
-------------------------  --------  ---------  -------------  ------------  -----------
Baseline (0% pruning)        0.9200     0.1500          0.000           144          144
10% Pruning (L0=0.01)        0.9180     0.1520          0.098           130          144
30% Pruning (L0=0.1)         0.9100     0.1600          0.301           101          144
50% Pruning (L0=0.5)         0.8950     0.1850          0.496            73          144
```

### Gate Value Visualization

The `gate_values.png` shows a heatmap where:
- **Rows**: BERT layers (0-11)
- **Columns**: Attention heads (0-11)
- **Color**: Gate value (red = 0/pruned, green = 1/active)

## Code Structure

```
bert_pruning/
├── models/
│   ├── concrete_gate.py           # Hard Concrete Gate implementation
│   ├── bert_with_gates.py         # BERT model with prunable heads
│   └── __init__.py
├── utils/
│   ├── glue.py                    # GLUE task utilities
│   ├── pruning.py                 # Pruning helper functions
│   └── __init__.py
├── scripts/
│   ├── train_glue.py              # Single model training
│   └── evaluate_pruning_ratios.py # Multi-ratio evaluation
├── requirements.txt
└── README.md
```

## Supported GLUE Tasks

| Task | Metric | Type | Description |
|------|--------|------|-------------|
| CoLA | Matthews Correlation | Single | Linguistic acceptability |
| SST-2 | Accuracy | Single | Sentiment analysis |
| MRPC | F1 | Pair | Paraphrase detection |
| QQP | F1 | Pair | Quora question pairs |
| STS-B | Pearson Correlation | Pair | Semantic similarity (regression) |
| MNLI | Accuracy | Pair | Natural language inference |
| QNLI | Accuracy | Pair | Question answering NLI |
| RTE | Accuracy | Pair | Textual entailment |
| WNLI | Accuracy | Pair | Winograd NLI |

## Advanced Usage

### Custom L0 Coefficients

Fine-tune L0 coefficient for specific pruning target:

```bash
# For ~10% pruning, try L0 = 0.01-0.05
python scripts/train_glue.py --task sst2 --l0_coefficient 0.01 --output_dir ./output/light_pruning

# For ~30% pruning, try L0 = 0.05-0.2
python scripts/train_glue.py --task sst2 --l0_coefficient 0.1 --output_dir ./output/medium_pruning

# For ~50% pruning, try L0 = 0.2-1.0
python scripts/train_glue.py --task sst2 --l0_coefficient 0.5 --output_dir ./output/heavy_pruning
```

### Analyze Pruning Results

Load and analyze sparsity information:

```python
import json

# Load sparsity info
with open('./output/sst2_pruned/sparsity_info.json') as f:
    sparsity_info = json.load(f)

# Print head-level details
for layer_info in sparsity_info:
    print(f"Layer {layer_info['layer']}:")
    print(f"  Active heads: {layer_info['num_active_heads']}/{layer_info['total']}")
    print(f"  Gate values: {layer_info['gate_values']}")
```

### Load Pruned Model for Inference

```python
from transformers import BertTokenizer
from models import BertForSequenceClassificationWithGates

# Load model
model = BertForSequenceClassificationWithGates.from_pretrained('./output/sst2_pruned/best_model')
tokenizer = BertTokenizer.from_pretrained('./output/sst2_pruned/best_model')

# Get pruning info
sparsity_info = model.get_sparsity_info()
print(f"Model has {sum(info['num_active_heads'] for info in sparsity_info)} active heads")

# Inference
inputs = tokenizer("This movie is great!", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1)
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{voita-etal-2019-analyzing,
    title = "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned",
    author = "Voita, Elena and Talbot, David and Moiseev, Fedor and Sennrich, Rico and Titov, Ivan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1580",
    pages = "5797--5808",
}
```

## Notes

- **L0 Coefficient Selection**: The relationship between L0 coefficient and final pruning ratio is non-linear and task-dependent. Start with suggested values and tune based on results.

- **Training Time**: Models with L0 regularization train slightly slower due to gate sampling and penalty computation.

- **Deterministic Pruning**: At inference time, gates use deterministic values (no sampling) based on learned parameters.

- **Head Importance**: Generally, lower layers tend to have more important heads than upper layers.

## Troubleshooting

**Issue**: Model achieves 0% or 100% pruning
- **Solution**: Adjust L0 coefficient. Too low = no pruning, too high = over-pruning.

**Issue**: Performance drops significantly with pruning
- **Solution**: Try:
  1. Lower the target pruning ratio
  2. Train for more epochs
  3. Use a lower learning rate
  4. Increase warmup ratio

**Issue**: Out of memory errors
- **Solution**: Reduce batch size or max sequence length

## License

This implementation is provided for research purposes. The method is based on the work by Voita et al. (2019).
