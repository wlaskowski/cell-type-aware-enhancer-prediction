````markdown
# Cell-Type-Aware Enhancer Activity Prediction

## Overview

This project investigates whether regulatory activity can be predicted directly from DNA sequence and whether incorporating cell-type information improves predictive performance.

Four convolutional neural network (CNN) architectures are compared:

- **Baseline CNN** – sequence-only model
- **Embedding CNN** – sequence model with learned cell-type embeddings
- **Attention CNN** – sequence model with attention-based pooling
- **Full CNN** – combines cell-type embeddings and attention-based pooling

The main objective is to evaluate whether explicit cell-type information improves aggregate prediction performance across multiple human cell types.

---

## Dataset

The dataset was derived from a large-scale ENCODE Phase 4 Massively Parallel Reporter Assay (MPRA) experiment.

The analyzed joint MPRA library consists primarily of potential enhancers, together with a smaller number of promoters and experimental control sequences. Regulatory activity was measured in three human cell types:

- HepG2
- K562
- WTC11

Each observation contains:

- a one-hot encoded DNA sequence
- a cell-type identifier
- a measured regulatory activity value represented as `log2(RNA/DNA)`

The dataset was split by regulatory element identifier into training (70%), validation (15%), and test (15%) sets. This prevents the same DNA sequence from appearing in more than one subset.

---

## Project Structure

```text
cell-type-aware-enhancer-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── results/
├── figures/
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── plots.py
│   └── analyze_by_cell_type.py
│
├── Snakefile
├── requirements.txt
└── README.md
```

---

## Model Variants

### Baseline CNN

A convolutional neural network operating only on the DNA sequence.

### Embedding CNN

The baseline CNN extended with a learned cell-type embedding concatenated with the sequence representation before the regression layers.

### Attention CNN

A sequence-only CNN in which global average pooling is replaced with attention-based pooling over the final convolutional feature map.

### Full CNN

A model combining cell-type embeddings with attention-based pooling.

---

## Evaluation Metrics

Models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson correlation
- Spearman correlation
- Coefficient of determination (R²)

---

## Results

### Overall Performance

| Model | MSE | MAE | RMSE | Pearson | Spearman | R² |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.632 | 0.607 | 0.795 | 0.433 | 0.370 | 0.186 |
| Embedding | **0.591** | **0.577** | **0.769** | **0.489** | **0.480** | **0.239** |
| Attention | 0.647 | 0.608 | 0.805 | 0.408 | 0.362 | 0.166 |
| Full | 0.595 | 0.579 | 0.772 | 0.486 | 0.468 | 0.233 |

The embedding-based model achieved the best aggregate performance on the combined test set. The full model produced similar results, while attention alone did not improve upon the sequence-only baseline.

### Cell-Type-Specific Performance of the Embedding Model

| Cell Type | Pearson | Spearman | R² |
|---|---:|---:|---:|
| HepG2 | 0.341 | 0.337 | 0.113 |
| K562 | 0.433 | 0.371 | 0.186 |
| WTC11 | 0.523 | 0.488 | 0.271 |

WTC11 achieved the highest Pearson correlation and R², indicating that the model captured relative variation most effectively for this cell type.

---

## Key Findings

- Cell-type embeddings improved aggregate performance on the combined test set.
- The embedding model achieved the best overall evaluation metrics.
- Attention-based pooling alone did not improve upon the baseline model.
- Combining embeddings and attention did not outperform embeddings alone.
- Predictive performance differed between cell types.
- WTC11 achieved the highest Pearson correlation and R².
- Highly active regulatory elements were frequently underestimated.

---

## Running the Project

### Run the complete workflow

```bash
snakemake --cores 1
```

This command performs preprocessing, trains all model variants, calculates evaluation metrics, and generates the result figures.

### Train individual model variants manually

```bash
python src/train.py --variant baseline --epochs 30
python src/train.py --variant embedding --epochs 30
python src/train.py --variant attention --epochs 30
python src/train.py --variant full --epochs 30
```

Available model variants:

```text
baseline
embedding
attention
full
```

### Generate model comparison plots

```bash
python src/plots.py
```

### Analyze performance by cell type

```bash
python src/analyze_by_cell_type.py
```

---

## Author

Wojciech Laskowski
````
