# Cell-Type-Aware Enhancer Activity Prediction

## Overview

This project investigates whether enhancer activity can be predicted directly from DNA sequence while incorporating cell-type information.

We compare several convolutional neural network (CNN) architectures:

- **Baseline CNN** – sequence-only model
- **Embedding CNN** – sequence model with cell-type embeddings
- **Attention CNN** – sequence model with an attention mechanism
- **Full Model** – combines cell-type embeddings and attention

The goal is to evaluate whether explicit cell-type information improves enhancer activity prediction across multiple human cell lines.

---

## Dataset

The dataset contains enhancer activity measurements from three human cell types:

- HepG2
- K562
- WTC11

Each sample consists of:

- one-hot encoded DNA sequence
- cell type identifier
- measured enhancer activity

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
|
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── plots.py
│   └── analyze_by_cell_type.py
│
├── Snakefile
├── requirements.txt
└── README.md
```

---

## Model Variants

### Baseline

CNN operating only on DNA sequence.

### Embedding

CNN with a learned cell-type embedding concatenated to sequence features.

### Attention

CNN augmented with an attention mechanism over sequence features.

### Full

CNN combining both cell-type embeddings and attention.

---

## Evaluation Metrics

Models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson Correlation
- Spearman Correlation
- R² Score

---

## Results

### Overall Performance

| Model | MAE | RMSE | Pearson | Spearman | R² |
|---------|---------|---------|---------|---------|---------|
| Baseline | 0.607 | 0.795 | 0.434 | 0.369 | 0.186 |
| Embedding | **0.577** | **0.769** | **0.489** | **0.479** | **0.238** |
| Attention | 0.608 | 0.804 | 0.408 | 0.361 | 0.166 |
| Full | 0.579 | 0.772 | 0.486 | 0.468 | 0.233 |

The embedding-based model achieved the best overall performance across nearly all evaluation metrics.

### Cell-Type Specific Performance (Embedding Model)

| Cell Type | Pearson | Spearman | R² |
|------------|----------|----------|----------|
| HepG2 | 0.341 | 0.337 | 0.113 |
| K562 | 0.433 | 0.371 | 0.186 |
| WTC11 | 0.523 | 0.488 | 0.271 |

The model performs best on WTC11 and worst on HepG2, indicating substantial differences in enhancer predictability between cellular contexts.

---

## Key Findings

- Cell-type information improves enhancer activity prediction.
- Learned embeddings consistently outperform the sequence-only baseline.
- Adding attention alone provides limited benefit.
- Combining embeddings and attention does not outperform embeddings alone.
- Prediction quality differs across cell types.
- WTC11 exhibits the strongest predictive signal.
- Extremely high enhancer activities are systematically underestimated by all models.

---

## Running the Project

### Train models

```bash
python src/train.py
```

### Evaluate models

```bash
python src/evaluate.py
```

### Generate plots

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
