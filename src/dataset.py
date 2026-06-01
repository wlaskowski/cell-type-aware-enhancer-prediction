import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset

PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

SEQ_MAPPING = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

CELL_MAPPING = {
    "HepG2": 0,
    "K562": 1,
    "WTC11": 2,
}


def encode_sequence(seq):
    """
    Converts a DNA sequence string into a one-hot encoded matrix.
    """
    x = np.array(
        [SEQ_MAPPING.get(base, [0, 0, 0, 0]) for base in str(seq).upper().strip()],
        dtype=np.float32,
    )

    return x.T # needed for Conv1d

def encode_cell_type(cell_type):
    """
    Converts a cell type string into an integer label.
    """
    CELL_MAPPING = {
        "HepG2": 0,
        "K562": 1,
        "WTC11": 2,
    }

    return np.array(
        [CELL_MAPPING[c] for c in cell_type],
        dtype=np.int64,
    )


def load_dataset(path):
    df = pd.read_csv(path, sep=",")

    x = np.array([encode_sequence(seq) for seq in df["sequence"]])

    y = df["activity"].values.astype(np.float32)

    cell_type = encode_cell_type(df["cell_type"])

    return df, x, y, cell_type


def load_train_val_test():
    train_df, x_train, y_train, cell_train = load_dataset(TRAIN_PATH)
    val_df, x_val, y_val, cell_val = load_dataset(VAL_PATH)
    test_df, x_test, y_test, cell_test = load_dataset(TEST_PATH)

    return {
        "train": (train_df, x_train, y_train, cell_train),
        "val": (val_df, x_val, y_val, cell_val),
        "test": (test_df, x_test, y_test, cell_test),
    }