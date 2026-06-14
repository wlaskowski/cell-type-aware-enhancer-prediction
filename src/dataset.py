"""
Load the processed data and convert it into numerical format
that can be used by neural networks.

"""


import numpy as np
import pandas as pd

from pathlib import Path


PROCESSED_DIR = Path("data/processed")

TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

# One-hot encoding for DNA bases
SEQ_MAPPING = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

# converting cell types to integer labels for futher use in the models
CELL_MAPPING = {
    "HepG2": 0,
    "K562": 1,
    "WTC11": 2,
}


def encode_sequence(seq):
    """
    converts a DNA sequence string into a one-hot encoded matrix
    """
    x = np.array(
        [SEQ_MAPPING.get(base, [0, 0, 0, 0]) for base in str(seq).upper().strip()],
        dtype=np.float32,
    )

    return x.T # needed for Conv1d

def encode_cell_type(cell_type):
    """
    converts a cell type string into an integer label.
    """

    return np.array(
        [CELL_MAPPING[c] for c in cell_type],
        dtype=np.int64,
    )


def load_dataset(path):
    """
    loads one processed dataset (train, val, test)
    """
    df = pd.read_csv(path, sep=",")

    # final shape: (number of samples, 4, sequence length)
    x = np.array([encode_sequence(seq) for seq in df["sequence"]])

    # target of the model, shape: (num,ber of samples,)
    y = df["activity"].values.astype(np.float32)

    # using a function for convertin cell types into integers
    cell_type = encode_cell_type(df["cell_type"])

    return df, x, y, cell_type


def load_train_val_test():
    """
    Loading and encoding all three datasets
    """
    train_df, x_train, y_train, cell_train = load_dataset(TRAIN_PATH)
    val_df, x_val, y_val, cell_val = load_dataset(VAL_PATH)
    test_df, x_test, y_test, cell_test = load_dataset(TEST_PATH)

    return {
        "train": (train_df, x_train, y_train, cell_train),
        "val": (val_df, x_val, y_val, cell_val),
        "test": (test_df, x_test, y_test, cell_test),
    }