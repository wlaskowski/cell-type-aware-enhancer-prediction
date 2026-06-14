"""
Module for preprocessing the raw data files.
The script:
- loads DNA sequences from a FASTA file
- loads activity data from TSV files for three cell types
- matches every regulatory element with its DNA sequence
- averages replicate activity measurements
- splits regulatory elements into train/val/test sets
- saves the processed datasets
"""

import pandas as pd

from pathlib import Path
from Bio import SeqIO

from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FASTA = RAW_DIR / "ENCFF795PSI.fasta"

FILES = {
    "HepG2": RAW_DIR / "ENCFF009HVM_HepG2.tsv", 
    "K562": RAW_DIR / "ENCFF068BWG_K562.tsv", 
    "WTC11": RAW_DIR / "ENCFF357GSY_WTC11.tsv"
     }


def main():
    """Running the complete preprocessing pipeline"""
    
    # verify that all input files are present
    missing = [str(p) for p in [FASTA, *FILES.values()] if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n" + "\n".join(missing)
        )

    # dictionary mapping regulatory el. IDs to their DNA seqs
    id_to_seq = {}

    for record in SeqIO.parse(str(FASTA), "fasta"):
        id_to_seq[record.id] = str(record.seq)

    #loading actiity data 
    df_hepg2 = pd.read_csv(FILES["HepG2"], sep="\t")
    df_k562 = pd.read_csv(FILES["K562"], sep="\t")
    df_wtc11 = pd.read_csv(FILES["WTC11"], sep="\t")

    # matching MPRA elemnt IDs to their sequences
    df_hepg2["sequence"] = df_hepg2["name"].map(id_to_seq)
    df_k562["sequence"] = df_k562["name"].map(id_to_seq)
    df_wtc11["sequence"] = df_wtc11["name"].map(id_to_seq)
    
    # adding cell type label
    df_hepg2["cell_type"] = "HepG2"
    df_k562["cell_type"] = "K562"
    df_wtc11["cell_type"] = "WTC11"

    # combining measurements from all cell types into a final dataset
    df = pd.concat([df_hepg2, df_k562, df_wtc11], ignore_index=True)

    # converting activity measurements to numeric vals
    df["log2"] = pd.to_numeric(df["log2"], errors="coerce")

    # average replicate measurements for each element and cell-type combination
    df_processed = (
        df.groupby(["name", "cell_type", "sequence"], as_index=False)
        .agg(activity=("log2", "mean"))
    )

    # remove rows with missing sequences or activities
    df_processed = df_processed.dropna(subset=["sequence", "activity"])

    unique_names = df_processed["name"].unique()

    # spliting into train/val/test sets based on unique element IDs to prevent data leakage
    train_names, temp_names = train_test_split(
        unique_names,
        test_size=0.30,
        random_state=42
    )

    val_names, test_names = train_test_split(
        temp_names,
        test_size=0.50,
        random_state=42
    )

    # selecting rows for each set
    train_df = df_processed[df_processed["name"].isin(train_names)]
    val_df = df_processed[df_processed["name"].isin(val_names)]
    test_df = df_processed[df_processed["name"].isin(test_names)]

    # saving final, processed datasets
    df_processed.to_csv(OUT_DIR / "data.csv", index=False)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    #print(df_processed.head())
    #print(df_processed["cell_type"].value_counts())

if __name__ == "__main__":
    main()
