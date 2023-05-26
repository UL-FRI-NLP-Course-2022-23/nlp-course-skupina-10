import numpy as np
import pandas as pd


def split_data(dataset_path, split_size=200, n_splits=3, delimiter=";"):
    """Split shuffled dataset into `n_splits` parts of size `split_size`."""
    df = pd.read_csv(dataset_path, delimiter=delimiter)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle dataset
    return [df.iloc[i * split_size: (i + 1) * split_size, 0:2] for i in range(n_splits)]


if __name__ == "__main__":
    # split the paraphrase mining dataset into 3 parts of size 300
    data = split_data("../paraphrase_mining/ss_dataset_annotated.csv", delimiter=";")
    new_columns = ["accuracy", "fluency", "diversity"]
    for i in range(len(data)):
        data[i] = data[i].reindex(columns=[*data[i].columns.tolist(), *new_columns])
        data[i].to_csv(f"../evaluation/data_subsets/paraphrase_mining_subset{i}.csv", index=False, sep=";")

    # split the backtranslation dataset into 3 parts of size 300
    data = split_data("../backtranslation/pairs-train.csv", delimiter="\t")
    for i in range(len(data)):
        data[i] = data[i].reindex(columns=[*data[i].columns.tolist(), *new_columns])
        data[i].to_csv(f"../evaluation/data_subsets/backtranslation_subset{i}.csv", index=False, sep=";")
