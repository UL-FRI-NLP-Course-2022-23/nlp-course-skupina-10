import os
import pandas as pd


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    dataset_dir = "./ss_dataset_filtered.csv"
    instruction = "Good match (1) or not (2)" # "Select best match sentence (1, 2 or 3) or delete (4):"
    df = pd.read_csv(dataset_dir, sep=";")
    clear_console()

    dataset = []
    idx_start = 0
    for idx in range(idx_start, len(df)):
        candidate = df.iloc[idx, :]
        print(f"sample: {idx}")
        for i, sen in enumerate(candidate):
            print(f"({i}): {sen}")

        sel_idx = int(input(instruction))
        clear_console()
        if sel_idx == 2:
            continue
        dataset.append([candidate.to_numpy()[0], candidate.to_numpy()[sel_idx]])

        if idx != 0 and idx % 10 == 0:
            pd.DataFrame(dataset).to_csv(
                "./ss_dataset_annotated.csv", mode="a", index=False, header=False, sep=";"
            )
            dataset = []
    
