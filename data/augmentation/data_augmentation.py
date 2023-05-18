import os
import pandas as pd
import argparse
import numpy as np
import wn



def _swap_paraphrases(data_dir):
    """Generate an augmented dataset by swapping sentence1 and sentence2."""
    print("Swapping paraphrases...", flush=True)
    df = pd.read_csv(data_dir, delimiter="\t")
    df_swap = df.copy()
    df_swap["sentence1"] = df["sentence2"]
    df_swap["sentence2"] = df["sentence1"]
    return pd.concat([df, df_swap], ignore_index=True)


def _add_duplicates(data_dir):
    """Generate an augmented dataset by adding duplicates of the original dataset."""
    print("Adding duplicates...", flush=True)
    df = pd.read_csv(data_dir, delimiter="\t")
    df_dup = pd.DataFrame(columns=["sentence1", "sentence2", "label"])
    df_dup["sentence1"] = pd.concat([df["sentence1"], df["sentence2"]], axis=0)
    df_dup["sentence2"] = df_dup["sentence1"].copy()
    df_dup["label"] = 0
    assert (df_dup["sentence1"] == df_dup["sentence2"]).sum() == df_dup["sentence1"].shape[0], \
            "sentence1 and sentence2 are not equal!"
    return df_dup

def paraphrase_augmentation(root_dir, file):
    """Paraphrase dataset augmentation."""
    data_dir = os.path.join(root_dir, file)
    df_swap = _swap_paraphrases(data_dir)  # original df + swapped df
    df_dup = _add_duplicates(data_dir)  # duplicates of original df
    df_aug = pd.concat([df_swap, df_dup], ignore_index=True)
    df_aug.to_csv(os.path.join(root_dir, f"{file.split('.')[0]}-aug.csv"), index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../backtranslation")
    parser.add_argument("--file", type=str, default="pairs-train.csv")
    args = parser.parse_args()
    paraphrase_augmentation(args.root_dir, args.file)

    """
    # NOTE: This introduces a lot of noise, so it's not a good idea to use it.
    from lemmagen3 import Lemmatizer
    wn.download("omw-sl:1.4")  # download SloWNet
    lem_sl = Lemmatizer('sl')
    sen = "France Kmetič iz Motorevije pojasnjuje, da je bil v začetku junija sestanek med slovensko in avstrijsko delegacijo, narejena pa je bila tudi analiza, na podlagi katere temeljijo varnostni ukrepi, saj je pokazala, da varnostne naprave ne delujejo ustrezno."
    print(sen)
    map = {}
    for word in sen.split(" "):
        # lemmatize the current word
        lemma = lem_sl.lemmatize(word)
        if lemma != word:
            continue
        # print(f"{word} -> {lemma}")

        ss = wn.synsets(lemma)
        if len(ss) == 0: 
            continue
        
        words = ss[0].words()
        new_words = [w.lemma() for w in words if w.lemma().lower() != word.lower()]
        if len(new_words) == 0:
            continue

        # print([w.lemma() for w in words if w != word])
        map[word] = new_words

    key = np.random.choice(list(map.keys()))
    new_word = np.random.choice(map[key])
    print(f"Replacing {key} with {new_word}")
    sen = sen.replace(key, new_word)
    print(sen)
    """
