import os
import pandas as pd
import argparse
import numpy as np


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

def _replace_synonyms(org_data_dir, synonym_data_dir):
    """Replace synonyms in the input dataset.
    NOTE that synonyms are replaced by hand, we only read the file here!"""
    df = pd.read_csv(org_data_dir, delimiter="\t")
    df_s = pd.read_csv(synonym_data_dir, delimiter="\t")
    return df_s[(df["sentence1"] + " " + df["sentence2"]) != (df_s["sentence1"] + " " + df_s["sentence2"])]


def paraphrase_augmentation(root_dir, file, add_duplicates=False):
    """Paraphrase dataset augmentation."""
    data_dir = os.path.join(root_dir, file)
    df_swap = _swap_paraphrases(data_dir)  # original df + swapped df    
    df_syn = _replace_synonyms(data_dir, os.path.join(os.getcwd(), "pairs-train-synonyms-aug.csv"))
    print(f"added {df_syn.shape[0]} parapharases with replaced synonyms")
    df_all = [df_swap, df_syn]
    if add_duplicates:
        df_dup = _add_duplicates(data_dir)  # duplicates of original df
        df_all.append(df_dup)
    df_aug = pd.concat(df_all, ignore_index=True)
    # word_cnt1 = df_aug.sentence1.apply(lambda x: len(x.split()))
    # word_cnt2 = df_aug.sentence2.apply(lambda x: len(x.split()))
    # print(f"longest sentence1 [words] {word_cnt1.max(axis=0)}")
    # print(f"longest sentence2 [words] {word_cnt2.max(axis=0)}")
    df_aug.to_csv(os.path.join(os.getcwd(), f"{file.split('.')[0]}-aug.csv"), 
                  index=False, sep="\t")

'''
# NOTE: This introduces a lot of noise, so we don't use it!
import wn
from wn.similarity import path
from lemmagen3 import Lemmatizer
lem_sl = Lemmatizer("sl")  # init slovenian lemmatizer
wn.download("omw-sl:1.4")  # download slovenian wordnet
swn = wn.Wordnet("omw-sl:1.4")  # init slovenian wordnet
def _find_closest_synonym(word_in, sim_measure=path, sim_thresh=0.5):
    """Find the closest synonym of the input word."""
    lemma_in = lem_sl.lemmatize(word_in)
    synsets = swn.synsets(lemma_in)
    if len(synsets) == 0:
        # print(f"Could not find synsets for {word_in}")
        return None, None
    # collect synonyms
    synonyms = []
    for w in synsets[0].words():
        if w.lemma().lower() != lemma_in.lower():
            synonyms.append(w.lemma())
    # find the closest synonym
    s = swn.synsets(lemma_in)[0]
    max_sim = 0
    best_synonym = None
    for si in synonyms:
        sim = sim_measure(s, swn.synsets(si)[0], simulate_root=True)
        if sim > max_sim and sim > sim_thresh:
            max_sim = sim
            best_synonym = si
        # print(f"Similarity between '{lemma_in}' and '{si}': {sim}")
    return best_synonym, max_sim


def _sen_synonyms(sen):
    """Find all synonyms in the input sentence."""
    for word in sen.split(" "):
        # lemmatize the current word
        lemma = lem_sl.lemmatize(word)  # we wan't to keep the same case!
        if lemma != word:
            continue

        new_word, sim = _find_closest_synonym(word)
        if new_word is None:
            continue
        print(word, new_word, sim)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../backtranslation")
    parser.add_argument("--file", type=str, default="pairs-train.csv")
    parser.add_argument("--add_duplicates", type=bool, default=False)
    args = parser.parse_args()
    paraphrase_augmentation(args.root_dir, args.file)
