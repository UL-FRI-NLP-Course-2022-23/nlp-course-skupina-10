import pandas as pd
from sklearn.model_selection import train_test_split
from utils import clean_unnecessary_spaces, load_data

# Google Data
train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/dev.tsv", sep="\t").astype(str)

train_df = train_df.loc[train_df["label"] == "1"]
eval_df = eval_df.loc[eval_df["label"] == "1"]

train_df = train_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)
eval_df = eval_df.rename(
    columns={"sentence1": "input_text", "sentence2": "target_text"}
)

train_df = train_df[["input_text", "target_text"]]
eval_df = eval_df[["input_text", "target_text"]]

train_df["prefix"] = "paraphrase"
eval_df["prefix"] = "paraphrase"
train_df["src_lang"] = "en_XX"
train_df["tgt_lang"] = "en_XX"
eval_df["src_lang"] = "en_XX"
eval_df["tgt_lang"] = "en_XX"

# MSRP Data
train_df = pd.concat(
    [
        train_df,
        load_data("data/msr_paraphrase_train.txt", "#1 String", "#2 String", "Quality"),
    ]
)
eval_df = pd.concat(
    [
        eval_df,
        load_data("data/msr_paraphrase_test.txt", "#1 String", "#2 String", "Quality"),
    ]
)

# Quora Data

# The Quora Dataset is not separated into train/test, so we do it manually the first time.
df = load_data(
    "data/quora_duplicate_questions.tsv", "question1", "question2", "is_duplicate"
)
q_train, q_test = train_test_split(df)

train_df = pd.concat([train_df, q_train])
eval_df = pd.concat([eval_df, q_test])

train_df = train_df[["prefix", "input_text", "target_text", "src_lang", "tgt_lang"]]
eval_df = eval_df[["prefix", "input_text", "target_text", "src_lang", "tgt_lang"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(clean_unnecessary_spaces)

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

train_df.to_csv("data/combined.csv", index=False, sep=";")
eval_df.to_csv("data/combined_eval.csv", index=False, sep=";")
