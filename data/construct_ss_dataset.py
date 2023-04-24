import srt
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
from scipy.spatial.distance import cdist
import pandas as pd
import os


def end_of_sentence(sen, stop=('...', '.', '?', '!', '!!!')):
    """Check if sentence ends with a stop character."""
    return sen.endswith(stop)


def extract_sentences(file):
    """Extract sentences from subtitle file."""
    sentances = []
    sub_composed = ""
    for sub in srt.parse(file):
        sub_ = sub.content.replace("\n", " ").lstrip()
        sub_ = sub_.replace("<i>", "").replace("</i>", "")

        sub_composed += " " + sub_
        if end_of_sentence(sub_composed):
            sentances.append(sub_composed)
            sub_composed = ""

    return sentances


def save_sentences(s, f_name):
    """Save sentences to file."""
    with open(f_name, "w") as file:
        for sen in s:
            file.write(sen + "\n")


@torch.no_grad()
def extract_embeddings(s, tokenizer, model, batch_size=64):
    """Extract embeddings from sentences."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the sentences
    input_ids = []
    attention_masks = []
    for sentence in s:
        encoded_dict = tokenizer.encode_plus(
            sentence, add_special_tokens=True, padding='max_length', max_length=64,
            truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    # Extract embeddings in batches
    embeddings = []
    for idx in range(0, len(s), batch_size):
        batch_input_ids = input_ids[idx:idx+batch_size]
        batch_attention_masks = attention_masks[idx:idx+batch_size]
        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        batch_embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)
    return torch.tensor(embeddings)


def count_common_words(s1, s2):
    """Count common words between two sentences."""
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2))


def filter_sentences(df, heuristic=count_common_words):
    """Filter sentences using heuristic."""
    data_filtered = []
    for i in range(df.shape[0]):
        s = df.iloc[i, 0]
        cnt_max = 0
        idx = 0

        for j in range(1, df.shape[1]):
            cnt = heuristic(s, df.iloc[i, j])
            if cnt > cnt_max:
                cnt_max = cnt
                idx = j

        if s.lower() == df.iloc[i, idx].lower():
            continue

        data_filtered.append([s, df.iloc[i, idx]])

    return data_filtered


def construct_dataset(f_name1, f_name2, topk=1, append=False):
    """Construct dataset by extracting sentences from two subtitle files
    and matching them using the crosloengual-bert model embeddings."""

    f1 = open(f_name1, encoding="Windows-1252").read()
    f2 = open(f_name2, encoding="Windows-1252").read()
    f1 = f1.replace("è", "č").replace("È", "Č")
    f2 = f2.replace("è", "č").replace("È", "Č")
    s1 = extract_sentences(f1)
    s2 = extract_sentences(f2)
    # save_sentences(s1, f_name="./s1.txt")
    # save_sentences(s2, f_name="./s2.txt")

    print("Loading model...")
    # https://huggingface.co/EMBEDDIA/crosloengual-bert
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
    model = AutoModel.from_pretrained("EMBEDDIA/crosloengual-bert")

    # Tokenize the sentences
    print("Extracting embeddings...")
    s1_emb = extract_embeddings(s1, tokenizer, model)
    s2_emb = extract_embeddings(s2, tokenizer, model)

    print("Computing cosine distances...")
    sim = cdist(s1_emb.numpy(), s2_emb.numpy(), metric='cosine')
    idxs1 = np.arange(len(s1)).astype(int)
    idxs2 = sim.argsort(axis=1)[:, :topk]
    print(idxs1.shape, idxs2.shape)

    print("Saving dataset...")
    data = {"sentence0": np.array(s1)[idxs1]}
    for i in range(topk):
        data[f"sentence{i + 1}"] = np.array(s2)[idxs2[:, i]]

    df = pd.DataFrame(data)
    print(df.head())
    mode = "a" if append else "w"
    df.to_csv("ss_dataset.csv", mode=mode, index=False, sep=";")

    data_filtered = filter_sentences(df)
    df = pd.DataFrame(data_filtered, columns=["sentence1", "sentence2"])
    print(df.head())
    df.to_csv("ss_dataset_filtered.csv", mode=mode, index=False, sep=";")


if __name__ == "__main__":
    # iterate over all subtitle files in the subtitles directory
    subtitles_dir = "./subtitles/"
    subtitles = {}
    for filename in os.scandir(subtitles_dir):
        if filename.is_file() and filename.path.endswith(".srt"):
            f_name = filename.path.split("/")[-1].split(".")[0]
            sub, num = f_name[:-1], int(f_name[-1])
            if sub not in subtitles:
                subtitles[sub] = []
            subtitles[sub].append(filename.path)

    # construct dataset for each pair of subtitles
    for f_name1, f_name2 in subtitles.values():
        print(f_name1, f_name2)
        construct_dataset(f_name1, f_name2, topk=3, append=True)
