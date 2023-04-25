import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
from scipy.spatial.distance import cdist
import pandas as pd
import os
from utils import load_sentences
import re
import nltk
from nltk.tokenize import word_tokenize
# Download the Slovenian data package for nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('stopwords')

def count_common_words(s1, s2):
    """Count common words between two sentences."""
    s1 = set(s1.split())
    s2 = set(s2.split())
    return len(s1.intersection(s2))


def remove_chars(sen, chars=("...", ",", ".", "?", "!", "!!!", "")):
    """Remove special characters from sentences."""
    for c in chars:
        sen = sen.replace(c, "")
    return sen


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

        toks1 = word_tokenize(re.sub(r'[^\w\s]', '', s).lower(), language="slovene")
        toks2 = word_tokenize(re.sub(r'[^\w\s]', '', df.iloc[i, idx]).lower(), language="slovene")
        if toks1 == toks2:
            continue
        data_filtered.append([s, df.iloc[i, idx]])

    return data_filtered


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


def paraphrase_mining(sen1, sen2, topk=1, append=False):
    """Construct paraphrase mining dataset, by comparing sentences from two files."""

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
    model = AutoModel.from_pretrained("EMBEDDIA/crosloengual-bert")

    print("Extracting embeddings...")
    s1_emb = extract_embeddings(sen1, tokenizer, model)
    s2_emb = extract_embeddings(sen2, tokenizer, model)

    print("Computing cosine distances...")
    sim = cdist(s1_emb.numpy(), s2_emb.numpy(), metric='cosine')
    idxs1 = np.arange(len(sen1)).astype(int)
    idxs2 = sim.argsort(axis=1)[:, :topk]
    print(idxs1.shape, idxs2.shape)

    print("Saving dataset...")
    data = {"sentence0": np.array(sen1)[idxs1]}
    for i in range(topk):
        data[f"sentence{i + 1}"] = np.array(sen2)[idxs2[:, i]]

    df = pd.DataFrame(data)
    df_filtered = pd.DataFrame(filter_sentences(
        df), columns=["sentence1", "sentence2"])

    return df, df_filtered


if __name__ == "__main__":
    sen_dir = "./sentences/"
    sentences = {}
    for filename in os.scandir(sen_dir):
        if not filename.is_file():
            continue

        f_name = filename.path.split("/")[-1].split(".")[0]
        root = f_name.split("_")[0][:-1]
        if root not in sentences:
            sentences[root] = []
        sentences[root].append(filename.path)

    # construct dataset for each pair of subtitles
    for sen1, sen2 in sentences.values():
        df, df_filtered = paraphrase_mining(load_sentences(
            sen1), load_sentences(sen2), topk=3, append=True)

        df.to_csv("ss_dataset.csv", mode="a", index=False, sep=";", header=False)
        df_filtered.to_csv("ss_dataset_filtered.csv", mode="a", index=False, sep=";", header=False)
