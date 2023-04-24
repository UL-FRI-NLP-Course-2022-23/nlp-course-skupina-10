import srt
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
from scipy.spatial.distance import cdist
import pandas as pd


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
def extract_embeddings(s, tokenizer, model, batch_size=32):
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


def construct_dataset(f_name1, f_name2, topk=1):
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

    print("Computing cosine similarities...")
    sim = cdist(s1_emb.numpy(), s2_emb.numpy(), metric='cosine')
    idxs1 = np.arange(len(s1)).astype(int)
    idxs2 = sim.argsort(axis=1)[:, :topk]
    print(idxs1.shape, idxs2.shape)

    print("Saving dataset...")
    data = {"sentence1": np.array(s1)[idxs1]}
    for i in range(topk):
        data[f"setnence{i}"] = np.array(s2)[idxs2[:, i]]

    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv("dataset.csv", index=False, sep=";")


if __name__ == "__main__":
    f_name1 = "./subtitles/avatar1.srt"
    f_name2 = "./subtitles/Avatar.2009.720p.BluRay.x264-TDM.srt"
    construct_dataset(f_name1, f_name2, topk=3)
