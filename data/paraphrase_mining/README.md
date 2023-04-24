# Construct SS Dataset

The script creates a dataset called "Subtitle-Subtitle" or "SS" dataset, which involves extracting sentences from two subtitle files of the same movie, each written by a different provider. The script then calculates BERT embeddings (using https://huggingface.co/EMBEDDIA/crosloengual-bert) of each sentence and compares the cosine similarity to identify pairs of sentences. It is important to note that this approach is **not completely accurate**, and therefore it is recommended that matched sentences are **checked by a human**.

## Usage

1. Add a pair of subtitles to the `./subtitles` directory, you should name them as follows: `<movie_name>1.srt` and `<movie_name>2.srt`.
2. Run script `python3 construct_dataset.py` which constructs a dataset using all subtitle pairs in the `./subtitles` directory.
