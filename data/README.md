# Construct SS Dataset

The script creates a dataset called "Subtitle-Subtitle" or "SS" dataset, which involves extracting sentences from two subtitle files of the same movie, each written by a different provider. The script then calculates BERT embeddings of each sentence and compares the cosine similarity to identify pairs of sentences. It is important to note that this approach is not completely accurate, and therefore it is recommended that matched sentences be checked by a human.

## Usage

1. Download two sets of subtitles (https://www.podnapisi.net/).
2. Run script `python3 construct_dataset.py`.
