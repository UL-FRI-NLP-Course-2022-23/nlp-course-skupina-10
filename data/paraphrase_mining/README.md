# Paraphrase Mining

The script creates a dataset called "Subtitle-Subtitle" or "SS" by extracting sentences from two subtitle files of the same movie, each written by a different provider. The script then calculates BERT embeddings (using https://huggingface.co/EMBEDDIA/crosloengual-bert) of each sentence and compares the cosine similarity to identify pairs of sentences. It is important to note that this approach is **not completely accurate**, and therefore it is recommended that matched sentences are **checked by a human**. Note that the BERT approach is not tied to the timing information of the subtitles, which means that it can be used to align subtitles that have been sampled at different frame rates or have different timecodes. 


## Dataset

Final dataset (checked by hand) is available in [Final SS-Dataset](./ss_dataset_annotated.csv)  