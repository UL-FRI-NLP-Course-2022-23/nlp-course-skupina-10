# Translated English dataset

This dataset is prepared by translating the data from PAWS, Quora, and MSR paraphrasing datasets.

## Data preparation

### Gather the original datasets

1.  Run `./data_download.sh` to download the datasets to the `data` forlder (MSR dataset has to be downloaded manually).
2.  Run `python ./prepare.py` to gather the data from downloaded datasets.

### Translate the data

Each data pair from the combined dataset is translated into Slovenian, then the following 4 training samples are generated:

-   `[en1, en2]`
-   `[sl1, sl2]`
-   `[en2, sl1]`
-   `[sl2, en1]`

Before translating the data, you need a running [NMT](https://github.com/clarinsi/Slovene_NMT) instance (see `translate.py` to set the correct endpoint). Then, run `python translate.py` to generate the translated pairs.
