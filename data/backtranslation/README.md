# Backtranslation dataset

## Data preparation

### Using data from CLARIN

1.  Download dataset in `CoNLL-U` fromat from <http://hdl.handle.net/11356/1747> and extract the contents into the `./suk` folder
2.  Run `python .\suk-sentences.py --out sentences.txt` to extract sentences from the dataset

### Using custom data

1.  Prepare a `.txt` file containing one sentence per line

## Generate backtranslated dataset

1.  Get a (free) DeepL API key from <https://www.deepl.com/pro-api>
2.  Run `python ./gen-data.py ./sentences.txt -o sentences.csv -k [API key]`
