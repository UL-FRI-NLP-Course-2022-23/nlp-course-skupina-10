# BART-Paraphrase Model

Based on <https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing>

Before running, copy the following files to `./data/` folder:

-   `pairs-train.csv`, `pairs-dev.csv` from backtransaltion
-   `ml_combined.csv`, `ml_combined_eval.csv` from pqc

```sh
python ./train.py
```

## Evaluation

After training, the script will also generate predictions for the sentences from the `dev` / `eval` datasets (see `./eval_predictions.txt`).

## Trained model

A trained model checkpoint can be downloaded from the [releases page](https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-skupina-10/releases). Extract the `outputs` directory from the downloaded zip, then run `python ./predict.py` to see it in action (note even though this is a cross-lingual model, the script only generates paraphrases for Slovenian sentences; for other uses, the appropriate `model_args.src_lang` and `model_args.tgt_lang` need to be specified).
