# BART-Paraphrase Model

Based on <https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing>

Before training, copy the following files to `./data/` folder (for training using base and aug data from T5):

-   `pairs-train-t5-base.csv`, `pairs-train-t5-aug.csv`, `pairs-dev-t5.csv`, `pairs-test-t5.csv` from T5 model data

or (for multilingual training):

-   `pairs-train.csv`, `pairs-dev.csv` from backtransaltion
-   `ml_combined.csv`, `ml_combined_eval.csv` from pqc

```sh
python ./train.py # slo only
python ./train_ml.py # multilingual (slo + en)
```

## Evaluation

After training, the script will also generate predictions for the sentences from the `dev` / `eval` datasets (see `./eval_predictions.txt`).

## Trained model

Trained model checkpoints can be downloaded from the [releases page](https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-skupina-10/releases). Extract one of the `outputs` directories from the downloaded zip, rename it to `outputs`, then run `python ./predict.py` to see it in action (note even though this is a cross-lingual model, the script only generates paraphrases for Slovenian sentences; for other uses, the appropriate `model_args.src_lang` and `model_args.tgt_lang` need to be specified).
