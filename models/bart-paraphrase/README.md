# BART-Paraphrase Model

Based on <https://github.com/ThilinaRajapakse/simpletransformers/tree/master/examples/seq2seq/paraphrasing>

Before running, copy the following files to `./data/` folder:

-   `pairs-train.csv`, `pairs-dev.csv` from backtransaltion
-   `ml_combined.csv`, `ml_combined_eval.csv` from pqc

```sh
python3 train.py
```

## Evaluation

After training, the script will also generate predictions for the sentences from the `dev` / `eval` datasets (see `./eval_predictions.txt`).
