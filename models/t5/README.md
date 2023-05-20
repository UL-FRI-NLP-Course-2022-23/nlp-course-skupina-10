# T5-Model

## Training 

Base model: [cjvt/t5-sl-large](https://huggingface.co/cjvt/t5-sl-large). You can set the hyper-parameters for training in the `./hyper_params.json` file. 

Before running, please copy the following files to the `./data/` folder.
+ `pairs-train-aug.csv`, `pairs-dev.csv` from augmentation and backtransaltion
+ `ss-pairs-train.csv` from paraphrase_mining

```console
python ./train.py
```

## Evaluation

After training you can run `./analysis.py` which will plot the training and validation loss over time
as well as make predictions using `pairs-dev.csv`. Both the plotted figure and validation predictions are saved in the `./results` directory.
