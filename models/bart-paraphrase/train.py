import logging
import os
from datetime import datetime

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel
from sklearn.model_selection import train_test_split
from utils import (CustomSimpleDataset, Seq2SeqArgsFix,
                   clean_unnecessary_spaces, load_data)

# torch.multiprocessing.set_sharing_strategy('file_system')


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

train_df = pd.read_csv("data/ml_combined.csv", sep="\t").astype(str)
eval_df = pd.read_csv("data/ml_combined_eval.csv", sep="\t").astype(str)

# Slovene data
train_df = pd.concat(
    [
        train_df,
        load_data(
            "data/pairs-train.csv", "sentence1", "sentence2", "label", src_lang="sl_SI"
        ),
    ]
)
eval_df = pd.concat(
    [
        eval_df,
        load_data(
            "data/pairs-dev.csv", "sentence1", "sentence2", "label", src_lang="sl_SI"
        ),
    ]
)

eval_df = load_data(
    "data/pairs-dev.csv", "sentence1", "sentence2", "label", src_lang="sl_SI"
)

train_df = train_df[["prefix", "input_text",
                     "target_text", "src_lang", "tgt_lang"]]
eval_df = eval_df[["prefix", "input_text",
                   "target_text", "src_lang", "tgt_lang"]]

train_df = train_df.dropna()
eval_df = eval_df.dropna()

train_df["input_text"] = train_df["input_text"].apply(clean_unnecessary_spaces)
train_df["target_text"] = train_df["target_text"].apply(
    clean_unnecessary_spaces)

eval_df["input_text"] = eval_df["input_text"].apply(clean_unnecessary_spaces)
eval_df["target_text"] = eval_df["target_text"].apply(clean_unnecessary_spaces)

print(train_df)

model_args = Seq2SeqArgsFix()
model_args.eval_batch_size = 64
model_args.evaluate_during_training = False
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = 5e-5
model_args.max_seq_length = 128
model_args.num_train_epochs = 2
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = False
model_args.use_cached_eval_features = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = 8
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False

model_args.do_sample = True
model_args.num_beams = 1
model_args.num_return_sequences = 3
model_args.max_length = 128
model_args.top_k = 50
model_args.top_p = 0.95
model_args.dataset_class = CustomSimpleDataset
model_args.tgt_lang = "sl_SI"
model_args.src_lang = "sl_SI"

# model_args.wandb_project = "Paraphrasing with BART"


model = Seq2SeqModel(
    encoder_decoder_type="mbart50",
    encoder_decoder_name="facebook/mbart-large-50",
    args=model_args,
    # use_cuda=False,
)

model.train_model(train_df, eval_data=eval_df)

to_predict = eval_df["input_text"].tolist()
truth = eval_df["target_text"].tolist()

preds = model.predict(to_predict)

# Saving the predictions if needed
os.makedirs("predictions", exist_ok=True)

with open(f"predictions/predictions_{datetime.now()}.txt", "w", encoding="utf-8") as f:
    for i, text in enumerate(eval_df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )
