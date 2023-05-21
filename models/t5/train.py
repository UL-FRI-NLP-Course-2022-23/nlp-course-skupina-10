import os
import json

import pandas as pd
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq

from t5 import load_model

tokenizer, model, model_name = load_model()


def collect_data(data_dir="./data"):
    """Collect data in ./data directory and put in the T5 format."""
    print(f"Collecting data in {data_dir}...", flush=True)

    df_train, df_val = pd.DataFrame(), pd.DataFrame()
    # iterate over all files in the data directory
    for file in os.listdir(data_dir):
        if not file.endswith(".csv") or "t5" in file:
            continue
        
        print(f"processing file: {file}")
        # read the annotations file and keep only the sentence1 and sentence2 columns
        df = pd.read_csv(os.path.join(data_dir, file), delimiter="\t")
        df_t5 = df[["sentence1", "sentence2"]]

        # add the data to the train or validation dataframe
        if "train" in file:
            df_train = pd.concat([df_train, df_t5], ignore_index=True)
        else:
            df_val = pd.concat([df_val, df_t5], ignore_index=True)

    # save the new dataframes to csv files
    df_train.to_csv(os.path.join(data_dir, "pairs-train-t5.csv"), sep="\t", index=False)
    df_val.to_csv(os.path.join(data_dir, "pairs-dev-t5.csv"), sep="\t", index=False)


def preprocess_function(examples):
    """Preprocess the data for the T5 model."""
    inputs = examples['sentence1']
    targets = examples['sentence2']
    inputs_encodings = tokenizer.batch_encode_plus(
        inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    targets_encodings = tokenizer.batch_encode_plus(
        targets, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return {
        'input_ids': inputs_encodings['input_ids'],
        'attention_mask': inputs_encodings['attention_mask'],
        'labels': targets_encodings['input_ids'],
    }


if __name__ == "__main__":
    # collect data in the ./data directory and put in the T5 format
    collect_data()

    # read the hyper-parameters
    params = json.load(open(os.path.join(os.getcwd(), "hyper_params.json"), "r"))
    
    # read and preprocess the data
    train_file = os.path.join(os.getcwd(), "data/pairs-train-t5.csv")
    val_file = os.path.join(os.getcwd(), "data/pairs-dev-t5.csv")
    dataset = load_dataset("csv", data_files={'train': train_file,'validation': val_file}, delimiter='\t')
    dataset = dataset.map(preprocess_function, batched=True)

    # train the model
    training_args = TrainingArguments(
        output_dir=f"/d/hpc/projects/FRI/DL/mm1706/nlp/{model_name}-para",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params["learning_rate"], 
        num_train_epochs=params["n_epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        save_total_limit=params["save_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
