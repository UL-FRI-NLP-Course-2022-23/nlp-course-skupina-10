import argparse
import torch
import json
import os
from t5 import gen_paraphrase, load_model
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def plot_loss(model_path, save_path="./results"):
    """Plot the train and validation loss."""
    print("Plotting train and validation loss...", flush=True)
    trainer_state = json.load(open(os.path.join(model_path, "trainer_state.json"), "r"))
    # print(trainer_state["log_history"])

    def collect_loss(key="loss"):
        loss, steps = [], []
        for info in trainer_state["log_history"]:
            if key not in info:
                continue
            loss.append(info[key])
            steps.append(info["epoch"])
        return loss, steps
    
    train_loss, train_steps = collect_loss("loss")
    val_loss, val_steps = collect_loss("eval_loss")
    plt.plot(train_steps, train_loss, "o-", label="train loss")
    plt.plot(val_steps, val_loss, "o-", label="validation loss")
    plt.title("Train and validation loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"loss_{datetime.now()}.pdf"))


def val_predictions(model, tokenizer, val_file, save_path="./results"):
    """Generate paraphrases for the validation set."""
    print("Making predictions on the validation set...", flush=True)
    root_df = pd.read_csv(val_file, delimiter="\t")
    pairs = []
    for input_text, truth in zip(root_df["sentence1"], root_df["sentence2"]):
        pairs.append({
            "input_text": input_text,
            "truth": truth,
            "preds": gen_paraphrase(input_text, model, tokenizer)
        })
        break

    with open(f"{save_path}/val_predictions_{datetime.now()}.txt", "w") as f:
        for pair in pairs:
            f.write(pair["input_text"] + "\n\n")
            f.write("Truth:\n")
            f.write(pair["truth"] + "\n\n")
            f.write("Prediction:\n")
            for pred in pair["preds"]:
                f.write(pred + "\n")
            f.write("________________________________________________________________________________\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    MODEL_PATH = "/d/hpc/projects/FRI/DL/mm1706/nlp/t5-sl-large-para/checkpoint-10000/"
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    plot_loss(args.model_path)

    tokenizer, model, model_name = load_model()
    model.load_state_dict(torch.load(args.model_path + "/pytorch_model.bin"))
    val_file = os.path.join(os.getcwd(), "data/pairs-dev-t5.csv")
    val_predictions(model, tokenizer, val_file)
