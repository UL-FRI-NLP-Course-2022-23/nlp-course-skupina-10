import argparse
import torch
import json
import os
from t5 import gen_paraphrase, load_model
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from parascore import ParaScorer
from parascore.utils import model2layers
import numpy as np


#######################################################
# NOTE: Since the parascore library doesn't support the `crosloengual-bert`
# (https://arxiv.org/pdf/2006.07890.pdf) we have to add it manually.
def add_contextual_emb_model(model_type, n_layers):
    """Add a custom contextual embedding model to the parascore library."""
    global model2layers
    model2layers[model_type] = n_layers

MODEL_TYPE = "EMBEDDIA/crosloengual-bert" 
MODEL_LAYERS = 12
add_contextual_emb_model(MODEL_TYPE, MODEL_LAYERS)
#######################################################


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
    val_df = pd.read_csv(val_file, delimiter="\t")
    pairs = []

    # iterate over the validation set and save predictions to save_path
    for input_text, truth in zip(val_df["sentence1"], val_df["sentence2"]):
        pairs.append({
            "input_text": input_text,
            "truth": truth,
            "preds": gen_paraphrase(input_text, model, tokenizer)
        })
        
    with open(f"{save_path}/val_predictions_{datetime.now()}.txt", "w") as f:
        for pair in pairs:
            f.write(pair["input_text"] + "\n\n")
            f.write("Truth:\n")
            f.write(pair["truth"] + "\n\n")
            f.write("Prediction:\n")
            for pred in pair["preds"]:
                f.write(pred + "\n")
            f.write("________________________________________________________________________________\n")


def test_model(model, tokenizer, scorer, test_file, save_path="./results"):
    """Test the model with the ParaScore metric."""
    print("Testing the model...", flush=True)
    para_s = {"ref-free": [], "ref-based": []}
    test_df = pd.read_csv(test_file, delimiter="\t")

    # iterate over the test set
    for in_text, truth in zip(test_df["sentence1"], test_df["sentence2"]):
        preds = gen_paraphrase(in_text, model, tokenizer)

        # compute ref-based and ref-free score for each prediction
        for pred in preds:
            score = scorer.base_score([pred], [in_text], [truth])[0]
            score_free = scorer.free_score([pred], [in_text])[0].item()
            para_s["ref-based"].append(score)
            para_s["ref-free"].append(score_free)

    # print the results
    mean_ref_based = np.mean(para_s["ref-based"])
    std_ref_based = np.std(para_s["ref-based"])
    mean_ref_free = np.mean(para_s["ref-free"])
    std_ref_free = np.std(para_s["ref-free"])
    print(f"Mean ref-based score: {mean_ref_based} +/- {std_ref_based}")
    print(f"Mean ref-free score: {mean_ref_free} +/- {std_ref_free}")

    # save the results
    metrics_df = pd.DataFrame(para_s)
    metrics_df.to_csv(
        os.path.join(save_path, f"metrics_{datetime.now()}.csv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    MODEL_PATH = "/d/hpc/projects/FRI/DL/mm1706/nlp/t5-sl-large-para/checkpoint-10000/"
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    
    # plot the loss over time
    # plot_loss(args.model_path)

    # make predicitons on the validation set and save them
    tokenizer, model, model_name = load_model()
    model.load_state_dict(torch.load(args.model_path + "/pytorch_model.bin"))
    val_file = os.path.join(os.getcwd(), "data/pairs-dev-t5.csv")
    # val_predictions(model, tokenizer, val_file)

    # init the parascorer and test the model
    scorer = ParaScorer(model_type=MODEL_TYPE)
    # test_model(model, tokenizer, scorer, val_file)
