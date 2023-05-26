import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bertviz import model_view
from parascore import ParaScorer
from parascore.utils import model2layers

from t5 import gen_paraphrase, load_model


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
    print(trainer_state["log_history"])

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


def test_model(model, tokenizer, scorer, test_file, save_path="./results"):
    """Evaluate the model on the test set using the ParaScore metric."""
    print("Testing the model...", flush=True)
    test_df = pd.read_csv(test_file, delimiter="\t")
    pairs = []

    for in_text, gt_text in zip(test_df["sentence1"], test_df["sentence2"]):
        # make predictions
        preds = gen_paraphrase(in_text, model, tokenizer)

        # evaluate predictions
        for pred in preds:
            score = scorer.base_score([pred], [in_text], [gt_text])[0]  # ref-based parascore
            score_free = scorer.free_score([pred], [in_text])[0].item() # ref-free parascore
            pairs.append([in_text, gt_text, pred, score, score_free])

    # save the predictions to a csv file
    preds_df = pd.DataFrame(
        pairs, columns=["in_text", "truth", "pred", "parascore-ref-based", "parascore-ref-free"])
    preds_df.to_csv(
        os.path.join(save_path, f"predictions_{datetime.now()}.csv"), sep="\t", index=False)


def test_predictions(scorer, test_file, save_path="./results"):
    """Evaluate the predictions using the ParaScore metric."""
    print("Computing parascore...", flush=True)
    df = pd.read_csv(test_file, delimiter="\t")
    pairs = []

    for in_text, gt_text, pred in zip(df["sentence1"], df["sentence2"], df["pred"]):
        score = scorer.base_score([pred], [in_text], [gt_text])[0]  # ref-based parascore
        score_free = scorer.free_score([pred], [in_text])[0].item() # ref-free parascore
        pairs.append([in_text, gt_text, pred, score, score_free])

    # save the predictions to a csv file
    preds_df = pd.DataFrame(
        pairs, columns=["in_text", "truth", "pred", "parascore-ref-based", "parascore-ref-free"])
    preds_df.to_csv(
        os.path.join(save_path, f"predictions_{datetime.now()}.csv"), sep="\t", index=False)


@torch.no_grad()
def viz_attention(model, tokenizer, in_text, gt_text, save_path="./results"):
    """Visualize the attention between words in the source sentence and in the target sentence."""
    print("Visualizing attention weights...", flush=True)

    # get encoded input vectors
    encoder_input_ids = tokenizer(in_text, return_tensors="pt", add_special_tokens=False).input_ids
    
    # create ids of encoded input vectors
    with tokenizer.as_target_tokenizer():
        decoder_input_ids = tokenizer(gt_text, return_tensors="pt", add_special_tokens=False).input_ids

    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
    encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
    decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

    # plot attention for all layers and all heads
    attn_maps = model_view(
        encoder_attention=outputs.encoder_attentions,
        decoder_attention=outputs.decoder_attentions,
        cross_attention=outputs.cross_attentions,
        encoder_tokens= encoder_text,
        decoder_tokens=decoder_text,
        display_mode="light",
        html_action="return"

    )
    # save the attention maps to a html file
    with open(os.path.join(save_path, f"attn_maps_{datetime.now()}.html"), "w") as file:
        file.write(attn_maps.data)


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    MODEL_PATH = "/d/hpc/projects/FRI/DL/mm1706/nlp/t5-sl-large-para/final-aug/"
    print(f"Using model from {MODEL_PATH}")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    
    # plot the loss over time
    print(args.model_path)
    # plot_loss(args.model_path)

    # make predicitons on the validation set and save them
    tokenizer, model, model_name = load_model()
    model.load_state_dict(torch.load(args.model_path + "/pytorch_model.bin"))
    test_file = os.path.join(os.getcwd(), "data/pairs-test-t5.csv")
    scorer = ParaScorer(model_type=MODEL_TYPE)
    # test_model(model, tokenizer, scorer, test_file)

    # load model with attention
    tokenizer, model, model_name = load_model(output_attentions=True)
    model.load_state_dict(torch.load(args.model_path + "/pytorch_model.bin"))

    # visualize attention weights
    in_text = "Cilj te študije je bil identificirati korelacije med prehrano in duševnim zdravjem mladostnikov."
    gt_text = "Glavni namen te raziskave je bil ugotoviti povezave med prehrano in psihičnim blagostanjem pri mladostnikih."

    in_text = "Vlada je sprejela ukrepe za spodbujanje trajnostnega razvoja in zmanjšanje ogljičnega odtisa."
    gt_text = "Vodstvo je implementiralo strategije s ciljem spodbujanja trajnostnega napredka ter zmanjšanja negativnega vpliva na okolje v obliki ogljičnega odtisa."

    print(f"in_text: {in_text}")
    print(f"gt_text: {gt_text}")
    viz_attention(model, tokenizer, in_text, gt_text)
    """

    # evaluate the predictions
    PRED_PATH = "./results/predictions_base.csv"
    test_file = os.path.join(os.getcwd(), PRED_PATH)
    scorer = ParaScorer(model_type=MODEL_TYPE)
    test_predictions(scorer, test_file, save_path="./extra-results")
