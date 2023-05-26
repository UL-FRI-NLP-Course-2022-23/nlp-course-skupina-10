import logging
import os

import pandas as pd
from parascore import ParaScorer
from parascore.utils import model2layers
from simpletransformers.seq2seq import Seq2SeqModel
from utils import load_data2

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

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

model = Seq2SeqModel(
    encoder_decoder_type="mbart50",
    encoder_decoder_name="./checkpoint-276136-epoch-2",
    # use_cuda=False
)


eval_df = load_data2("data/pairs-test-t5.csv", "sentence1",
                     "sentence2", src_lang="sl_SI")

to_predict = eval_df["input_text"].tolist()
truth = eval_df["target_text"].tolist()

pairs = []
preds = model.predict(to_predict)

save_path = "./results"


scorer = ParaScorer(model_type=MODEL_TYPE)

for i, text in enumerate(eval_df["input_text"].tolist()):
    in_text = text
    gt_text = truth[i]
    for pred in preds[i]:
        score = scorer.base_score([pred], [in_text], [gt_text])[
            0]  # ref-based parascore
        score_free = scorer.free_score([pred], [in_text])[
            0].item()  # ref-free parascore
        pairs.append([in_text, gt_text, pred, score, score_free])


# save the predictions to a csv file
preds_df = pd.DataFrame(
    pairs, columns=["in_text", "truth", "pred", "parascore-ref-based", "parascore-ref-free"])
preds_df.to_csv(
    os.path.join(save_path, f"predictions_1.csv"), sep="\t", index=False)
