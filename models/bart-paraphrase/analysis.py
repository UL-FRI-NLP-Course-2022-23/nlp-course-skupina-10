import logging
import os
from datetime import datetime

import pandas as pd
from parascore import ParaScorer
from simpletransformers.seq2seq import Seq2SeqModel
from utils import load_data2

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

MODEL_TYPE = "facebook/mbart-large-50"

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
