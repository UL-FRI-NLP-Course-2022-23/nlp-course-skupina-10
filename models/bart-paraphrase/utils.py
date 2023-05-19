import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass
from multiprocessing import Pool

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqArgs
from simpletransformers.seq2seq.seq2seq_utils import (preprocess_data_bart,
                                                      preprocess_data_mbart)
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_data(
    file_path, input_text_column, target_text_column, label_column, keep_label=1, src_lang='en_XX', tgt_lang=None
):
    if tgt_lang is None:
        tgt_lang = src_lang
    df = pd.read_csv(file_path, sep="\t", on_bad_lines='skip')
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(
        columns={input_text_column: "input_text",
                 target_text_column: "target_text"}
    )
    df = df[["input_text", "target_text"]]
    df["prefix"] = "paraphrase"
    df["src_lang"] = src_lang
    df["tgt_lang"] = tgt_lang

    return df


def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


class CustomSimpleDataset(Dataset):
    def __init__(self, tokenizer, _, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name + "_cached_" +
            str(args.max_seq_length) + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s",
                        cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(
                " Creating features from dataset file at %s", args.cache_dir)

            l = {}

            def get_args(src_lang, tgt_lang):
                k = '{}_{}'.format(src_lang, tgt_lang)
                if k in l:
                    return l[k]
                d = args.get_args_for_saving()
                d.update(tgt_lang=tgt_lang, src_lang=src_lang)
                d = Seq2SeqArgsFix(**d)
                l[k] = d
                return d

            data = [
                (input_text, target_text, tokenizer, get_args(src_lang, tgt_lang))
                for input_text, target_text, src_lang, tgt_lang in zip(
                    data["input_text"], data["target_text"], data["src_lang"], data["tgt_lang"]
                )
            ]

            preprocess_fn = (
                preprocess_data_mbart
                if args.model_type == "mbart50"
                else preprocess_data_bart
            )

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                if args.multiprocessing_chunksize == -1:
                    chunksize = max(len(data) // (args.process_count * 2), 500)
                else:
                    chunksize = args.multiprocessing_chunksize

                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_fn, data, chunksize=chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [
                    preprocess_fn(d) for d in tqdm(data, disable=args.silent)
                ]

            os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass
class Seq2SeqArgsFix(Seq2SeqArgs):

    model_class: str = "Seq2SeqModelFix"

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["dataset_class"] is not None:
                args_dict["dataset_class"] = type(
                    args_dict["dataset_class"]).__name__
            json.dump(args_dict, f)
