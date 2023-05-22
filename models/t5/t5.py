import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(output_attentions=False):
    """Load the pretrained T5-Model and Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("cjvt/t5-sl-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("cjvt/t5-sl-large", output_attentions=output_attentions)
    return tokenizer, model, "t5-sl-large"


def gen_paraphrase(input_text, model, tokenizer):
    """Generate paraphrases for the input text."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids, max_length=512, num_return_sequences=2, num_beams=4)
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


if __name__ == "__main__":
    MODEL_PATH = "/d/hpc/projects/FRI/DL/mm1706/nlp/t5-sl-large-para/v1/"
    INPUT_TEXT = "Zanima me, ali moram svetovalki zaposlitve prinesti potrdilo o zdravljenju med njegovim potekom ali pa lahko to opravim po končanem zdravljenju?"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--input_text", type=str, default=INPUT_TEXT)
    args = parser.parse_args()

    tokenizer, model, model_name = load_model()
    model.load_state_dict(torch.load(args.model_path + "/pytorch_model.bin"))
    gen_paraphrase(args.input_text, model, tokenizer)
