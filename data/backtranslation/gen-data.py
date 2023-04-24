import argparse
import csv
import os

import requests
import spacy
from nltk.tokenize import sent_tokenize
import re

endpoint = 'https://api.deepl.com/v2/translate'


def translate(
    text,
    to,
        language='sl',
        backtransleate=False,
        key=os.environ.get('TRANSLATOR_TEXT_SUBSCRIPTION_KEY', None)
):
    if not isinstance(to, list):
        to = [to]

    chain = [language] + to
    chains = []

    if backtransleate:
        for i in range(2, len(chain) + 1):
            r = chain[0:i-1]
            r.reverse()
            c2 = chain[0:i] + r
            chains.append(list(zip(c2[0:-1], c2[1:])))
    else:
        chains.append(list(zip(chain[0:-1], chain[1:])))

    headers = {
        'Authorization': 'DeepL-Auth-Key {}'.format(key)
    }

    if not isinstance(text, list):
        text = [text]

    input = text
    out = []

    mp = {}

    for c in chains:
        for j, p in enumerate(c):
            f, t = p
            k = '{}_{}_{}'.format(f, t, j)
            if k in mp:
                input = mp[k]
            else:
                request = requests.post(endpoint, headers=headers, data={
                    'source_lang': f,
                    'target_lang': t,
                    'text': input
                })
                response = request.json()

                input = mp[k] = [r['text'] for r in response["translations"]]

        out.append(input)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('-i', '--input-language', default='sl')
    parser.add_argument(
        '-k', '--key', default=os.environ.get('TRANSLATOR_TEXT_SUBSCRIPTION_KEY', None))
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'de'])
    args = parser.parse_args()

    with open(args.inputfile, 'rb') as f:
        contents = f.read().decode('utf8')

    contents = re.sub(r'[\t\n\r]+', ' ', contents)

    nlp = spacy.load('en_core_web_trf')

    # sentences = [str(f).strip() for f in nlp(contents).sents if str(f).strip()]
    sentences = [str(f).strip() for f in sent_tokenize(
        contents, language='slovene') if str(f).strip()][0:100]
    res = [sentences]
    if len(sentences):
        res = translate(
            sentences,
            language=args.input_language,
            to=args.languages,
            backtransleate=True,
            key=args.key
        )

    with open(args.out, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='unix')
        l = len(res)
        writer.writerow(['sentence{}'.format(i + 1) for i in range(l + 1)])
        for r in zip(sentences, *res):
            writer.writerow(r)
