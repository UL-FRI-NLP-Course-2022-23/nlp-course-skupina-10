import argparse
import csv
import os
import uuid

import requests
import spacy
from nltk.tokenize import sent_tokenize

endpoint = 'https://api.cognitive.microsofttranslator.com'


def translate(
    text,
    to,
        language='sl',
        backtransleate=False,
        region=os.environ.get('TRANSLATOR_TEXT_REGION', None),
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

    path = '/translate?api-version=3.0'

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    if not isinstance(text, list):
        text = [text]

    input = [{
        'text': t
    } for t in text]
    out = []

    mp = {}

    for c in chains:
        for j, p in enumerate(c):
            f, t = p
            params = '&from={}&to={}'.format(f, t)
            k = '{}_{}'.format(params, j)
            if k in mp:
                input = mp[k]
            else:
                constructed_url = endpoint + path + params

                request = requests.post(
                    constructed_url, headers=headers, json=input)
                response = request.json()

                input = mp[k] = [{
                    'text': r["translations"][0]['text']
                } for r in response]

        out.append([t['text'] for t in input])

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('-i', '--input-language', default='sl')
    parser.add_argument(
        '-r', '--region', default=os.environ.get('TRANSLATOR_TEXT_REGION', None))
    parser.add_argument(
        '-k', '--key', default=os.environ.get('TRANSLATOR_TEXT_SUBSCRIPTION_KEY', None))
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'de'])
    args = parser.parse_args()

    with open(args.inputfile, 'rb') as f:
        contents = f.read().decode('utf8')

    contents = contents.replace('\n', ' ')

    nlp = spacy.load('en_core_web_trf')

    # sentences = [str(f).strip() for f in nlp(contents).sents if str(f).strip()]
    sentences = [str(f).strip() for f in sent_tokenize(
        contents, language='slovene') if str(f).strip()]
    res = [sentences]
    if len(sentences):
        res = translate(
            sentences,
            language=args.input_language,
            to=args.languages,
            backtransleate=True,
            key=args.key,
            region=args.region
        )

    with open(args.out, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='unix')
        l = len(res)
        writer.writerow(['sentence{}'.format(i + 1) for i in range(l + 1)])
        for r in zip(sentences, *res):
            writer.writerow(r)
