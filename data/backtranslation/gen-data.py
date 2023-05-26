import argparse
import csv
import os

import requests
from tqdm import tqdm

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


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('-i', '--input-language', default='sl')
    parser.add_argument(
        '-k', '--key', default=os.environ.get('TRANSLATOR_TEXT_SUBSCRIPTION_KEY', None))
    parser.add_argument('-o', '--out', required=True)
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'de'])
    parser.add_argument('-s', '--skip', default=0, type=int)
    args = parser.parse_args()

    with open(args.inputfile, 'rb') as f:
        contents = f.read().decode('utf8')

    sentences = [s.strip() for s in contents.split('\n') if s.strip()]
    skip = args.skip > 0

    with open(args.out, 'a' if skip else 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, dialect='unix')
        l = len(args.languages)
        if skip:
            sentences = sentences[args.skip:]

        t = tqdm(total=len(sentences))
        for batch in divide_chunks(sentences, 100):
            if len(batch):
                res = translate(
                    batch,
                    language=args.input_language,
                    to=args.languages,
                    backtransleate=True,
                    key=args.key
                )
                for r in zip(batch, *res):
                    writer.writerow(r)
            t.update(len(batch))
        t.close()
