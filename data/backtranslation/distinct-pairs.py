import argparse
import csv
import os

import requests
import random

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
    parser.add_argument('-t', '--out-train', required=True)
    parser.add_argument('-d', '--out-dev', required=True)
    args = parser.parse_args()

    with open(args.out_train, 'w', newline='', encoding='utf-8') as file_t:
        writer_t = csv.writer(file_t, delimiter='\t')
        writer_t.writerow(['sentence1', 'sentence2', 'label'])

        with open(args.out_dev, 'w', newline='', encoding='utf-8') as file_d:
            writer_d = csv.writer(file_d, delimiter='\t')
            writer_d.writerow(['sentence1', 'sentence2', 'label'])

            with open(args.inputfile, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for l in reader:
                    l = list(set(l))
                    if len(l) > 1:
                        pairs = [[a, b]
                                 for idx, a in enumerate(l) for b in l[idx + 1:]]

                        if len(pairs) > 1:
                            # If more than one pair, pick one as validation
                            pick = random.randint(0, len(pairs) - 1)
                        else:
                            # If just one pair, use it either for train or validation
                            pick = 0
                            if random.random() > 0.5:
                                pick = 1

                        for i, p in enumerate(pairs):
                            if i != pick:
                                writer_t.writerow(p + ['1'])
                            else:
                                writer_d.writerow(p + ['1'])
