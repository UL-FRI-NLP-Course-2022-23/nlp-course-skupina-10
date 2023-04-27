import argparse
import glob
import re

from conllu import parse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', required=True)
    args = parser.parse_args()

    existing = set()

    with open(args.out, 'w', encoding='utf-8') as out:
        files = glob.glob('./suk/*.conllu')
        for file in files:
            with open(file, 'rb') as f:
                contents = f.read().decode('utf8')
            sentences = parse(contents)
            for s in sentences:
                txt = s.metadata['text']
                txt = re.sub(r'[\t\n\r\s]+', ' ', txt)

                if txt not in existing:
                    existing.add(txt)
                    out.write(txt + '\n')
