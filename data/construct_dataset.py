import srt
import re
import pandas as pd


def match(text):
    """Match text in between > < tags"""
    return re.findall(r'<.*?>(.*?)<.*?>', text)


def construct_dataset(f_name):
    """Construct dataset from merged subtitle file"""
    dataset = []
    for sub in srt.parse(open(f_name, "r", encoding="utf-8").read()):
        sub = sub.content.replace("\n", " ").lstrip()
        subs = match(sub)
        if len(subs) != 2: continue
        sub1, sub2 = subs[0], subs[1]
        if sub1.lower() != sub2.lower():
            # print([sub1, sub2])
            dataset.append([sub1, sub2])
    return dataset


if __name__ == "__main__":
    # $iconv -f WINDOWS-1252 -t UTF-8 avatar1.srt
    # https://subtitletools.com/merge-subtitles-online/ab6dba896edd427a
    f_name = "./subtitles/avatar_merged.srt"
    dataset = construct_dataset(f_name)
    print(f"Number of extracted 'paraphrases': {len(dataset)}")
    df = pd.DataFrame(dataset, columns=["sentence 1", "sentence 2"])
    print(df.head())
    df.to_csv("dataset.csv", index=False)
