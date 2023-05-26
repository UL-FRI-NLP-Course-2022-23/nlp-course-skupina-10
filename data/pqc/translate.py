import csv

import requests
from joblib import Parallel, delayed
from tqdm import tqdm

endpoint = "http://localhost:4000/api/translate"


def tr(input):
    request = requests.post(
        endpoint,
        headers={},
        json={"src_language": "en", "tgt_language": "sl", "text": input},
    )
    response = request.json()
    return response["result"]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


for path in ["combined.csv", "combined_eval.csv"]:
    with open("data/" + path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")

        with open("ml_" + path, "w", newline="", encoding="utf-8") as file_t:
            writer = csv.writer(file_t, delimiter="\t")
            rs = list(reader)
            writer.writerow(rs[0])

            def proc(l):
                en = [f for x in l for f in x[1:3]]
                sl = tr(en)

                en = list(chunks(en, 2))
                sl = list(chunks(sl, 2))
                es = zip(en, sl)

                r = []
                for en, sl in es:
                    en1, en2 = en
                    sl1, sl2 = sl

                    vals = [
                        (sl1, "sl_SI"),
                        (sl2, "sl_SI"),
                        (en1, "en_XX"),
                        (en2, "en_XX"),
                    ]
                    pairs = [
                        [vals[0], vals[1]],
                        [vals[2], vals[3]],
                        [vals[1], vals[2]],
                        [vals[3], vals[0]],
                    ]

                    for p in pairs:
                        p1, p2 = p
                        r.append([l[0], p1[0], p2[0], p1[1], p2[1]])

                return r

            rs = list(chunks(rs[1:], 8))
            upd = Parallel(n_jobs=3)(delayed(proc)(l) for l in tqdm(rs))
            for f in upd:
                for r in f:
                    writer.writerow(r)
