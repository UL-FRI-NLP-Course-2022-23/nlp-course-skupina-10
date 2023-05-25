import pandas as pd

df = pd.read_csv("./ss_dataset_annotated.csv", delimiter=";")

# add header to df
df.columns = ["sentence1", "sentence2"]
df.to_csv("./ss-pairs-train.csv", sep="\t", index=False)
