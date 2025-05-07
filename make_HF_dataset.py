# %%

import pandas as pd
from datasets import Dataset
from datasets import load_from_disk


# %%

# get the data
df = pd.read_csv("data/cleaned_feuilleton.csv", sep="\t")
df = df[['text', 'is_feuilleton', 'feuilleton_id', 'article_id']]
# in "is_feuilleton", replace y with fiction and n with non-fiction
df['is_feuilleton'] = df['is_feuilleton'].replace({'y': 'fiction', 'n': 'non-fiction'})
df.tail()
# %%
# rename cols
df = df.rename(columns={"text": "text", "is_feuilleton": "label", "feuilleton_id": "feuilleton_id", "article_id": "article_id"})
print(len(df))
df.head()

# %%
# load the dates & merge
dates = pd.read_excel("data/feuilleton_annotation.xlsx")
dates = dates[['date', 'feuilleton_author', 'article_id']]
# merge on article_id
df = df.merge(dates, on="article_id", how="left")
print(len(df))
# remove duplicates
df = df.drop_duplicates(subset=["article_id", "text", "label", "feuilleton_id"])
print(len(df))
df.head()
# %%
# upload to HF
dataset = Dataset.from_pandas(df)
# push
dataset.push_to_hub("chcaa/feuilleton_dataset", split="train")

# %%
