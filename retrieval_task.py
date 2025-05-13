
# %%
from datasets import load_dataset

# "Given a query, retrieve relevant documents from a corpus."

# %%

# get data from HF
dataset = load_dataset("chcaa/feuilleton_dataset")
# get the train split
df = dataset["train"].to_pandas()
df = df[["text", "label", "feuilleton_id", "article_id"]]
df.head()
# %%
import pandas as pd
from collections import defaultdict

# group all text by feuilleton_id (documents)
corpus_dict = defaultdict(list)
for _, row in df.iterrows():
    corpus_dict[row["feuilleton_id"]].append(row["text"])

# build the corpus: _id, title, text (if title not available, omit)
corpus = {
    fid: {
        "title": "",  # optional: use metadata if you have headlines
        "text": " ".join(texts)
    }
    for fid, texts in corpus_dict.items()
}

# now pick queries: e.g., article-level entries
queries = {}
qrels = defaultdict(dict)

for _, row in df.iterrows():
    qid = str(row["article_id"])
    fid = str(row["feuilleton_id"])
    query_text = row["text"]  # simple setup: treat full article as query

    # one article = one query
    queries[qid] = query_text
    qrels[qid][fid] = 1  # assuming relevance is binary