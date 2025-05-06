# %%

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import spacy
from transformers import pipeline, AutoTokenizer

from feats_functions import get_spacy_of_text
from feats_functions import get_nominal_verb_ratio_from_saved
from feats_functions import avg_sentlen, avg_wordlen
from feats_functions import calculate_dependency_distances
from feats_functions import compressrat
from feats_functions import get_sentiment
from lexical_diversity import lex_div as ld

import logging

# %%

# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/get_feats_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

# get data
df = pd.read_csv("data/cleaned_feuilleton.csv", sep="\t")
df.head()

# %%
# 1. MFW 100 & 500

def get_mfw(df, number_of_mfws=100):
    # init CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    # df from count matrix
    word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df.index)

    # get top N most frequent words across all documents
    total_counts = word_counts.sum(axis=0)
    top_words = total_counts.sort_values(ascending=False).head(number_of_mfws).index

    # slice df to get only set number of MFWs
    mfw_df = word_counts[top_words]

    # normalize each row by its total word count (i.e., per-document normalization)
    mfw_df = mfw_df.div(word_counts.sum(axis=1), axis=0)
    return mfw_df

mfw_100_df = get_mfw(df, 100)
# add the is_feuilleton column
mfw_100_df["feuilleton_id"] = df["feuilleton_id"]
mfw_100_df["is_feuilleton"] = df["is_feuilleton"]
mfw_100_df["article_id"] = df["article_id"]
# save
mfw_100_df.to_csv("data/mfw_100.csv", sep="\t")

mfw_500_df = get_mfw(df, 500)
# add the is_feuilleton column
mfw_500_df["feuilleton_id"] = df["feuilleton_id"]
mfw_500_df["is_feuilleton"] = df["is_feuilleton"]
mfw_500_df["article_id"] = df["article_id"]
# save
mfw_500_df.to_csv("data/mfw_500.csv", sep="\t")

logging.info(f"get_mfw: created mfw_100 and mfw_500 dataframes. Saved to data/mfw_100.csv and data/mfw_500.csv.")
mfw_100_df.head()


# %%
# 2. create tf-idf vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["text"]).toarray()

# make TF-IDF matrix df
tfidf_df = pd.DataFrame(X_tfidf, columns=tfidf_vectorizer.get_feature_names_out(), index=df.index)
tfidf_df['is_feuilleton'] = df['is_feuilleton']
tfidf_df['feuilleton_id'] = df['feuilleton_id']
tfidf_df['article_id'] = df['article_id']

# save
tfidf_df.to_csv("data/tfidf_5000.csv", sep="\t")
logging.info(f"get_mfw: created tfidf_5000 dataframe. Saved to data/tfidf_5000.csv.")

# %%
# 3. get stylistic features

# define model
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
# load SA model
xlm_model = pipeline("text-classification", model=model_name)
# & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# to save stuff
stylistics_data = []

# loop to get stylistics
for i, row in df.iterrows():
    text_id = row['article_id']
    # get the spacy
    spacy_df = get_spacy_of_text(row["text"], model_name="da_core_news_sm", out_dir="data", text_id=text_id)

    # get words and sentences
    # words
    words_list = spacy_df[~spacy_df["token_pos_"].isin(["SPACE", "NUM"])]
    words = words_list["token_text"].tolist()
    # sentences
    sentence_list = spacy_df.groupby("sent_id")
    sentences = [" ".join(group["token_text"].tolist()) for _, group in sentence_list]

    # get nominal verb ratio, ttr of nouns, and noun count
    nominal_verb_ratio, num_nouns, noun_ttr, verb_ttr = get_nominal_verb_ratio_from_saved(text_id)
    # get the avg word length
    wordlen = avg_wordlen(text_id)
    # get the avg sentence length
    sentlen, num_sentences = avg_sentlen(text_id)
    # msttr (window len 100) (we use the words from the spacy_df)
    msttr = ld.msttr(words, window_length=40)
    # dependency distances
    full_stop_indices = spacy_df[spacy_df['token_text'].str.strip() == '.'].index
    # Adding the last index of the DataFrame to handle the last sentence
    full_stop_indices = list(full_stop_indices) + [spacy_df.index[-1]]
    ndd_mean, ndd_std, dd_mean, dd_std = calculate_dependency_distances(spacy_df, full_stop_indices)
    # compression ratio
    bzip = compressrat(sentences)

    # SA
    sent_scores = []

    for sent in sentences:
        # get the sentiment
        sent_scores.append(get_sentiment(sent, xlm_model, tokenizer))
    # get the mean and std of the sentiment scores
    sa_score = np.mean(sent_scores)
    # get the std of the sentiment scores
    sa_std = np.std(sent_scores)

    # save all
    stylistics_data.append({
        "article_id": text_id,
        "nominal_verb_ratio": nominal_verb_ratio,
        #"num_nouns_per_sent": num_nouns/num_sentences,
        "noun_ttr": noun_ttr,
        "verb_ttr": verb_ttr,
        "avg_word_length": wordlen,
        "avg_sentence_length": sentlen,
        "msttr": msttr,
        "ndd_mean": ndd_mean,
        "ndd_std": ndd_std,
        "bzip": bzip,
        "is_feuilleton": row["is_feuilleton"],
        "feuilleton_id": row["feuilleton_id"],
        "article_id": row["article_id"],
        "sa_score": sa_score,
        "sa_std": sa_std,
    })

# Create the final DataFrame after the loop
stylistics_df = pd.DataFrame.from_dict(stylistics_data)

# save
stylistics_df.to_csv("data/stylistics.csv", sep="\t", index=False)
logging.info(f"get_mfw: created stylistics dataframe. Saved to data/stylistics.csv.")

# %%

# get embeddings

from datasets import load_from_disk
from datasets import Dataset
# %%

# Load the dataset from arrow
path = "data/pooled/2025-04-30_embs_jina"#  "data/pooled/2025-04-29_embs_e5"
dataset = load_from_disk(path)
# Convert to a pandas DataFrame
embs = dataset.to_pandas()
# merge feuilleton (y/n) on article_id
embs_df = pd.merge(embs[['article_id', 'embedding']], df[["article_id", "is_feuilleton", "feuilleton_id"]], on="article_id", how="left")
print(len(embs_df))
embs_df.head()

# %%
# save to parquet
embs_df.to_parquet("data/embeddings_jina.parquet", index=False)
logging.info(f"get_mfw: created embeddings dataframe. Saved to data/embeddings.parquet.")
# %%

from transformers import pipeline
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
sentiment_task("T'estimo!")
# %%
