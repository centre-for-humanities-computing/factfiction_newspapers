
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset, load_from_disk
import logging
from tqdm import tqdm

# %%

# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/get_feats_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,              # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True,                # Force logging even if already configured
)

# get data
# load it from HF
dataset = load_dataset("chcaa/feuilleton_dataset")
# get the train split
df = dataset["train"].to_pandas()
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
# add the label column
mfw_100_df["feuilleton_id"] = df["feuilleton_id"]
mfw_100_df["label"] = df["label"]
mfw_100_df["article_id"] = df["article_id"]
# save
mfw_100_df.to_csv("data/mfw_100.csv", sep="\t")

mfw_500_df = get_mfw(df, 500)
# add the is_feuilleton column
mfw_500_df["feuilleton_id"] = df["feuilleton_id"]
mfw_500_df["label"] = df["label"]
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
tfidf_df['label'] = df['label']
tfidf_df['feuilleton_id'] = df['feuilleton_id']
tfidf_df['article_id'] = df['article_id']

# save
tfidf_df.to_csv("data/tfidf_5000.csv", sep="\t")
logging.info(f"get_mfw: created tfidf_5000 dataframe. Saved to data/tfidf_5000.csv.")



# %%
# 3. get stylistics

from scripts.feature_utils import process_text
from scripts.feature_utils import compressrat
from scripts.feature_utils import get_pos_derived_features
from scripts.feature_utils import avg_sentlen, avg_wordlen
from scripts.feature_utils import calculate_dependency_distances
from scripts.feature_utils import get_sentiment

# define model
model_name = "MiMe-MeMo/MeMo-BERT-SA"
# load SA model
pipe = pipeline("text-classification", model=model_name)
# & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# make progress bar
tqdm.pandas(desc="Processing texts")

stylistics_features = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
    text_id = row['article_id']
    text = row["text"]
    
    process_text(text, text_id)

    features = {}

    # POS and morph features
    features.update(get_pos_derived_features(text_id))

    # Average word length
    features["avg_wordlen"] = avg_wordlen(text_id)

    # Average sentence length and number of sentences
    features["avg_sentlen"], features["num_sents"] = avg_sentlen(text_id)

    # Dependency distance metrics
    features.update(calculate_dependency_distances(text_id))

    # Compression ratio
    features["compression_ratio"] = compressrat(text_id)

    # Sentiment analysis with our MeMo-BERT model
    features["sentiment"] = get_sentiment(text_id, pipe, tokenizer)
    #print(features["sentiment"])

    if features["sentiment"] is not None and len(features["sentiment"]) > 1:
        features["sentiment_mean"] = np.mean(features["sentiment"])
        features["sentiment_std"] = np.std(features["sentiment"])
        # we also want an "absolute" strenght
        features["sentiment_abs"] = np.sum(np.abs(features["sentiment"])) / len(features["sentiment"])
    else:
        features["sentiment_mean"] = np.nan
        features["sentiment_std"] = np.nan
    # Add feuilleton and article IDs
    features["feuilleton_id"] = row["feuilleton_id"]
    features["article_id"] = row["article_id"]
    # Add the label
    features["label"] = row["label"]

    # Add the features to the list
    stylistics_features.append(features)

# just print colnames to check
print("stylistics_features columns:")
print(stylistics_features[0].keys())
# Create a DataFrame from the list of dictionaries
stylistics_df = pd.DataFrame(stylistics_features)
# Save the DataFrame to a CSV file
stylistics_df.to_csv("data/stylistics.csv", sep="\t", index=False)

# log it
logging.info(f"Created stylistic features. Colnames: {stylistics_features[0].keys()}")
logging.info(f"stylistics_df shape: {stylistics_df.shape}")
for col in stylistics_features[0].keys():
    # log the number of missing values
    logging.info(f"stylistics_df {col} missing values: {stylistics_df[col].isnull().sum()}")
    # print the distribution of the column
    logging.info(f"stylistics_df {col} distribution: {stylistics_df[col].describe()}")
    logging.info("\n")


# %%

# get embeddings

# Load the dataset from arrow
# embeddings need to be extracted via script at https://anonymous.4open.science/r/encode_feuilletons-6922
path = "data_all/pooled/2025-05-14_embs_jina"
dataset = load_from_disk(path)
# Convert to a pandas DataFrame
embs = dataset.to_pandas()
# merge feuilleton (y/n) on article_id
embs_df = pd.merge(embs[['article_id', 'embedding']], df[["article_id", "label", "feuilleton_id"]], on="article_id", how="left")
# drop anything without label (i.e., not on HF)
embs_df = embs_df[~embs_df['label'].isna()]
print(len(embs_df))
embs_df.head()

# save to parquet
embs_df.to_parquet("data/embeddings.parquet", index=False)
logging.info(f"get_mfw: created embeddings dataframe. Saved to data/embeddings.parquet.")

# %%
