
# %%

# --- Clustering task for testing embeddings ---

# we want to test 4 different embeddings on how they can determine clusters
# the gold standard is the feuilleton_id

# we will use the following embeddings:
# 1. e5 large
# 2. MeMo-BERT
# 3. 
# 4. 

# we will use the KMeans clustering algorithms:

# %%

from sklearn.metrics.cluster import v_measure_score

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datasets import load_from_disk, load_dataset
import logging
import os

# %%
# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/clustering_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,              # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True,                # Force logging even if already configured
)


# %%

embeddings_dir = "data/pooled"

embeddings_paths = [
    "2025-04-29_embs_e5",
    "2025-04-30_embs_memo",
    "2025-05-14_embs_jina",
    "2025-05-14_embs_bilingual",
    "2025-05-14_embs_solon"
    ]


# -- Load and prepare data --

# Load feuilleton dataset
dataset = load_dataset("chcaa/feuilleton_dataset")
df = dataset["train"].to_pandas()
df = df[["article_id", "label", "feuilleton_id"]]
df.head()

# %%

def get_clusters(df):
    X = np.vstack(df["embedding"].values)
    y = df["feuilleton_id"].values

    # Set number of clusters to number of unique feuilleton_ids
    n_clusters = np.unique(y).shape[0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    return clusters

# %%

for path in embeddings_paths:
    # Load the embeddings
    embs = load_from_disk(f"{embeddings_dir}/{path}")
    embs = embs.to_pandas()
    logging.info("----------------------")
    logging.info(f"Loaded embeddings from {path}")
    logging.info(f"Number of rows in {path}: {len(embs)}")

    # merge with df
    merged = embs[['article_id', 'embedding']].merge(df, on="article_id")
    merged = merged[merged['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
    merged = merged[~merged['feuilleton_id'].isna()]
    print("Number of datapoints after removing no feuilleton_id:", len(merged))
    logging.info(f"Number of datapoints after removing no feuilleton_id: {len(merged)}")

    # Check the number of rows
    print(f"Number of rows in {path} after filtering: {len(merged)}")
    logging.info(f"Number of rows in {path} after filtering: {len(merged)}")

    clusters = get_clusters(merged)
    print("number of clusters:", len(np.unique(clusters)))
    print("number of feuilleton_ids:", len(np.unique(merged["feuilleton_id"].values)))

    y = merged["feuilleton_id"].values

    # get performance metrics
    ari = round(adjusted_rand_score(y, clusters),3)
    nmi = round(normalized_mutual_info_score(y, clusters),3)
    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information Score:", nmi)
    logging.info(f"Adjusted Rand Index: {ari}")
    logging.info(f"Normalized Mutual Information Score: {nmi}")

    # get v-score
    v_score = round(v_measure_score(y, clusters),3)
    print("V-measure Score:", v_score)
    logging.info(f"V-measure Score: {v_score}")



# %%

# get the jina model embeddings and merge with the feuilleton dataset
embs = load_from_disk(f"{embeddings_dir}/2025-05-14_embs_jina")
embs = embs.to_pandas()
# see the n_chunks
embs['n_chunks_orig'].value_counts()

# plot it
sns.histplot(embs['n_chunks_orig'], bins=100)
plt.xlabel("Number of chunks")
plt.ylabel("Number of articles")
plt.title("Distribution of number of chunks per article")
plt.show()

# %%
# Separate out the rows with dummy IDs and real feuilleton IDs
real_feuilletons = merged[~merged['feuilleton_id'].str.contains("noid_")]
# Randomly pick n feuilleton_ids
n = 20
sampled_ids = np.random.choice(real_feuilletons["feuilleton_id"].unique(), size=n, replace=False)
sampled_df = real_feuilletons[real_feuilletons["feuilleton_id"].isin(sampled_ids)]

print("Total samples:", len(sampled_df))
print("Number of clusters:", sampled_df["feuilleton_id"].nunique())

# Extract embeddings and labels
X = np.vstack(sampled_df["embedding"].values)
y = sampled_df["feuilleton_id"].values

# Clustering
n_clusters = np.unique(y).shape[0]
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab20", s=50, alpha=0.9)
plt.title("KMeans Clustering of Feuilleton Embeddings (PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()
# %%
# %%
