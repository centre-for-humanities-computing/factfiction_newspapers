
# %%

# --- Clustering task for testing embeddings ---

# we want to test 4 different embeddings on how they can determine clusters
# the gold standard is the feuilleton_id

# we will use the KMeans clustering algorithm

# %%

from sklearn.metrics.cluster import v_measure_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datasets import load_from_disk, load_dataset
import logging

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

# f to retun the clusters
def get_clusters(df):
    X = np.vstack(df["embedding"].values)
    y = df["feuilleton_id"].values

    # Set number of clusters to number of unique feuilleton_ids
    n_clusters = np.unique(y).shape[0]
    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters


for path in embeddings_paths:
    # Load the embeddings
    embs = load_from_disk(f"{embeddings_dir}/{path}")
    embs = embs.to_pandas()
    logging.info("----------------------")
    logging.info(f"Loaded embeddings from {path}")
    logging.info(f"Number of rows in {path}: {len(embs)}")

    # merge with df to get feuilleton_id
    merged = embs[['article_id', 'embedding']].merge(df, on="article_id")
    # remove faulty embeddings
    merged = merged[merged['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
    # remove rows without feuilleton_id
    # i.e., we are only clustering those that can be clustered (!!)
    merged = merged[~merged['feuilleton_id'].isna()]

    # Check the number of datapoints
    print(f"Number of rows in {path} after filtering: {len(merged)}")
    logging.info(f"Number of rows in {path} after filtering: {len(merged)}")

    # get the clusters
    clusters = get_clusters(merged)
    # note down the number of clusters
    print("number of clusters:", len(np.unique(clusters)), ", should be same as:", len(np.unique(merged["feuilleton_id"].values)))

    y = merged["feuilleton_id"].values

    # get performance metrics
    ari = round(adjusted_rand_score(y, clusters),3)
    print("Adjusted Rand Index:", ari)
    logging.info(f"Adjusted Rand Index: {ari}")

    # get v-score
    v_score = round(v_measure_score(y, clusters),3)
    print("V-measure Score:", v_score)
    logging.info(f"V-measure Score: {v_score}")



# %%

# just extra

# we can check out som random clusters

# get jina embeddings
embs = load_from_disk(f"{embeddings_dir}/2025-05-14_embs_jina")
embs = embs.to_pandas()
# merge w df to get feuilleton_id
merged = embs[['article_id', 'embedding']].merge(df, on="article_id")

# Separate out the rows with dummy IDs and real feuilleton IDs
merged = merged[merged['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
real_feuilletons = merged[~merged['feuilleton_id'].isna()]

# Randomly pick n feuilleton_ids
n = 50
sampled_ids = np.random.choice(real_feuilletons["feuilleton_id"].unique(), size=n, replace=False)
sampled_df = real_feuilletons[real_feuilletons["feuilleton_id"].isin(sampled_ids)]

print("Total samples:", len(sampled_df))

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
