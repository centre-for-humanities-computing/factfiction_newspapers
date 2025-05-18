
# %%

# --- Clustering task for testing embeddings ---

# we want to test 4 different embeddings on how they can determine clusters
# the gold standard is the feuilleton_id

# we will use the KMeans clustering algorithm

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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

embeddings_dir = "data_all/pooled"

embeddings_paths = [
    "2025-04-29_embs_e5",
    "2025-04-30_embs_memo",
    "2025-05-14_embs_jina",
    "2025-05-14_embs_bilingual",
    "2025-05-14_embs_solon",
    "2025-05-16_old_news"
    ]

# -- Load and prepare data --

# Load feuilleton dataset
dataset = load_dataset("chcaa/feuilleton_dataset")
df = dataset["train"].to_pandas()
df = df[["article_id", "label", "feuilleton_id"]]
df.head()


# %%
# -- Clustering task --


save_dict = {}

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
    X = np.vstack(merged["embedding"].values)
    y = merged["feuilleton_id"].values
    # Set number of clusters to number of unique feuilleton_ids
    n_clusters = np.unique(y).shape[0]
    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # note down the number of clusters
    print("number of clusters:", len(np.unique(clusters)), ", should be same as:", len(np.unique(merged["feuilleton_id"].values)))

    # get performance metrics
    ari = round(adjusted_rand_score(y, clusters),3)
    print("Adjusted Rand Index:", ari)
    logging.info(f"Adjusted Rand Index: {ari}")

    # get v-score
    v_score = round(v_measure_score(y, clusters),3)
    print("V-measure Score:", v_score)
    logging.info(f"V-measure Score: {v_score}")

    # save 
    save_dict[path] = {
        "ari": ari,
        "v_score": v_score,
        "n_clusters": n_clusters
    }

# write savedict to file
with open("logs/clustering_results.txt", "a") as f:
    f.write("\n\n")
    f.write("Clustering report:\n")
    for path, metrics in save_dict.items():
        f.write(f"{path}: {metrics}\n")
    f.write("\n\n")


# %%

# just extra checking
import umap.umap_ as umap

# Load embeddings
embs = load_from_disk(f"{embeddings_dir}/2025-05-14_embs_jina")
embs = embs.to_pandas()

# Merge to get feuilleton_id
merged = embs[['article_id', 'embedding']].merge(df, on="article_id")

# Filter out bad embeddings
merged = merged[merged['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]

# Keep only real feuilleton entries
real_feuilletons = merged[~merged['feuilleton_id'].isna()]

# Sample feuilleton_ids
n = 100
sampled_ids = np.random.choice(real_feuilletons["feuilleton_id"].unique(), size=n, replace=False)
sampled_df = real_feuilletons[real_feuilletons["feuilleton_id"].isin(sampled_ids)]

print("Total samples:", len(sampled_df))

# Prepare data
X = np.vstack(sampled_df["embedding"].values)
y = sampled_df["feuilleton_id"].values

# Clustering
n_clusters = np.unique(y).shape[0]
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# UMAP projection
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
X_umap = reducer.fit_transform(X)

# Prepare for plotting
plot_df = pd.DataFrame(X_umap, columns=['x', 'y'])
plot_df['feuilleton_id'] = y
plot_df['cluster'] = clusters

sns.set(style="whitegrid")
# Plot
plt.figure(figsize=(10, 10))
sns.scatterplot(data=plot_df, x='x', y='y', hue='feuilleton_id', palette='Set1', alpha=0.7, s=200, edgecolor='w')
plt.title('UMAP Projection of Clusters')
plt.tight_layout()
plt.legend([],[], frameon=False)
plt.savefig("figs/umap_clusters_JINA.png", dpi=300)
plt.show()
# %%
