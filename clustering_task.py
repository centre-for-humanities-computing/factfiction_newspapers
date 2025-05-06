
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
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datasets import load_from_disk
from datasets import Dataset

# %%

# get the data
df = pd.read_csv("data/cleaned_feuilleton.csv", sep="\t")
# only columns we need
df = df[['feuilleton_id', 'article_id', 'is_feuilleton']]
# we replace the suffixes (e.g. _a, _b) in the feuilleton_id with empty string to get original id
df['feuilleton_id'] = df['feuilleton_id'].str.replace(r'_[a-z]$', '', regex=True)
df.head()

# %%
# get the embeddings from dataset
path = "data/pooled/2025-04-30_embs_jina"#"data/pooled/2025-04-30_embs_memo" #"data/pooled/2025-04-29_embs_e5"
dataset = load_from_disk(path)
# to a pandas DataFrame
embs = dataset.to_pandas()
embs

# %%
# check how many rows have chunk == 1
only_1_chunk = embs[embs['n_chunks_orig'] == 1]
print("Number of rows with chunk == 1:", len(only_1_chunk))
# print the percentage of rows with chunk == 1
print("Percentage of rows with chunk == 1:", len(only_1_chunk) / len(embs) * 100)

# visualize the distribution of n_chunks_orig
plt.figure(figsize=(6, 4), dpi=500)
sns.histplot(embs['n_chunks_orig'])
plt.title("Number of original chunks")
plt.xlabel("Chunks/article")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# %%

# merge to get the data we need
merged = embs[['article_id', 'embedding']].merge(df, on="article_id")
# we remove the embeddings that are invalid
merged = merged[merged['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
# print how many embeddings we removed
print("Number of embeddings invalid:", len(embs) - len(merged))

# numbers
# now we want to see again how many unique feuilleton_ids we have
print("Number of unique feuilleton_ids:", merged["feuilleton_id"].nunique())
# see the number of unique feuilleton_ids that are also is_feuilleton == y
print("Number of unique feuilleton_ids that are also fiction", merged[merged["is_feuilleton"] == 'y']["feuilleton_id"].nunique())
# see number of datapoints
print("Number of datapoints:", len(merged))

#  we give dummyIDs to the ones missing IDs
missing_mask = merged['feuilleton_id'].isna()
merged.loc[missing_mask, 'feuilleton_id'] = [f"noid_{i}" for i in range(missing_mask.sum())]
merged.tail()
# %%

# --- Kmeans clustering ---

def get_clusters(df):
    X = np.vstack(df["embedding"].values)
    y = df["feuilleton_id"].values

    # Set number of clusters to number of unique feuilleton_ids
    n_clusters = np.unique(y).shape[0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    # print the number of clusters
    print("Number of clusters:", len(np.unique(clusters)), ", should be '907'")

    # get performance metrics
    ari = round(adjusted_rand_score(y, clusters),3)
    nmi = round(normalized_mutual_info_score(y, clusters),3)
    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information Score:", nmi)

    return clusters

clusters = get_clusters(merged)


# %%

# Separate out the rows with dummy IDs and real feuilleton IDs
real_feuilletons = merged[~merged['feuilleton_id'].str.contains("noid_")]
# Randomly pick n feuilleton_ids
n = 100
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
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab20", s=30, alpha=0.7)
plt.title("KMeans Clustering of Feuilleton Embeddings (PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.show()
# %%
# %%
