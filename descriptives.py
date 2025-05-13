# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset


# and significance testing
from scipy import stats
# get spearmans
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

# %%

# Some descriptive statistics for the dataset (fiction vs. nonfiction)

# get the stylistics
df = pd.read_csv("data/stylistics.csv", sep="\t")
df.head()

# load it from HF
dataset = load_dataset("chcaa/feuilleton_dataset")
# get the train split
text_df = dataset["train"].to_pandas()
text_df = text_df[["article_id", "text"]]
text_df.head()

# merge the two dataframes
df = pd.merge(df, text_df, on="article_id", how="left")
df.head()

# %%

df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))

# remove very short texts
threshold =100
df = df[df["text_len"] > threshold]

# we split the data into two groups: fiction and nonfiction
df_fiction = df[df["label"] == 'fiction']
df_nonfiction = df[df["label"] == 'non-fiction']

# get the number of texts in each group
print("Number of fiction texts:", len(df_fiction))
print("Number of nonfiction texts:", len(df_nonfiction))

# get the mean and std of the text length
mean_fiction = df_fiction["text_len"].mean()
std_fiction = df_fiction["text_len"].std()
mean_nonfiction = df_nonfiction["text_len"].mean()
std_nonfiction = df_nonfiction["text_len"].std()
print("Mean text length fiction:", mean_fiction)
print("Std text length fiction:", std_fiction)
print("Mean text length nonfiction:", mean_nonfiction)
print("Std text length nonfiction:", std_nonfiction)

# %%

# get the mean and std of the stylistics features
cols_to_drop = ["article_id", "text", "text_len", "feuilleton_id", "is_feuilleton", "label", 
                "sentiment", "past_tense_ratio", "present_tense_ratio", "num_sents", "avg_mdd", "std_mdd", "apen_sentiment"]
stylistics_features = [col for col in df.columns if col not in cols_to_drop]

# get the mean and std of the stylistics features
def get_mean_std(df, features):
    means = {}
    stds = {}
    for feature in features:
        means[feature] = df[feature].mean()
        stds[feature] = df[feature].std()

    return means, stds

means_fiction, stds_fiction = get_mean_std(df_fiction, stylistics_features)
means_nonfiction, stds_nonfiction = get_mean_std(df_nonfiction, stylistics_features)
# make it a dataframe
means_fiction_df = pd.DataFrame.from_dict(means_fiction, orient='index', columns=['mean_fiction'])
means_nonfiction_df = pd.DataFrame.from_dict(means_nonfiction, orient='index', columns=['mean_nonfiction'])
stds_fiction_df = pd.DataFrame.from_dict(stds_fiction, orient='index', columns=['std_fiction'])
stds_nonfiction_df = pd.DataFrame.from_dict(stds_nonfiction, orient='index', columns=['std_nonfiction'])
# merge the dataframes
means_std_df = pd.merge(means_fiction_df, means_nonfiction_df, left_index=True, right_index=True)
means_std_df = pd.merge(means_std_df, stds_fiction_df, left_index=True, right_index=True)
means_std_df = pd.merge(means_std_df, stds_nonfiction_df, left_index=True, right_index=True)
means_std_df = means_std_df.reset_index()
means_std_df



# %%
# we want to see which features are significantly different between the two groups
# get the p-values for each feature
def get_p_values(df_fiction, df_nonfiction, features):
    p_values = {}
    for feature in features:
        # get the p-value
        t_stat, p_value = stats.ttest_ind(df_fiction[feature], df_nonfiction[feature])
        p_values[feature] = p_value, t_stat
    return p_values, t_stat

p_values, t_stat = get_p_values(df_fiction, df_nonfiction, stylistics_features)
# make it a dataframe
p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value', 't_stat'])
# merge with t_stat
p_values_df
# %%

# drop outlier in sentence length
threshold = 100
df_fiction = df_fiction[df_fiction["avg_sentlen"] < threshold]
df_nonfiction = df_nonfiction[df_nonfiction["avg_sentlen"] < threshold]

# and outlier in noun_ttr
threshold = 6
df_fiction = df_fiction[df_fiction["noun_ttr"] < threshold]
df_nonfiction = df_nonfiction[df_nonfiction["noun_ttr"] < threshold]

# get len of df_fiction and df_nonfiction
print("Number of fiction texts after dropping outliers:", len(df_fiction))
print("Number of nonfiction texts after dropping outliers:", len(df_nonfiction))

# lets do sns histplot of the features
sns.set(style="whitegrid")
for feature in stylistics_features:
    if feature == "nominal_verb_ratio":
        # remove the values above 13
        df_fiction = df_fiction[df_fiction[feature] < 12]
        df_nonfiction = df_nonfiction[df_nonfiction[feature] < 12]

    plt.figure(figsize=(7, 3))
    sns.histplot(df_fiction[feature], color='blue', label='Fiction', alpha=0.5, stat='density')
    sns.histplot(df_nonfiction[feature], color='red', label='Nonfiction', alpha=0.5, stat='density')
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"figs/{feature}_distribution.png", dpi=300)
    plt.show()

# %%
