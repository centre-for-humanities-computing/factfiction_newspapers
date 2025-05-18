# %%


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import StratifiedGroupKFold

import pandas as pd
import numpy as np

from datasets import load_dataset
import logging

# %%

# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/classification_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True
)

# %%
# --- DATA CONFIG ---
DF_NAME = "tfidf_5000" #"mfw_100" # "mfw_500", "tfidf_5000", "embeddings", "stylistics"

# --- CLEANING CONFIG ---
MIN_LENGTH = 100
FILTER = False

logging.info("Starting classification script.")
# write out the config
logging.info(f"DF_NAME: {DF_NAME}")
logging.info(f"MIN_LENGTH: {MIN_LENGTH}, filtering: {FILTER}")

# %%
# GET DATA

if not DF_NAME == "embeddings":
    # load the data from csv
    data = pd.read_csv(f"data/{DF_NAME}.csv", sep="\t")
else:
    # load the data from parquet
    data = pd.read_parquet(f"data/{DF_NAME}.parquet")

data.head()

# %%

def filter_df(df, min_length=MIN_LENGTH):
    # merge with the original dataframe to get the text
    # get the original feuilleton data from HF
    dataset = load_dataset("chcaa/feuilleton_dataset")
    feuilleton_data = dataset["train"].to_pandas()
    df = df.merge(feuilleton_data[["article_id", "text"]], on="article_id", how="left")
    # Filter DataFrame for texts longer than min_length words
    df = df.dropna(subset=["text"])
    df_cleaned = df[df["text"].str.split().apply(len) >= min_length]
    # Drop the "text" column
    df_cleaned = df_cleaned.drop(columns=["text"])
    print(f"Cleaning done. Removed {len(df) - len(df_cleaned)} texts with fewer than {min_length} words.")
    logging.info(f"Cleaning done. Removed {len(df) - len(df_cleaned)} texts.")
    return df_cleaned

def clean_and_process_df(df):
    # Remove duplicates based on 'article_id' and unwanted columns
    df = df.drop_duplicates(subset=["article_id"])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Clean embeddings if present
    if "embedding" in df.columns:
        initial_len = len(df)
        df = df[df['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
        logging.info(f"Removed {initial_len - len(df)} rows of invalid embeddings.")
    else:
        print("No additional filtering of embeddings needed.")

    # Drop the "article_id" column
    df = df.drop(columns=["article_id"])

    # Final check on category counts
    print(f"Number of datapoints: All={len(df)}; non-fiction={len(df[df['label'] == 'non-fiction'])}; fiction={len(df[df['label'] == 'fiction'])}")
    
    return df

if FILTER:
    data = filter_df(data, min_length=MIN_LENGTH)

use_df = clean_and_process_df(data)
use_df.head()

# %%
# drop the sentiment column
if "sentiment" in use_df.columns:
    use_df = use_df.drop(columns=["sentiment"])
# check for NaN values
# drop nan value in label
use_df = use_df.dropna(subset=["label"])
print("NaN values in use_df:")
print(use_df.isna().sum())

if "msttr" in use_df.columns:
    # then we drop some stylistic features
    use_df = use_df[['nominal_verb_ratio', 'msttr', 'noun_ttr', 'verb_ttr',
        'personal_pronoun_ratio', 'function_word_ratio', 
        'of_ratio',
        'that_ratio', 
        #'past_tense_ratio', #'present_tense_ratio', # present tense has a lot of nans sadly
        'passive_ratio', 'active_ratio', 
        'adjective_adverb_ratio',
        'avg_wordlen', 'avg_sentlen',
        #'num_sents', 
        'avg_ndd', 'std_ndd', 
        #'avg_mdd', 'std_mdd',
        'compression_ratio', 
        'sentiment_mean', 
        'sentiment_std',
        'sentiment_abs', 
        'feuilleton_id', 'label']]

# %%

def balance_classes(df):
    # Separate the dataframe into two classes
    df_fiction = df[df["label"] == "fiction"]
    df_nonfiction = df[df["label"] == "non-fiction"]

    # Undersample the nonfiction class to balance with fiction
    df_nonfiction_undersampled = resample(df_nonfiction, 
                                          replace=False,  # Sample without replacement
                                          n_samples=len(df_fiction),  # Match minority class
                                          random_state=42)

    # Concatenate the two classes back together
    balanced_df = pd.concat([df_fiction, df_nonfiction_undersampled])
    # shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle

    # check balance
    print('> len of fiction:', len(df_fiction), '/ nonfiction:', len(df_nonfiction_undersampled))
    logging.info(f"Balanced classes, {len(balanced_df)} samples. Fiction: {len(df_fiction)}, Nonfiction: {len(df_nonfiction_undersampled)}")
    return balanced_df

# %%

# Define features (bit different if using embeddings)
def get_features(df):
    if "embedding" in df.columns:
        X = np.vstack(df['embedding'].values)
        print("features used: embedding")
        logging.info("Features used: EMBEDDING")
    else:
        X = df.drop(columns=['label','feuilleton_id'])
        print("features used:", X.columns)
        logging.info(f"Features used: {X.columns}")
    return X


# Stratified Group K-Fold Cross-Validation

# we want to
# 1 Balance the dataset: Equal number of feuilleton ("y") and non-feuilleton ("n") texts.
# 2 Avoid data leakage: Make sure that texts from the same feuilleton (via feuilleton_id) don't appear in both training and test sets.

# we remove the suffix (_a, _b, etc.) from the feuilleton_id, to have the full feuilletons logged
use_df['feuilleton_id'] = use_df['feuilleton_id'].str.replace(r'_[a-z]$', '', regex=True)

# Balance the classes
balanced_df = balance_classes(use_df)

# Define features
X = get_features(balanced_df)
# Define target
y = balanced_df["label"]

# we want to make sure we have no feuilletons of the same ID in train and testsets
# first we give dummyIDs to the ones missing IDs
missing_mask = balanced_df['feuilleton_id'].isna()
balanced_df.loc[missing_mask, 'feuilleton_id'] = [f"noid_{i}" for i in range(missing_mask.sum())] # fill missing IDs with dummy IDs
# The we group by feuilleton_id
groups = balanced_df['feuilleton_id'] 

# Initialize StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5)  # Split into 5 folds

# store performance metrics across folds
accuracies = []
precisions_y = []
recalls_y = []
f1_scores_y = []
precisions_n = []
recalls_n = []
f1_scores_n = []

importances = []

# Use GroupKFold to split into train/test while ensuring same feuilleton_id stays in one set
for train_index, test_index in sgkf.split(X, y, groups):
    if "embedding" in balanced_df.columns:
        # For embeddings, we need to stack the arrays
        X_train, X_test = np.vstack(X[train_index]), np.vstack(X[test_index])
    else:
        # For other features, we can use the DataFrame directly
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    
    # Get performance metrics for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    # Store precision, recall, and F1 score from the classification report for both 'y' (fiction) and 'n' (nonfiction)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # For 'y' (fiction class)
    precisions_y.append(report['fiction']['precision'])
    recalls_y.append(report['fiction']['recall'])
    f1_scores_y.append(report['fiction']['f1-score'])
    
    # For 'n' (nonfiction class)
    precisions_n.append(report['non-fiction']['precision'])
    recalls_n.append(report['non-fiction']['recall'])
    f1_scores_n.append(report['non-fiction']['f1-score'])

    # save importances
    importances.append(clf.feature_importances_)

    # Print fold performance
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Print train and test set sizes and class balance
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print("Train class balance:\n", y_train.value_counts())
    print("Test class balance:\n", y_test.value_counts())

    print("-------")

# log features used
if not DF_NAME == "embeddings":
    logging.info(f"Features used: {X.columns}")

# Calculate overall (mean) performance across all folds
print("\nOverall performance across all folds:")
print(f"Average Accuracy: {np.mean(accuracies):.2f}")
print(f"Average Precision for 'y' (fiction): {np.mean(precisions_y):.2f}")
print(f"Average Recall for 'y' (fiction): {np.mean(recalls_y):.2f}")
print(f"Average F1 Score for 'y' (fiction): {np.mean(f1_scores_y):.2f}")
print(f"Average Precision for 'n' (nonfiction): {np.mean(precisions_n):.2f}")
print(f"Average Recall for 'n' (nonfiction): {np.mean(recalls_n):.2f}")
print(f"Average F1 Score for 'n' (nonfiction): {np.mean(f1_scores_n):.2f}")
# log it
logging.info(f"Overall performance across all folds:")
logging.info(f"Average Accuracy: {np.mean(accuracies):.2f}")
logging.info(f"Average Precision for 'y' (fiction): {np.mean(precisions_y):.2f}")
logging.info(f"Average Recall for 'y' (fiction): {np.mean(recalls_y):.2f}")
logging.info(f"Average F1 Score for 'y' (fiction): {np.mean(f1_scores_y):.2f}")
logging.info(f"Average Precision for 'n' (nonfiction): {np.mean(precisions_n):.2f}")
logging.info(f"Average Recall for 'n' (nonfiction): {np.mean(recalls_n):.2f}")
logging.info(f"Average F1 Score for 'n' (nonfiction): {np.mean(f1_scores_n):.2f}")

# only do this if not only using embeddings
if "embedding" not in balanced_df.columns:
    # Print feature importances
    # Convert to DataFrame for easier handling
    importances_df = pd.DataFrame(importances, columns=X.columns)
    # Average across folds
    mean_importances = importances_df.mean().sort_values(ascending=False)
    # Print top features by importance
    print("\nTop 20 Features by Average Importance:")
    print(mean_importances.head(20))
    # log it
    logging.info("\nTop 20 Features by Average Importance:")
    logging.info(mean_importances.head(20))

# %%
# get std of precision, recall, f1
print("\nStandard Deviation of Performance Metrics:")
print(f"Std of Accuracy: {np.std(accuracies):.2f}")
print(f"Std of Precision for 'y' (fiction): {np.std(precisions_y):.2f}")
print(f"Std of Recall for 'y' (fiction): {np.std(recalls_y):.2f}")
print(f"Std of F1 Score for 'y' (fiction): {np.std(f1_scores_y):.2f}")

print(f"Std of Precision for 'n' (nonfiction): {np.std(precisions_n):.2f}")
print(f"Std of Recall for 'n' (nonfiction): {np.std(recalls_n):.2f}")
print(f"Std of F1 Score for 'n' (nonfiction): {np.std(f1_scores_n):.2f}")

# %%
