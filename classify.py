# %%


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.inspection import PartialDependenceDisplay

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import logging

# %%

# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/classification_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

# GET DATA
# get the mfws
mfw_df = pd.read_csv("data/mfw_100.csv", sep="\t")
mfw_500_df = pd.read_csv("data/mfw_500.csv", sep="\t")
# get the tfidf
tfidf_df = pd.read_csv("data/tfidf_5000.csv", sep="\t")
# get the stylistics
stylistics = pd.read_csv("data/stylistics.csv", sep="\t")
# embeddings from parquet
embeddings = pd.read_parquet("data/embeddings_jina.parquet")

# get the original feuilleton data
df = pd.read_csv("data/cleaned_feuilleton.csv", sep="\t")


# --- CLEANING CONFIG ---
MIN_LENGTH = 100
CLEAN = False

# option to clean away very short texts
def clean_short_texts(df, min_length=MIN_LENGTH):
    # Compute word count per row
    df = df[df["text"].str.split().apply(len) >= MIN_LENGTH]
    return df


if CLEAN:
    # Step 1: Filter the cleaned feuilleton to only include longer texts
    df_cleaned = clean_short_texts(df, min_length=MIN_LENGTH)

    # Step 2: Get the valid article_ids
    valid_ids = set(df_cleaned["article_id"])

    # Step 3: Filter all feature DataFrames to only include those IDs
    mfw_df = mfw_df[mfw_df["article_id"].isin(valid_ids)]
    mfw_500_df = mfw_500_df[mfw_500_df["article_id"].isin(valid_ids)]
    tfidf_df = tfidf_df[tfidf_df["article_id"].isin(valid_ids)]
    stylistics = stylistics[stylistics["article_id"].isin(valid_ids)]
    embeddings = embeddings[embeddings["article_id"].isin(valid_ids)]

    all_dfs = [mfw_df, mfw_500_df, tfidf_df, stylistics, embeddings]

    # check if all dfs are the same len as the df_cleaned
    for dataf in all_dfs:
        if len(dataf) != len(df_cleaned):
            print(f"Length mismatch for {dataf.head(2)}")
            print(f"Mismatch: {len(dataf)} vs {len(df_cleaned)}")

    print(f"Cleaning done. Removed {len(df) - len(df_cleaned)} texts that had < {MIN_LENGTH} words.")
    logging.info(f"Cleaning done. Removed {len(df) - len(df_cleaned)} texts.")


# --- SETTING UP THE DATAFRAME ---
# set USE_DF
use_df = stylistics#tfidf_df#mfw_df.copy()#embeddings

# remove duplicates of article_id
use_df = use_df.drop_duplicates(subset=["article_id"])
# and any columns starting "Unnamed"
use_df = use_df.drop(columns=[col for col in use_df.columns if col.startswith("Unnamed")])

# Additional cleaning if using embeddings
if "embedding" in use_df.columns:
    # Drop rows where the embedding is not a proper array or contains any NaN
    len_before = len(use_df)
    use_df = use_df[use_df['embedding'].apply(lambda x: isinstance(x, np.ndarray) and not np.isnan(x).any())]
    len_after = len(use_df)
    print(f"Removed {len_before - len_after} invalid embeddings.")
    logging.info(f"Removed {len_before - len_after} rows of invalid embeddings.")
    print("")
else:
    print("No additional filtering needed.")

# finally, remove "article_ID" from use_df
use_df = use_df.drop(columns=["article_id"])
# final check
print("Number of datapoints in each category: All:", len(use_df), "; nonfic:", len(use_df.loc[use_df['is_feuilleton'] == 'n']), "; fiction:", len(use_df.loc[use_df['is_feuilleton'] == 'y']))
use_df.head()

# %%

def balance_classes(df):
    # Separate the dataframe into two classes
    df_fiction = df[df["is_feuilleton"] == "y"]
    df_nonfiction = df[df["is_feuilleton"] == "n"]

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

# balance classes
balanced_df = balance_classes(use_df)

# Define features (bit different if using embeddings)
def get_features(df):
    if "embedding" in df.columns:
        X = np.vstack(df['embedding'].values)
        print("features used: embedding")
        logging.info(f"Features used: EMBEDDING")
    else:
        X = df.drop(columns=['is_feuilleton','feuilleton_id'])
        print("features used:", X.columns)
        logging.info(f"Features used: {X.columns}")
    return X

X = get_features(balanced_df)

# Define target
y = balanced_df["is_feuilleton"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get precision, recall, and F1-score
print(classification_report(y_test, y_pred))



# %%
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
y = balanced_df["is_feuilleton"]

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
    precisions_y.append(report['y']['precision'])
    recalls_y.append(report['y']['recall'])
    f1_scores_y.append(report['y']['f1-score'])
    
    # For 'n' (nonfiction class)
    precisions_n.append(report['n']['precision'])
    recalls_n.append(report['n']['recall'])
    f1_scores_n.append(report['n']['f1-score'])

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

# Calculate overall (mean) performance across all folds
print("\nOverall performance across all folds:")
print(f"Average Accuracy: {np.mean(accuracies):.2f}")
print(f"Average Precision for 'y' (fiction): {np.mean(precisions_y):.2f}")
print(f"Average Recall for 'y' (fiction): {np.mean(recalls_y):.2f}")
print(f"Average F1 Score for 'y' (fiction): {np.mean(f1_scores_y):.2f}")
print(f"Average Precision for 'n' (nonfiction): {np.mean(precisions_n):.2f}")
print(f"Average Recall for 'n' (nonfiction): {np.mean(recalls_n):.2f}")
print(f"Average F1 Score for 'n' (nonfiction): {np.mean(f1_scores_n):.2f}")

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

# %%
# Visualize Partial Dependence for top features by importance
# Plot Partial Dependence for the top 3 features with the highest importance
# Plot the top 3 features by mean importance
top_features = mean_importances.head(10).index

sns.set_style("whitegrid")
plt.figure(figsize=(20, 12))
PartialDependenceDisplay.from_estimator(
    clf,
    X_train,  # Use training set to avoid test set leakage
    features=[X.columns.get_loc(feat) for feat in top_features],  # Convert names to indices
    feature_names=X.columns.tolist(),
    kind='average',  # 'individual' for ICE, 'both' for both ICE and PDP
    grid_resolution=50
)
plt.suptitle('Partial Dependence of Top Features Across Last Fold')
plt.show()



# %%
# %%
