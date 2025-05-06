
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging

from transformers import pipeline
import torch
# %%

# CONFIGURE

# Configure logging
logging.basicConfig(
    filename='logs/cleaning_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO               # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

# define the gold standard column name
gold_standard = 'is_feuilleton_pascale'
# get data
annot = pd.read_excel('data/feuilleton_annotation.xlsx')
# we will use 'is_feuilleton' as the standard
annot['is_feuilleton'] = annot[gold_standard]
# show
annot.head()


# %%

# ORGANIZE SHEET

# keep only id, text & is_feu
df_raw = annot[['article_id', 'text', 'is_feuilleton', 'feuilleton_id']].copy()

def clean_sheet(df):
    # keep only id, text & is_feu
    df = df[['article_id', 'text', 'is_feuilleton', 'feuilleton_id']].copy()
    print('raw df len:', len(df))
    # remove duplicates from article_id
    df = df.drop_duplicates(subset=['article_id'])
    print('removed duplicates, len now:', len(df))
    # keep only non-na
    df = df.loc[df['is_feuilleton'].notna()]
    print('removed NAs, len now:', len(df))
    # remove the "mixed" annotations
    df = df.loc[df['is_feuilleton'] != 'mixed']
    print('removed "mixed", len now:', len(df))
    # remove "usikker" if there are any
    df = df.loc[df['is_feuilleton'] != 'usikker']
    print('removed "usikker", len now:', len(df))
    # remove whitespace
    df['is_feuilleton'] = df['is_feuilleton'].str.strip()
    # make sure article & feuilleton ID, & is_feuilleton are strings
    df['article_id'] = df['article_id'].astype(str)
    df['is_feuilleton'] = df['is_feuilleton'].astype(str)
    print("report: cleaned sheet")
    logging.info(f"Cleaning: cleaned sheet. Removed NAs, mixed, usikker. Length of processed df: {len(df)}")
    return df

df = clean_sheet(df_raw)


# %%

# DEFINE CATEGORIES FOR BINARY CLASSIFICATION

categories_dict = {
    "y": "y",
    "n": "n",
    #"merged": "y",
    # plus:
    # various genres
    "bio": "y",
    "anecdote": "n", # using anecdotes?
    "speech": "n",
    "essay": "n",
    "poem": "n",
}
print('Raw categories:', df['is_feuilleton'].value_counts())
logging.info(f"Raw categories: {df['is_feuilleton'].value_counts()}")

# Cleaning and making binary labels
# if merged --> y, if speech --> n, if essay --> n, if bio --> y
df["is_feuilleton"] = df["is_feuilleton"].map(categories_dict)

print(df['is_feuilleton'].value_counts())
logging.info("Merged categories into binary labels, y/n.")
logging.info(f"Cleaned categories: {df['is_feuilleton'].value_counts()}")

# %%

# PREPROCESS AND SAVE TEXT

# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^\D\s\[\]\"'“”‘’—\-]+", "", text)  # out w numbers, keep citation marks and brackets
    text = re.sub(r"\s+", " ", text).strip()  # remove multiple spaces
    return text

df['text'] = df["text"].apply(preprocess)
logging.info("Preprocessed text: lowercased, removed numbers, multiple spaces, etc.")

# save it
df.to_csv('data/cleaned_feuilleton.csv', sep="\t", index=False)
logging.info("Saved cleaned feuilleton df to data/cleaned_feuilleton.csv")


# %%

# NER

# Remove NER from texts and save seperately
no_ner_df = df.copy()

# NER
ner = pipeline(task='ner', 
               model='saattrupdan/nbailab-base-ner-scandi', 
               aggregation_strategy='first')
logging.info(f"NER model loaded: {ner}")

# function to subst. NER for [entity]
def remove_ner(text):
    """Removes named entities (PER,LOC) from text."""
    entities = ner(text)
    for entity in entities:
        if entity['score'] > 0.81 and entity['entity_group'] in {'PER', 'LOC'}:
            text = text.replace(entity['word'], '[entity]')  # Replace with placeholder
    return text

# Apply preprocessing
no_ner_df["text"] = no_ner_df["text"].apply(remove_ner).apply(preprocess)

# save
no_ner_df.to_csv("data/cleaned_feuilleton_noNER.csv", sep="\t", index=False)
logging.info("Saved cleaned feuilleton df with no NER to data/cleaned_feuilleton_noNER.csv")

# %%

# ADDITIONAL MERGE (IF NEEDED)

# Here, we can merge each article of one feuilleton to the same row (so all text is connected)
# function to groupby feuilleton_id and aggregate
def merge_by_group(group):
    first_row = group.iloc[0].copy()
    # collect all article_ids that belong to this feuilleton
    all_ids = group['article_id'].tolist()
    all_texts = group['text'].tolist()

    # merge article_ids and texts
    first_row['article_id'] = ','.join(all_ids)
    first_row['text'] = ' '.join(all_texts)  # join them (putting space)
    return first_row

# function to perform the merge
def merge_feuilleton(df, save_path=None):

    # fillna in feuilleton_id with x
    df['feuilleton_id'] = df['feuilleton_id'].fillna('x')
    print('number of x in feuilleton_id:', len(df[df['feuilleton_id'] == 'x']))

    # Keep only rows that have a feuilleton_id
    has_feuilleton = df.loc[df['feuilleton_id'] != 'x']
    print('has feuilleton ID:', len(has_feuilleton))
    logging.info(f"No of feuilleton with ID: {len(has_feuilleton)}")
    logging.info(f"No of feuilleton without ID: {len(df[df['feuilleton_id'] == 'x'])}")

    # apply the groupby-merge
    merged = has_feuilleton.groupby('feuilleton_id', sort=False).apply(merge_by_group).reset_index(drop=True)

    # keep the non-feuilleton rows (with empty feuilleton_id):
    no_feuilleton = df.loc[df['feuilleton_id'] == 'x']
    print('no feuilleton ID:', len(no_feuilleton))

    # merge feuilleton and non-feuilleton rows
    merged_df = pd.concat([merged, no_feuilleton], ignore_index=True)

    # Check how many are marked as feuilleton
    print(merged_df['is_feuilleton'].value_counts())
    logging.info(f"Feuilleton counts when merging: {merged_df['is_feuilleton'].value_counts()}")

    if save_path:
        # save the merged dataframe
        merged_df.to_csv(save_path, sep="\t", index=False)
        logging.info(f"Saved merged by feuilleton_ID dataframe to {save_path}")

    return merged_df

# Merge feuilleton IDs and save
merged_df = merge_feuilleton(df, save_path='data/merged_by_feuilletonIDs.csv')

# Merge feuilleton IDs with noNER text and save it as well
merged_no_ner = merge_feuilleton(no_ner_df, save_path='data/merged_by_feuilletonIDs_noNER.csv')


# STATS

# Compute word length of each text
merged_df['text_len'] = merged_df['text'].apply(lambda x: len(str(x).split()))

# Visualize the distribution
sns.set_style("whitegrid")
plt.figure(figsize=(5, 2))
sns.histplot(merged_df['text_len'], bins=30)
plt.title('Distribution of Textlen (in words)')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('figs/text_length_distribution_merged_feuilletons.png', dpi=300)

# print average len of fiction/nonfiction
print("avg no words, nonfiction", round(np.mean(merged_df.loc[merged_df['is_feuilleton'] == "n"]['text_len']),1))
print("avg no words, fiction", round(np.mean(merged_df.loc[merged_df['is_feuilleton'] == "y"]['text_len']),1))
logging.info(f"STATS: Avg no words, nonfiction: {round(np.mean(merged_df.loc[merged_df['is_feuilleton'] == 'n']['text_len']), 1)}")
logging.info(f"STATS: Avg no words, fiction: {round(np.mean(merged_df.loc[merged_df['is_feuilleton'] == 'y']['text_len']), 1)}")



