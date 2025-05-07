from pathlib import Path
import pandas as pd
import numpy as np
import spacy
import bz2
from collections import Counter
import re


# SYNTACTICS

def get_spacy_attributes(token, sent_id):
    # Save all token attributes in a list
    token_attributes = [
        token.i,
        token.text,
        token.lemma_,
        token.is_punct,
        token.is_stop,
        token.morph,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.head.i,
        #token.ent_type_,
        sent_id # Added sentence ID to get sent-len
    ]

    return token_attributes


def create_spacy_df(doc_attributes: list) -> pd.DataFrame:
    df_attributes = pd.DataFrame(
        doc_attributes,
        columns=[
            "token_i",
            "token_text",
            "token_lemma_",
            "token_is_punct",
            "token_is_stop",
            "token_morph",
            "token_pos_",
            "token_tag_",
            "token_dep_",
            "token_head",
            "token_head_i",
            #"token_ent_type_",
            "sent_id" # Added sentence ID to get sent-len
        ],
    )
    return df_attributes


def save_spacy_df(spacy_df, filename, out_dir) -> None:
    Path(f"{out_dir}/spacy_books/").mkdir(exist_ok=True)
    spacy_df.to_csv(f"{out_dir}/spacy_books/{filename}_spacy.csv")


nlp = spacy.load("da_core_news_sm")

def get_spacy_of_text(text, text_id, out_dir):
    # Check if text is empty
    if not text:
        raise ValueError(f"Text with ID {text_id} is empty!")

    # Process text with spacy
    doc = nlp(text)

    # Initialize sentence id and document attributes
    sent_id = 0
    doc_attributes = []

    # Use Spacy's built-in sentence segmentation
    for sent in doc.sents:
        for token in sent:
            doc_attributes.append(get_spacy_attributes(token, sent_id))
        sent_id += 1  # Increment sentence id after each sentence

    # Create a DataFrame from the parsed attributes
    spacy_df = create_spacy_df(doc_attributes)

    # Save the DataFrame to a file
    save_spacy_df(spacy_df, filename=text_id, out_dir=out_dir)

    # Return the DataFrame for further use
    return spacy_df

# STYLISTICS

# stuff we get directly from the books

# get nominal verb ratio, ttr of nouns, and noun count
def get_pos_derived_features(text_id):
    df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")

    # Pre-filter
    pos = df["token_pos_"]
    morph = df["token_morph"].fillna("")

    nouns = df[pos == "NOUN"]
    verbs = df[pos == "VERB"]
    nominals = df[pos.isin(["PROPN", "ADJ"])]
    propn_pers = df[(pos == "PRON") & morph.str.contains("PronType=Prs")]

    # Pre-compute counts
    num_nouns = nouns.shape[0]
    num_verbs = verbs.shape[0]
    num_nominals = nominals.shape[0]
    num_pronouns = propn_pers.shape[0]
    total_tokens = df.shape[0]

    # Ratios
    nominal_verb_ratio = (num_nominals + num_nouns) / num_verbs if num_verbs else np.nan
    noun_ttr = nouns["token_lemma_"].nunique() / num_nouns if num_nouns else np.nan
    verb_ttr = verbs["token_lemma_"].nunique() / num_verbs if num_verbs else np.nan
    personal_pronoun_ratio = num_pronouns / total_tokens if total_tokens else np.nan

    return nominal_verb_ratio, num_nouns, noun_ttr, verb_ttr, personal_pronoun_ratio

# worldlength
def avg_wordlen(text_id: str) -> float:
    try:
        # Load processed SpaCy dataframe
        spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")

        # Filter out punctuation, spaces, and numbers
        words = spacy_df[~spacy_df["token_pos_"].isin(["PUNCT", "SPACE", "NUM"])]

        # Handle case of no valid words
        if words.empty:
            return np.nan

        # Compute average word length
        word_lengths = [len(word) for word in words["token_text"] if isinstance(word, str)]
        return sum(word_lengths) / len(word_lengths) if word_lengths else np.nan

    except FileNotFoundError:
        print(f"File for {text_id} not found.")
        return np.nan


# sentence length
def avg_sentlen(text_id) -> float:
    # get saved spacy_book
    spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")
    
    # Group by sentence ID (sent_id column in the spacy_df)
    sentence_groups = spacy_df.groupby("sent_id")
    
    # Calculate sentence lengths
    sent_lengths = [len(group) for _, group in sentence_groups]
    if not sent_lengths:
        return np.nan
    
    # get the number of sentences
    num_sentences = len(sent_lengths)
    
    # return avg
    return sum(sent_lengths) / len(sent_lengths), num_sentences


# dependency distances
def calculate_dependency_distances(text_id):
    spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")

    dependency_distances = []
    normalized_dependency_distances = []

    for sent_id, sentence_df in spacy_df.groupby("sent_id"):
        sentence_df_filtered = sentence_df[
            ~sentence_df['token_is_punct'] & (sentence_df['token_pos_'] != 'SPACE')
        ].copy()

        if not sentence_df_filtered.empty:
            root_token_row = sentence_df_filtered[sentence_df_filtered['token_dep_'] == 'ROOT']

            if not root_token_row.empty:
                root_idx = root_token_row['token_i'].iloc[0]
                sentence_start_idx = sentence_df_filtered['token_i'].min()
                root_distance = root_idx - sentence_start_idx

                sentence_df_filtered['dependency_distance'] = (
                    np.abs(sentence_df_filtered['token_i'] - sentence_df_filtered['token_head_i'])
                )

                mdd = sentence_df_filtered['dependency_distance'].mean()
                dependency_distances.append(mdd)

                sentence_length = len(sentence_df_filtered)

                if mdd > 0 and sentence_length > 0 and root_distance >= 0:
                    root_sentence_product = (root_distance + 1) * sentence_length
                    if root_sentence_product > 0:
                        ndd = abs(np.log(mdd / np.sqrt(root_sentence_product)))
                        normalized_dependency_distances.append(ndd)

    average_ndd = np.mean(normalized_dependency_distances) if normalized_dependency_distances else np.nan
    std_ndd = np.std(normalized_dependency_distances) if normalized_dependency_distances else np.nan
    avg_mdd = np.mean(dependency_distances) if dependency_distances else np.nan
    std_mdd = np.std(dependency_distances) if dependency_distances else np.nan

    return average_ndd, std_ndd, avg_mdd, std_mdd


# Entropy & compressibility

# we want to use the sents of the spacy_df
def compressrat(text_id):
    """
    Calculates the BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
    # get the sents
    spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")
    # get the sentences
    sentences_df = spacy_df.groupby("sent_id")
    sents = list(sentences_df["token_text"].apply(lambda x: " ".join(x)))

    # hopefully skipping the first few that can be noisy
    if len(sents) > 50:
        selection = sents[10:50]
    elif len(sents) > 40:
        selection = sents[:40]
    else:
        # if less than 40 sentences, just take all
        selection = sents
    
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return bzipr


def cal_entropy(base, log_n, transform_prob):
    entropy = 0
    for count in transform_prob.values():
        if count > 0:  # Avoid log of zero
            probability = count / sum(transform_prob.values())
            entropy -= probability * (log(probability, base) - log_n)
    return entropy


def text_entropy(words, base=2, asprob=True):
    total_len = len(words) - 1
    bigram_transform_prob = Counter()
    word_transform_prob = Counter()

    # Loop through each word and calculate the probabilities
    for i, word in enumerate(words):
        if i > 0:
            bigram_transform_prob[(words[i-1], word)] += 1
            word_transform_prob[word] += 1

    if asprob:
        return word_transform_prob, bigram_transform_prob

    log_n = log(total_len, base)
    bigram_entropy = cal_entropy(base, log_n, bigram_transform_prob)
    word_entropy = cal_entropy(base, log_n, word_transform_prob)

    return bigram_entropy / total_len, word_entropy / total_len


# SA functions

# OBS, we want to make this sentence-based

# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(label, score, spec_lab):  # single label and score
    """
    Converts transformer-based sentiment scores to a uniform scale based on specified labels.
    We need to lowercase since sometimes, a model will have as label "Neutral" or "neutral" or "NEUTRAL"
    """
    if len(spec_lab) == 2:
        if label.lower() == spec_lab[0]:  # "positive"
            return score
        elif label.lower() == spec_lab[1]:  # "negative"
            return -score  # return negative score

    elif len(spec_lab) == 3:
        if label.lower() == spec_lab[0]:  # "positive"
            return score
        elif label.lower() == spec_lab[1]:  # "neutral"
            return 0  # return 0 for neutral
        elif label.lower() == spec_lab[2]:  # "negative"
            return -score  # return negative score

    else:
        raise ValueError("spec_lab must contain either 2 or 3 labels.")


# Function to find the maximum allowed tokens for the model
def find_max_tokens(tokenizer):
    """
    Determines the maximum token length for the tokenizer, ensuring it doesn't exceed a reasonable limit.
    """
    max_length = tokenizer.model_max_length
    if max_length > 2000:  # sometimes, they default to ridiculously high values, so we set a max
        max_length = 512
    return max_length


# split long sentences into chunks
def split_text_to_chunks(text, tokenizer) -> list:
    """
    Splits long sentences into chunks if their token length exceeds the model's maximum length.
    """
    words = text.split()
    parts = []
    current_part = []
    current_length = 0

    max_length = find_max_tokens(tokenizer)

    for word in words:
        # Encode word and get the token length
        tokens = tokenizer.encode(word)
        seq_len = len(tokens)

        # Check if adding this word would exceed max length
        if current_length + seq_len > max_length:
            parts.append(" ".join(current_part))  # Append the current part as a chunk
            current_part = [word]  # Start a new part with the current word
            current_length = seq_len  # Reset the current length to the length of the current word
        else:
            current_part.append(word)  # Add the word to the current chunk
            current_length += seq_len  # Update the current length

    # Append any remaining part as a chunk
    if current_part:
        parts.append(" ".join(current_part))

    return parts


# get SA scores from model
def get_sentiment(text_id, model, tokenizer):
    """
    Gets the sentiment score per sentence in a text, including splitting long sentences into chunks if needed.
    """
    # Load spaCy-parsed sentences
    spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")
    sentences_df = spacy_df.groupby("sent_id")
    sents = list(sentences_df["token_text"].apply(lambda x: " ".join(x)))

    if not sents:
        print(f"Warning: No sentences found for text: '{text_id}'. Skipping.")
        return None

    sentiment_scores = []

    for sent in sents:
        # check empty
        if not sent:
            print(f"Warning: Empty sentence found in '{text_id}'. Skipping.")
            continue

        chunks = split_text_to_chunks(sent, tokenizer)

        if len(chunks) == 0:
            print(f"Warning: No chunks created for a sentence in '{text_id}'. Skipping.")
            continue
        elif len(chunks) > 1:
            print(f"Warning: Sentence split into {len(chunks)} chunks for text: '{text_id}'.")

        for chunk in chunks:
            result = model(chunk)
            if not result:
                continue

            model_label = result[0].get("label")
            model_score = result[0].get("score")

            converted_score = conv_scores(model_label, model_score, ["positive", "neutral", "negative"])
            sentiment_scores.append(float(converted_score))

    return sentiment_scores
