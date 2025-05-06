from pathlib import Path
import pandas as pd
import numpy as np
import spacy
import bz2
from collections import Counter


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
        token.ent_type_,
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
            "token_ent_type_",
            "sent_id" # Added sentence ID to get sent-len
        ],
    )
    return df_attributes


def save_spacy_df(spacy_df, filename, out_dir) -> None:
    Path(f"{out_dir}/spacy_books/").mkdir(exist_ok=True)
    spacy_df.to_csv(f"{out_dir}/spacy_books/{filename}_spacy.csv")


# PARENT function
def get_spacy_of_text(text, text_id, out_dir, model_name="da_core_news_sm"):
    nlp = spacy.load(model_name)
    # Process the text
    doc = nlp(text)
    
    doc_attributes = []
    # Assign a unique sent_id to each sentence and its tokens
    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            # Get the token attributes with the sentence ID
            doc_attributes.append(get_spacy_attributes(token, sent_id))
    
    # Create a DataFrame from the attributes
    spacy_df = create_spacy_df(doc_attributes)
    # Save the DataFrame to a CSV file
    save_spacy_df(spacy_df, filename=text_id, out_dir=out_dir)
    
    return spacy_df

# STYLISTICS

# stuff we get directly from the books

# get nominal verb ratio, ttr of nouns, and noun count
def get_nominal_verb_ratio_from_saved(text_id):
    # get spacy book for id
    spacy_df = pd.read_csv(f"data/spacy_books/{text_id}_spacy.csv")
    # Filter by POS
    nouns = spacy_df[spacy_df["token_pos_"] == "NOUN"]
    verbs = spacy_df[spacy_df["token_pos_"] == "VERB"]
    nominals = spacy_df[spacy_df["token_pos_"].isin(["PROPN", "ADJ"])]

    # Counts
    num_verb = len(verbs)
    num_nominal = len(nominals)
    num_nouns = len(nouns)

    # Avoid division by zero
    nominal_verb_ratio = (num_nominal + num_nouns) / num_verb if num_verb > 0 else 0
    noun_ttr = len(set(nouns)) / num_nouns if num_nouns > 0 else 0
    verb_ttr = len(set(verbs)) / num_verb if num_verb > 0 else 0

    return nominal_verb_ratio, num_nouns, noun_ttr, verb_ttr

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
        return 0
    
    # get the number of sentences
    num_sentences = len(sent_lengths)
    
    # return avg
    return sum(sent_lengths) / len(sent_lengths), num_sentences


# dependency distance
def calculate_dependency_distances(df, full_stop_indices):

    dependency_distances = []
    normalized_dependency_distances = []
    start_idx = 0

    for stop_idx in full_stop_indices:
        # Extract each sentence based on full stops
        sentence_df = df.loc[start_idx:stop_idx].copy()
        sentence_df_filtered = sentence_df[~sentence_df['token_is_punct'] & (sentence_df['token_pos_'] != 'SPACE')].copy()

        if not sentence_df_filtered.empty:
            # Find the root by using 'ROOT' in 'token_dep_' column
            root_token_row = sentence_df_filtered[sentence_df_filtered['token_dep_'] == 'ROOT']

            if not root_token_row.empty:

                root_idx = root_token_row['token_i'].iloc[0]

                # Calculating the root distance relative to the start of the sentence
                sentence_start_idx = sentence_df_filtered['token_i'].min()

                root_distance = root_idx - sentence_start_idx  # Adjusted root distance

                # Calculate MDD for the sentence
                sentence_df_filtered['dependency_distance'] = np.abs(sentence_df_filtered['token_i'] - sentence_df_filtered['token_head_i'])
                mdd = sentence_df_filtered['dependency_distance'].mean()

                dependency_distances.append(mdd)

                # Calculate sentence length excluding punctuation
                sentence_length = len(sentence_df_filtered)

                # Calculate NDD using the formula, avoiding division by zero or negative numbers
                if mdd > 0 and sentence_length > 0 and root_distance >= 0:
                    root_sentence_product = (root_distance + 1) * sentence_length  # +1 to avoid zero distance issue
                    if root_sentence_product > 0:
                        ndd = abs(np.log(mdd / np.sqrt(root_sentence_product)))
                        normalized_dependency_distances.append(ndd)

        # Move to the next sentence
        start_idx = stop_idx + 1

    # Calculate average NDD across all sentences
    average_ndd = np.mean(normalized_dependency_distances) if normalized_dependency_distances else None
    std_ndd = np.std(normalized_dependency_distances) if normalized_dependency_distances else None

    return average_ndd, std_ndd, np.mean(dependency_distances), np.std(dependency_distances)



# Entropy & compressibility

def compressrat(sents):
    """
    Calculates the BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
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

# to convert transformer scores to the same scale as the dictionary-based scores
def conv_scores(label, score, spec_lab):  # single label and score
    """
    Converts transformer-based sentiment scores to a uniform scale based on specified labels.
    """
    if len(spec_lab) == 2:
        if label == spec_lab[0]:  # "positive"
            return score
        elif label == spec_lab[1]:  # "negative"
            return -score  # return negative score

    elif len(spec_lab) == 3:
        if label == spec_lab[0]:  # "positive"
            return score
        elif label == spec_lab[1]:  # "neutral"
            return 0  # return 0 for neutral
        elif label == spec_lab[2]:  # "negative"
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
def split_long_sentence(text, tokenizer) -> list:
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


# get SA scores from xlm-roberta
def get_sentiment(text, model, tokenizer):
    """
    Gets the sentiment score for a given text, including splitting long sentences into chunks if needed.
    """
    # Check that the text is a string
    if not isinstance(text, str):
        print(f"Warning: Text is not a string for text: '{text}'. Skipping.")
        return None

    # Split the sentence into chunks if it's too long
    chunks = split_long_sentence(text, tokenizer)

    if len(chunks) == 0:
        print(f"Warning: No chunks created for text: '{text}'. Skipping.")
        return None

    elif len(chunks) == 1:
        # If the sentence is short enough, just use it as is
        chunks = [text]

    # Loop through the chunks and get sentiment scores for each
    sentiment_scores = []

    for chunk in chunks:
        # Get sentiment from the model
        sent = model(chunk)
        xlm_label = sent[0].get("label")
        xlm_score = sent[0].get("score")

        # Transform score to continuous scale
        xlm_converted_score = conv_scores(xlm_label, xlm_score, ["positive", "neutral", "negative"])
        sentiment_scores.append(xlm_converted_score)

    # Calculate the mean sentiment score from the chunks
    mean_score = sum(sentiment_scores) / len(sentiment_scores)

    return mean_score



# def conv_scores(label, score, spec_lab):  # single label and score

#     if len(spec_lab) == 2:
#         if label == spec_lab[0]:  # "positive"
#             return score
#         elif label == spec_lab[1]:  # "negative"
#             return -score # return negative score

#     elif len(spec_lab) == 3:
#         if label == spec_lab[0]:  # "positive"
#             return score
#         elif label == spec_lab[1]:  # "neutral"
#             return 0
#         elif label == spec_lab[2]:  # "negative"
#             return -score # return negative score

#     else:
#         raise ValueError("spec_lab must contain either 2 or 3 labels.")


# # split long sentences into chunks
# def split_long_sentence(text, tokenizer, model) -> list:
#     words = text.split()
#     parts = []
#     current_part = []
#     current_length = 0

#     max_length = tokenizer.model_max_length
#     if max_length > 2000: # sometimes, they default to ridiculously high values here, so we set a max
#         max_length = 512

#     for word in words:
#         tokens = tokenizer.encode(word)
#         seq_len = len(tokens)

#         if current_length + seq_len > max_length:
#             parts.append(current_part)
#             current_part = []
#             current_length = 0
#         current_part.append(word)
#         current_length += seq_len

#     if current_part:
#         parts.append(" ".join(current_part))
#     return parts


# # get SA scores from xlm-roberta
# def get_sentiment(text, model, tokenizer):
#     # check that the text is a string
#     if not isinstance(text, str):
#         print(f"Warning: Text is not a string for text: '{text}'. Skipping.")
#         return None
    
#     # If the sentence is too long, split it into chunks
#     chunks = split_long_sentence(text, tokenizer, model)

#     if len(chunks) == 0:
#         print(f"Warning: No chunks created for text: '{text}'. Skipping.")
#         return None
    
#     elif len(chunks) == 1:
#         # If the sentence is short enough, just use it as is
#         chunks = [text]

#     # Loop & get sentiment scores for each chunk
#     sentiment_scores = []

#     for chunk in chunks:
#         # Get sentiment from the model
#         sent = model(chunk)
#         xlm_label = sent[0].get("label")
#         xlm_score = sent[0].get("score")
        
#         # Transform score to continuous
#         xlm_converted_score = conv_scores(xlm_label, xlm_score, ["positive", "neutral", "negative"])
#         sentiment_scores.append(xlm_converted_score)
    
#     # Calculate the mean sentiment score from the chunks
#     mean_score = sum(sentiment_scores) / len(sentiment_scores)

#     return mean_score