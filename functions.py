from pathlib import Path
import pandas as pd
import spacy
import numpy as np
import bz2
from lexical_diversity import lex_div as ld
from antropy import app_entropy

# %%

DATA_DIR = Path("data")
SPACY_DIR = DATA_DIR / "spacy_books"
SPACY_MODEL = "da_core_news_sm"

SPACY_DIR.mkdir(parents=True, exist_ok=True)


# --- SPACY ---
nlp = spacy.load(SPACY_MODEL)

def get_spacy_attributes(token, sent_id):
    return [
        token.i, token.text, token.lemma_, token.is_punct, token.is_stop,
        token.morph, token.pos_, token.tag_, token.dep_,
        token.head, token.head.i, sent_id
    ]

def create_spacy_df(doc_attributes):
    columns = [
        "token_i", "token_text", "token_lemma_", "token_is_punct",
        "token_is_stop", "token_morph", "token_pos_", "token_tag_",
        "token_dep_", "token_head", "token_head_i", "sent_id"
    ]
    return pd.DataFrame(doc_attributes, columns=columns)

def process_text(text, text_id):
    if not text:
        raise ValueError(f"Empty text: {text_id}")

    doc = nlp(text)
    doc_attrs = [
        get_spacy_attributes(token, i)
        for i, sent in enumerate(doc.sents)
        for token in sent
    ]

    df = create_spacy_df(doc_attrs)
    df.to_csv(SPACY_DIR / f"{text_id}_spacy.csv", index=False)
    return df

def load_spacy_df(text_id):
    return pd.read_csv(SPACY_DIR / f"{text_id}_spacy.csv")


# --- SYNTACTICS & STYLISTICS ---

def read_spacy_df(text_id, base_dir="data"):
    path = f"{base_dir}/spacy_books/{text_id}_spacy.csv"
    return pd.read_csv(path)

def get_pos_derived_features(text_id):
    df = read_spacy_df(text_id)
    pos = df["token_pos_"]
    morph = df["token_morph"].fillna("")

    words = df[~df["token_pos_"].isin(["PUNCT", "SPACE", "NUM"])]
    nouns = df[pos == "NOUN"]
    verbs = df[pos == "VERB"]
    passive_and_active_verbs = df[(pos == "VERB") & (morph.str.contains("Voice=Act") | morph.str.contains("Voice=Pass"))]
    passive_verbs = df[(pos == "VERB") & (morph.str.contains("Voice=Pass"))]
    active_verbs = df[(pos == "VERB") & (morph.str.contains("Voice=Act"))]
    past_tense_verbs = df[(pos == "VERB") & (morph.str.contains("Tense=Past"))]
    present_tense_verbs = df[(pos == "VERB") & (morph.str.contains("Tense=Pres"))]
    nominals = df[pos.isin(["PROPN", "ADJ"])]
    propn_pers = df[(pos == "PRON") & morph.str.contains("PronType=Prs")]
    function_words = df[pos.isin(["ADP", "CCONJ", "SCONJ", "AUX", "PART"])]
    of_like = df[(pos == "ADP") & df["token_text"].str.lower().isin(["af"])]
    that_like = df[(pos == "SCONJ") & df["token_text"].str.lower().isin(["at"])]

    return {
        "nominal_verb_ratio": (len(nominals) + len(nouns)) / len(verbs) if len(verbs) else np.nan,
        "noun_count": len(nouns),
        "msttr": ld.msttr(words["token_text"].tolist(), window_length=100),
        "noun_ttr": nouns["token_lemma_"].nunique() / len(nouns) if len(nouns) else np.nan,
        "verb_ttr": verbs["token_lemma_"].nunique() / len(verbs) if len(verbs) else np.nan,
        "personal_pronoun_ratio": len(propn_pers) / len(df) if len(df) else np.nan,
        "function_word_ratio": len(function_words) / len(df) if len(df) else np.nan,
        "of_ratio": len(of_like) / len(df) if len(df) else np.nan,
        "that_ratio": len(that_like) / len(df) if len(df) else np.nan,
        # "active_ratio": len(active_verbs) / len(verbs) if len(passive_verbs) else np.nan,
        # "passive_ratio": len(passive_verbs) / len(verbs) if len(passive_verbs) else np.nan,
        "past_tense_ratio": len(past_tense_verbs) / len(verbs) if len(past_tense_verbs) else np.nan,
        "present_tense_ratio": len(present_tense_verbs) / len(verbs) if len(present_tense_verbs) else np.nan,
        "passive_ratio": len(active_verbs) / len(passive_and_active_verbs) if len(passive_and_active_verbs) else np.nan,
        "active_ratio": len(passive_verbs) / len(passive_and_active_verbs) if len(passive_and_active_verbs) else np.nan,
    }


def avg_wordlen(text_id):
    df = read_spacy_df(text_id)
    words = df[~df["token_pos_"].isin(["PUNCT", "SPACE", "NUM"])]
    word_lengths = [len(w) for w in words["token_text"] if isinstance(w, str)]
    return np.mean(word_lengths) if word_lengths else np.nan

def avg_sentlen(text_id):
    df = read_spacy_df(text_id)
    sent_lens = df.groupby("sent_id").size()
    return sent_lens.mean(), len(sent_lens)

def calculate_dependency_distances(text_id):
    df = read_spacy_df(text_id)
    dds, ndds = [], []

    for _, sent in df.groupby("sent_id"):
        sent = sent[~sent['token_is_punct'] & (sent["token_pos_"] != "SPACE")].copy()
        if sent.empty:
            continue
        root = sent[sent["token_dep_"] == "ROOT"]
        if root.empty:
            continue
        root_i = root["token_i"].iloc[0]
        start_i = sent["token_i"].min()
        root_dist = root_i - start_i

        sent["dd"] = abs(sent["token_i"] - sent["token_head_i"])
        mdd = sent["dd"].mean()
        dds.append(mdd)

        slen = len(sent)
        if mdd > 0 and slen > 0 and root_dist >= 0:
            prod = (root_dist + 1) * slen
            ndd = abs(np.log(mdd / np.sqrt(prod))) if prod else np.nan
            ndds.append(ndd)

    return {
        "avg_ndd": np.mean(ndds) if ndds else np.nan,
        "std_ndd": np.std(ndds) if ndds else np.nan,
        "avg_mdd": np.mean(dds) if dds else np.nan,
        "std_mdd": np.std(dds) if dds else np.nan,
    }

def compressrat(text_id):
    df = read_spacy_df(text_id)
    sentences_df = df.groupby("sent_id")
    sents = list(sentences_df["token_text"].apply(lambda x: " ".join([str(t) for t in x if isinstance(t, str)])))

    if len(sents) > 50:
        selection = sents[10:50]
    else:
        selection = sents[:40]

    as_str = " ".join(selection)
    encoded = as_str.encode("utf-8")
    compressed = bz2.compress(encoded, compresslevel=9)
    return len(encoded) / len(compressed) if len(compressed) else np.nan


# --- SENTIMENT ANALYSIS ---

def get_apen(sentiment_list):
    """
    Computes the approximate entropy of a list of sentiment scores.
    """
    if len(sentiment_list) < 10: # should probably be 20
        return np.nan
    return app_entropy(sentiment_list, order=2)


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
    sents = list(sentences_df["token_text"].apply(lambda x: " ".join(str(token) for token in x)))

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
            print(f"Note: Sentence split into {len(chunks)} chunks for text: '{text_id}'.")

        for chunk in chunks:
            result = model(chunk)
            if not result:
                continue

            model_label = result[0].get("label")
            model_score = result[0].get("score")

            converted_score = conv_scores(model_label, model_score, ["positive", "neutral", "negative"])
            sentiment_scores.append(float(converted_score))

    return sentiment_scores
