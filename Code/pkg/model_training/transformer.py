"""
Inference for the fine-tuned DistilBERT sarcasm classifier.

Shared by the Streamlit app (app.py) and the CLI (Code/console.py) so that both agree on
model resolution, preprocessing, thresholds and attention extraction.

Deliberately free of any Streamlit import -- the CLI must not need it.
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data_processing.cleaning import clean_for_model

# Where a locally-trained model is written by the training notebook.
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / 'trained_models' / 'distilbert-sarcasm'

# Matches the max_length used at training time. Headlines are short; 48 word-pieces covers
# well past the 99th percentile.
MAX_LENGTH = 48

# The three-way band from the original console.py: anything between the two cut-offs is
# reported as Neutral rather than forced to a side.
SARCASTIC_THRESHOLD = 0.6
NON_SARCASTIC_THRESHOLD = 0.4


def resolve_model_id(explicit=None) -> str:
    """
    Decide which model to load, in order of precedence:
      1. an explicit id passed in (the app passes st.secrets['SARCASM_MODEL_ID'])
      2. the SARCASM_MODEL_ID environment variable
      3. a locally trained model under Code/pkg/trained_models/distilbert-sarcasm
    """
    if explicit:
        return str(explicit)

    from_env = os.environ.get('SARCASM_MODEL_ID')
    if from_env:
        return from_env

    if LOCAL_MODEL_DIR.is_dir():
        return str(LOCAL_MODEL_DIR)

    raise FileNotFoundError(
        'No sarcasm model available. Either train one with notebooks/train_distilbert.ipynb '
        '(it writes to ' + str(LOCAL_MODEL_DIR) + '), or point SARCASM_MODEL_ID at a '
        'HuggingFace Hub repo id.')


def load_model(model_id=None):
    """Load tokenizer + model. Returns (tokenizer, model, resolved_id)."""
    resolved = resolve_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    # `eager` attention is required to get attention matrices back at all -- the default
    # sdpa kernel does not expose them.
    model = AutoModelForSequenceClassification.from_pretrained(resolved, attn_implementation='eager')
    model.eval()
    return tokenizer, model, resolved


def _sarcastic_index(model) -> int:
    """Find which logit column means 'sarcastic', falling back to 1."""
    id2label = getattr(model.config, 'id2label', None) or {}
    for index, label in id2label.items():
        if 'sarcas' in str(label).lower() and 'non' not in str(label).lower() and 'not' not in str(label).lower():
            return int(index)
    return 1 if model.config.num_labels > 1 else 0


def _merge_wordpieces(tokens: list, weights: list):
    """
    Fold '##' continuation pieces back into whole words, summing their attention.

    Without this the heat-map shows fragments like 'thirty', '##some', '##thing', which is
    unreadable and splits one word's importance across several cells.
    """
    merged_tokens, merged_weights = [], []
    for token, weight in zip(tokens, weights):
        if token.startswith('##') and merged_tokens:
            merged_tokens[-1] += token[2:]
            merged_weights[-1] += weight
        else:
            merged_tokens.append(token)
            merged_weights.append(weight)
    return merged_tokens, merged_weights


def predict(text: str, tokenizer, model) -> dict:
    """
    Classify one piece of text.

    :return: dict with 'score' (probability of sarcasm), 'label', 'tokens', 'weights'
             (per-token attention scaled to 0-1 for display) and 'clean_text'.
    """
    clean_text = clean_for_model(text)

    encoded = tokenizer(clean_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        output = model(**encoded, output_attentions=True)

    probabilities = torch.softmax(output.logits, dim=-1)[0]
    score = float(probabilities[_sarcastic_index(model)])

    # Last layer, averaged over heads, then the row attended to FROM the [CLS] position --
    # [CLS] is the vector the classification head actually reads.
    attention = output.attentions[-1][0].mean(dim=0)[0]

    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    # Drop the structural tokens only. Notably NOT unk_token: filtering every special
    # token would make an out-of-vocabulary word vanish from the heat-map rather than
    # show up as [UNK], silently misrepresenting what the model actually read.
    structural = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    kept = [(token, float(weight)) for token, weight in zip(tokens, attention)
            if token not in structural]

    if kept:
        token_list, weight_list = _merge_wordpieces([t for t, _ in kept], [w for _, w in kept])
        # Raw attention sums to 1 across the sequence, so individual values are tiny and
        # would all render near-white. Scale by the max so the strongest token is fully
        # saturated and the rest are relative to it.
        largest = max(weight_list)
        weight_list = [w / largest for w in weight_list] if largest > 0 else weight_list
    else:
        token_list, weight_list = [], []

    return {'score': score, 'label': label_for_score(score), 'tokens': token_list,
            'weights': weight_list, 'clean_text': clean_text}


def label_for_score(score: float) -> str:
    if score > SARCASTIC_THRESHOLD:
        return 'Sarcastic'
    if score < NON_SARCASTIC_THRESHOLD:
        return 'Non-Sarcastic'
    return 'Neutral'
