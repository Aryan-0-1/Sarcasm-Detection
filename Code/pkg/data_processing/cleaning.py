import re

_nlp = None


def _get_nlp():
    """
    Load spaCy on first use rather than at import time.

    Only remove_stopwords() needs spaCy, and that path is off by default. Importing the
    model at module scope pulled ~40MB into every consumer of this module -- including the
    Streamlit app, which does no stopword removal at all.
    """
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load('en_core_web_md')
    return _nlp


def data_cleaning(data_string: str, rm_urls=True, rm_punc=True, lower=True, rm_numbers=True,
                  rm_dp_wspc=True, rm_stop=False, normalise_social=True):
    """
    Given data as a string and a set of flags, clean data accordingly
    :param data_string:
    :param rm_urls: remove urls
    :param rm_punc: remove punctuation, optional parameter list can be provided of the punctuation to remove
    :param lower: convert text to lowercase
    :param rm_numbers: remove numbers
    :param rm_dp_wspc: remove duplicate whitespaces -> converting to a single whitespace
    :param rm_stop: remove stopwords (requires spaCy + en_core_web_md)
    :param normalise_social: fold user mentions and hashtags, and drop the #sarcasm label tag
    :return: cleaned string
    """
    def remove_urls(text: str) -> str:
        return re.sub(r'http\S+', '', text)  # remove URLs

    def normalise_social_text(text: str) -> str:
        # Split out of remove_punctuation so that punctuation can be preserved without
        # also preserving '#sarcasm' -- which is the label on the ptacek corpus and would
        # hand the model the answer.
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'([@][\w_-]+)', '<user>', text)  # remove user mentions
        text = text.replace("#sarcasm", ' ')    # remove #sarcasm for ptacek dataset
        text = text.replace("#not", 'not')  # replace #not
        return re.sub(r'([#][\w_-]+)', ' ', text)  # remove hash tags

    def remove_punctuation(text: str) -> str:
        banned_punctuation = set([char for char in '#$%&()*+-/:;<>[]^_`{|}~'])
        return ''.join(ch for ch in text if ch not in banned_punctuation)  # remove punctuation marks

    def remove_numbers(text: str) -> str:
        return re.sub("[0-9]", "", text)

    def remove_duplicate_whitespaces(text: str) -> str:
        return ' '.join(text.split())  # remove duplicate whitespaces

    def remove_stopwords(text: str) -> str:
        words = [token.text for token in _get_nlp()(text) if not token.is_stop]
        return ' '.join(words)

    if rm_urls:
        data_string = remove_urls(data_string)  # remove URLs

    data_string += ' '  # add space so that user mentions are detected

    if lower:
        data_string = data_string.lower()  # convert to lowercase

    if normalise_social:
        data_string = normalise_social_text(data_string)

    if rm_punc:
        data_string = remove_punctuation(data_string)  # remove punctuation

    if rm_numbers:
        data_string = remove_numbers(data_string)  # convert to lowercase

    if rm_dp_wspc:
        data_string = remove_duplicate_whitespaces(data_string)  # remove duplicate whitespaces

    data_string = data_string.strip()

    if rm_stop:
        data_string = remove_stopwords(data_string)  # remove stop words
    return data_string


def clean_for_model(x: str) -> str:
    """
    The single preprocessing preset used by the transformer model.

    Both the training notebook and the inference path call this, so train-time and
    serve-time text can never drift apart.

    Punctuation and digits are deliberately KEPT. The 2020 preset stripped both, but '!',
    '?' and quote marks carry real sarcasm signal, and a WordPiece tokeniser handles them
    natively -- there is nothing to gain by throwing them away. Lowercasing is kept
    because the corpus is already lowercase and the checkpoint is `-uncased`; stating it
    explicitly means free-text input matches the training distribution.
    """
    return data_cleaning(x, rm_urls=True, rm_punc=False, lower=True, rm_numbers=False,
                         rm_dp_wspc=True, rm_stop=False, normalise_social=True)


def apply_params(x: str):
    """Legacy (2020) preset: aggressive stripping, used by the pre-transformer pipeline."""
    settings = {
        "remove_urls": True,
        "remove_punctuation": True,
        "lowercase": True,
        "remove_numbers": True,
        "remove_duplicate_whitespaces": True,
        "remove_stopwords": True}

    return data_cleaning(x, settings["remove_urls"], settings["remove_punctuation"],
                         settings["lowercase"], settings["remove_numbers"], settings["remove_duplicate_whitespaces"],
                         settings['remove_stopwords'])
