import spacy
import pandas as pd
from warnings import filterwarnings
from pathlib import Path
from os.path import isfile

filterwarnings('ignore')
nlp = spacy.load('en_core_web_md')


def form_similar_sequences(text: str, n: int) -> list:
    tokens = nlp(text)
    token_tuples = [(tokens[i].text, i) for i in range(len(tokens)) if tokens[i].pos_ in {'ADJ', 'NOUN', 'VERB'}][:5]
    texts_only = [t.text for t in tokens]

    def most_similar(word: str):
        token = nlp.vocab[word]
        queries = [w for w in token.vocab if w.is_lower == token.is_lower and w.prob >= -15]
        by_similarity = sorted(queries, key=lambda w: token.similarity(w), reverse=True)[1:n]
        return [w.lower_ for w in by_similarity]

    modified_sentences = []
    while token_tuples:
        replace, position = token_tuples.pop(0)
        synonyms = most_similar(replace)
        subsentences = []
        for ind in range(len(synonyms)):
            new_sentence = ' '.join(texts_only[0:position] + [synonyms[ind]])

            if position < len(tokens) - 1:
                new_sentence += ' '
                new_sentence += ' '.join(texts_only[position+1:])

            subsentences.append(new_sentence)
        if subsentences:
            modified_sentences.append(subsentences)

    index = 0
    short_list = [text]
    while modified_sentences:
        index %= len(modified_sentences)
        current = modified_sentences[index]
        if current:
            next_sentence = current.pop(0)
            if next_sentence not in short_list:
                short_list.append(next_sentence)
        else:
            del modified_sentences[index]
        index += 1
        if len(short_list) >= n:
            break

    return short_list


def synonym_replacement(dataset_name: str, text_data: pd.Series, label_data: pd.Series, target_length) -> tuple:
    base_path = Path(__file__).parent#
    augmented_path = str(base_path / ('../datasets/' + dataset_name + "/processed_data/AugmentedData.csv"))

    if isfile(augmented_path):
        augmented_data = pd.read_csv(augmented_path, encoding="ISO-8859-1")
        return augmented_data['clean_data'][:target_length], augmented_data['sarcasm_label'][:target_length]

    duplicate_no = (target_length // len(text_data)) + 1
    list_of_texts = []
    list_of_labels = []

    for index in range(len(text_data)):
        print(index)
        text = text_data[index]
        label = label_data[index]

        for t in form_similar_sequences(text, duplicate_no):
            list_of_texts.append(t)
            list_of_labels.append(label)
        if len(list_of_labels) > target_length:
            break

    clean_data, sarcasm_labels = pd.Series(list_of_texts, name="clean_data")[:target_length], pd.Series(list_of_labels, name='sarcasm_label')[
                                                                 :target_length]
    new_data = pd.concat([sarcasm_labels, clean_data], axis=1)
    new_data.to_csv(
        path_or_buf=str(base_path / ('../datasets/' + dataset_name + '/processed_data/AugmentedData.csv')),
        index=False)

    return pd.Series(list_of_texts, name="clean_data")[:target_length], pd.Series(list_of_labels, name='sarcasm_label')[:target_length]
