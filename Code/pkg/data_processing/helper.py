from ..vectors.create_features import extract_features
from ..vectors.create_vectors import compute_vectors
from .cleaning import data_cleaning
import pandas as pd
import os.path
from random import randint
import spacy
from ..analysis.evaluate_data import feature_evaluate
from pathlib import Path

nlp = spacy.load('en_core_web_md')


def get_clean_data_col(data_frame: pd.DataFrame, path_to_dataset_root: str, re_clean: bool) -> pd.DataFrame:
    base_path = Path(__file__).parent

    """
    Retrieve the column of cleaned data -> either by cleaning the raw data, or by retrieving pre-cleaned data
    :param data_frame: data_frame containing a 'text_data' column -> this is the raw textual data
    :param path_to_dataset_root: path to where the dataset is stored
    :param re_clean: boolean flag -> set to True to have the data cleaned again
    :param extend_path: choose to read the cleaned data at an extended path -> this is not the default clean data
    :return: a pandas DataFrame containing cleaned data
    """
    if re_clean:
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-cleaned data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            # This could potentially overwrite pre-cleaned text if triggered accidentally
            # The process of cleaning data can take a while, so -> proceed with caution
            print('RE-CLEANING ... PROCEED WITH CAUTION!')
            # exit()  # comment this line if you would still like to proceed
            data_frame['clean_data'] = data_frame['text_data'].apply(data_cleaning)
            extend_path = '' if not os.path.isfile((base_path / (path_to_dataset_root + "/processed_data/CleanData.csv")).resolve()) else \
                ''.join([randint(0, 9) for _ in range(0, 8)])
            data_frame['clean_data'].to_csv(
                path_or_buf=str(base_path / (path_to_dataset_root + "/processed_data/CleanData.csv")), index=False,
                header=['clean_data'])
    return pd.read_csv(str(base_path / (path_to_dataset_root + "/processed_data/CleanData.csv")), encoding="ISO-8859-1")


def get_vector_col(data_frame: pd.DataFrame, path_to_root, vector_type: str) -> list:
    base_path = Path(__file__).parent
    """
    Given a vector type, retrieve pre-computed vectors for data, or compute them and return them as a list
    :param data_frame: DataFrame containing 'clean_data' column
    :param path_to_root: path to dataset root -> used for file naming
    :param vector_type: a string representing the type of vectors to produce
    :return: a list of vectors
    """
    valid_vector_types = {'elmo', 'bag_of_words', 'tf_idf', 'glove'}
    vector_type = vector_type.lower().strip()

    if vector_type not in valid_vector_types:
        raise TypeError('Invalid vector type "' + vector_type + '"')

    if not os.path.isfile(str(base_path / (path_to_root + "/processed_data/Vectors/" + vector_type + ".pckl"))):
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-vectorised data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            print('RE-VECTORIZING ... PROCEED WITH CAUTION!')
            # exit()  # comment this line if you would still like to proceed
            if vector_type in {'bag_of_words', 'tf_idf', 'glove'}:
                data_frame['token_data'] = data_frame['clean_data'].apply(
                    lambda x: " ".join([token.text for token in nlp(x)]))
                compute_vectors(path_to_root, data_frame,  vector_type)
            elif vector_type == 'elmo':
                compute_vectors(path_to_root, data_frame, vector_type)
    return pd.read_pickle(str(base_path / (path_to_root + "/processed_data/Vectors/" + vector_type + ".pckl")))


def get_feature_col(data_frame: pd.DataFrame, path_to_root: str, feature_type: str):
    base_path = Path(__file__).parent
    pd.read_pickle(str(base_path / (path_to_root + "/processed_data/Features/" + feature_type + ".pckl")))

    valid_feature_types = {'sentiment', 'punctuation', 'topic_model'}
    feature_type = feature_type.lower().strip()

    if feature_type not in valid_feature_types:
        raise TypeError('Invalid feature type "' + feature_type + '"')

    if not os.path.isfile(str(base_path / (path_to_root + "/processed_data/Features/" + feature_type + ".pckl"))):
        input_data = ''
        while not input_data:
            input_data = input('\nWARNING - This action could overwrite pre-extracted data: proceed? y / n\n')
            input_data = input_data.strip().lower() if input_data.strip().lower() in {'y', 'n'} else ''

        if input_data == 'y':
            print('RE-VECTORIZING ... PROCEED WITH CAUTION!')
            # exit()  # comment this line if you would still like to proceed
            extract_features(path_to_root, data_frame,  feature_type)
    return pd.read_pickle(str(base_path / (path_to_root + "/processed_data/Features/" + feature_type + ".pckl")))


def combine_features(feature_columns: list) -> pd.Series:
    if not feature_columns:
        raise ValueError('List of features cannot be empty')
    elif len(feature_columns) == 1:
        return feature_columns.pop()

    def add(l1, l2) -> list:
        if type(l1) != list:
            l1 = l1.tolist()
        elif type(l2) != list:
            l2 = l2.tolist()
        return l1 + l2

    feature_col = feature_columns[0]
    for index in range(1, len(feature_columns)):
        feature_col = feature_col.combine(feature_columns[index], add)

    return feature_col

def get_dataset_name(dataset_number: int) -> str:
    dataset_paths = ["amazon_reviews", "news_headlines", "ptacek"]
    return dataset_paths[dataset_number]


def prepare_data(dataset_number: int, vector_type: str, feature_list: list, set_size=None):
    if not vector_type and not feature_list:
        raise ValueError('Vector or feature must exist')

    base_path = Path(__file__).parent
    dataset_paths = ["amazon_reviews", "news_headlines", "ptacek"]
    dataset_name = get_dataset_name(dataset_number)
    print('\nSelected dataset: ' + dataset_name)

    path_to_dataset_root = "../datasets/" + dataset_name

    # Choose a dataset from the list of valid data sets
    #path_to_dataset_root = dataset_paths[dataset_number]

    if dataset_number < 0 or dataset_number >= len(dataset_paths):
        raise ValueError('Dataset number must be between 0 and ' + str(len(dataset_paths) - 1))

    if set_size is not None:
        # Read in raw data
        data = pd.read_csv((base_path / (path_to_dataset_root + "/processed_data/OriginalData.csv")).resolve(),
                           encoding="ISO-8859-1")[:set_size]
        # Clean data, or retrieve pre-cleaned data
        data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)[:set_size]

        if vector_type:
            data['vector'] = get_vector_col(data, path_to_dataset_root, vector_type)[:set_size]
    else:
        # Read in raw data
        data = pd.read_csv((base_path / (path_to_dataset_root + "/processed_data/OriginalData.csv")).resolve(), encoding="ISO-8859-1")

        # Clean data, or retrieve pre-cleaned data
        data['clean_data'] = get_clean_data_col(data, path_to_dataset_root, False)

        if vector_type:
            data['vector'] = get_vector_col(data, path_to_dataset_root, vector_type)

    print('Vector Type(s): ' + vector_type)


    # Create features, or retrieve pre-generated features
    # remember, punctuation features require raw data

    if len(feature_list) > 0:
        print('Feature Type(s): ' + ','.join(feature_list))
        feature_series = [pd.Series(get_feature_col(data, path_to_dataset_root, feature_type)) for feature_type in feature_list]
        data['feature'] = combine_features(feature_series)
        feature_evaluate(data)
        if vector_type:
            data['vector'] = combine_features([data['vector'], data['feature']])
        else:
            data['vector'] = data['feature']
    else:
        print('Feature Type(s): None')

    return dataset_name, data['sarcasm_label'], data['vector'].apply(pd.Series), data['clean_data'], data['text_data']
