import sys
import pathlib; base_path = pathlib.Path(__file__).parent.parent.resolve(); sys.path.insert(1, str(base_path))
import spacy
from Code.pkg.model_training.DLmodels import get_dl_results
from Code.pkg.model_training.MLmodels import get_ml_results
from itertools import combinations

nlp = spacy.load('en_core_web_md')


# ------------------------- HELPER FUNCTIONS --------------------------
def run_ml(models: list, vectors: list, data_sets: list, features: list) -> None:
    """
        Given lists of machine learning models, vectors, features and dataset numbers, train, or collect the
        pre-trained model, then return results
        :param models: list of abbreviated model names
        :param vectors: list of vector types
        :param data_sets: list of dataset numbers (ints)
        :param features: list of feature types
        :return: NoneType
    """
    feature_combos = []
    for r in range(0, len(features) + 1):
        combo = combinations(features, r)
        feature_combos += [list(c) for c in combo]

    for dataset in data_sets:  # 0 - AMAZON, 1 - NEWS, 2 - TWEETS
        for model in models:
            for vector in vectors:
                for feature in feature_combos:
                    if vector or len(feature) != 0:
                        get_ml_results(model, vector, feature, dataset)


def run_dl(models: list, vectors: list, data_sets: list) -> None:
    """
    Given lists of deep learning models, vectors and dataset numbers, train, or collect the pre-trained model,
    and return results
    :param models: list of abbreviated model names
    :param vectors: list of vector types
    :param data_sets: list of dataset numbers (ints)
    :return: NoneType
    """
    for dataset in data_sets:  # 0 - AMAZON, 1 - NEWS, 2 - TWEETS
        for model in models:
            for vector in vectors:
                get_dl_results(model, dataset, vector)
# ---------------------------------------------------------------------


if __name__ == '__main__':
    # --------------------------
    model_list = ['cnn']
    vector_list = ['elmo']
    feature_list = []
    dataset_list = [2]
    # --------------------------
    # valid vector-types: 'bag_of_words', 'tf_idf', 'glove', 'elmo'
    # valid feature-types: 'sentiment', 'punctuation', 'topic_model'
    # valid ml models: 'svm', 'log_reg', 'rfc', 'n_bayes', 'knn'
    # valid dl models: 'cnn', 'dcnn' 'lstm', 'bi-lstm', 'vanilla-rnn', 'gru', 'attention-lstm', 'bi-gru'
    # --------------------------
    # run_ml(model_list, vector_list, dataset_list, feature_list)
    run_dl(model_list, vector_list, dataset_list)