from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from ..data_processing.helper import prepare_data
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import isfile
import numpy as np
import pickle
import time
from pathlib import Path

# ------------------------------------ Model dictionaries --------------------------------------
# List of supported models
models = {"svm": SVC(gamma='auto', C=10, kernel='linear'),
          "log_reg": LogisticRegression(C=10, max_iter=300),
          "rfc": RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt'),
          "n_bayes": GaussianNB(),
          "knn": KNeighborsClassifier(n_neighbors=5)
          }

# Full names of supported models for printing to the console
full_names = {"svm": "Support Vector Machine",
              "log_reg": "Logistic Regression",
              "rfc": "Random Forest Classifier",
              "n_bayes": "Gaussian Naive Bayes",
              "knn": "K-Nearest Neighbour"
              }


# -------------------------------------- Main Functions ---------------------------------------
def get_model(model_name: str) -> tuple:
    """
    Given an abbreviated model name as a string, return the model's full name and the corresponding unfit classifier
    :param model_name: a string containing an abbreviated model name - this is the model to be fetched n.b. for a full
    list of valid names, view the keys of the full_names dictionary
    """
    try:
        return full_names[model_name], models[model_name]
    except KeyError:
        print('Invalid model name')


def evaluate_model(trained_model, testing_data: pd.Series, testing_labels: pd.Series) -> None:
    """
    Given a trained model, and some testing data and labels, calculate the F1 score of the model
    :param trained_model: a sci-kit learn classifier, trained on corpus of vectors
    :param testing_data: data
    :param testing_labels: a string containing an abbreviated model name - this is the model to be fetched n.b. for a full
    :return: NoneType
    """
    y_pred = trained_model.predict(np.array(testing_data))
    f1 = f1_score(np.array(testing_labels), np.array(y_pred))
    precision = precision_score(np.array(testing_labels), np.array(y_pred))
    recall = recall_score(np.array(testing_labels), np.array(y_pred))
    mcc = matthews_corrcoef(np.array(testing_labels), np.array(y_pred))
    print('F1 Score: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)

    print('MCC: ', mcc)


def get_ml_results(model_name: str, vector_type: str, feature_list: list, dataset_number: int) -> None:
    """
    Train and evaluate a machine learning model, given vector and feature types, on a chosen dataset
    :param model_name: abbreviated name of model
    :param vector_type: type of vectors to use in training
    :param feature_list: name of additional features (optional), [] if no additional features required
    :param dataset_number: int representing dataset number: 0 - AMAZON, 1 - NEWS, 2 - TWEETS
    :return: NoneType
    """
    split = 0.1

    base_path = Path(__file__).parent
    stem = '../trained_models/' + model_name + '_with_' + '_'.join([vector_type] + feature_list) + '_on_' + str(dataset_number) + '.pckl'
    file_name = str(base_path / stem)
    start = time.time()
    _, sarcasm_labels, sarcasm_data, _, _ = prepare_data(dataset_number, vector_type, feature_list)
    training_data, testing_data, training_labels, testing_labels = train_test_split(sarcasm_data, sarcasm_labels, test_size=split, shuffle=True)

    if isfile(file_name):
        print('Model with filename "' + stem + '" already exists - collecting results')
        ml_model = pd.read_pickle(file_name)
        evaluate_model(ml_model, testing_data, testing_labels)
        return

    while True:
        response = input(
            'Model with filename "' + stem + '" not found: would you like to train one? y / n\n').lower().strip()
        if response in {'y', 'n'}:
            if response == 'y':
                break
            else:
                print('\nCancelling training...')
                return

    print('\nTraining ML models')
    classifier_name, classifier = get_model(model_name)
    print('Classifier: ' + classifier_name)

    # ------------------------- Evaluate new model with five-fold cross validation -----------------------
    scores = cross_val_score(classifier, sarcasm_data, sarcasm_labels, cv=5, scoring='f1_macro')
    five_fold_cross_validation = np.mean(scores)
    print('5-fold cross validation: ', five_fold_cross_validation)
    print('Time taken: ' + str(round((time.time() - start)/60, 2)) + ' minutes')
    classifier.fit(training_data, training_labels)
    evaluate_model(classifier, testing_data, testing_labels)

    with open(file_name, 'wb') as f:
        pickle.dump(classifier, f)
