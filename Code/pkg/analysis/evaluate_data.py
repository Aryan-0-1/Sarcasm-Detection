import pandas as pd


def split_positive_and_negative_samples(dataset: pd.DataFrame) -> tuple:
    """
    Given a dataset, separate the samples into positive (sarcastic) and negative (non-sarcastic) samples
    :param dataset: dataset containing a column called sarcasm_label, this column must contain integers
    :return: sarcastic and non-sarcastic data split
    """
    return dataset[dataset['sarcasm_label'] == 1], dataset[dataset['sarcasm_label'] == 0]


def feature_evaluate(data: pd.DataFrame):
    """
    Given a DataFrame containing the columns data['feature'] and data['sarcasm_label'], deduce the difference
    in features of the sarcastic and non-sarcastic data
    :param data: pandas DataFrame
    :return:
    """
    positive_samples, negative_samples = split_positive_and_negative_samples(data)

    positive_feature_df = positive_samples['feature'].apply(pd.Series)
    negative_feature_df = negative_samples['feature'].apply(pd.Series)

    print('Positive features: ', positive_feature_df.mean(axis=0).tolist())
    print('Negative features: ', negative_feature_df.mean(axis=0).tolist())


