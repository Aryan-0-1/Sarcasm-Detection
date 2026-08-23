import matplotlib.pyplot as plt
from Code.pkg.analysis.evaluate_data import split_positive_and_negative_samples
from Code.pkg.data_processing.helper import get_feature_col
import numpy as np
import pandas as pd
from pathlib import Path


def compute_differences(sample1: np.ndarray, sample2: np.ndarray) -> np.ndarray:
    output = sample1 - sample2
    return np.absolute(output)


def box_plot_visualisation(dataset_1: pd.DataFrame, dataset_2: pd.DataFrame):
    """
    Visualise difference in overall sentiment of sarcastic and non-sarcastic data on one dataset
    Produce a box plot
    :param dataset_1: pd.DataFrame containing sarcasm labels and overall sentiment labels for each instance in dataset
    :param dataset_2: pd.DataFrame containing sarcasm labels and overall sentiment labels for each instance in dataset
    :return:
    """

    sarcastic_news, non_sarcastic_news = split_positive_and_negative_samples(dataset_1)
    sarcastic_twitter, non_sarcastic_twitter = split_positive_and_negative_samples(dataset_2)

    sarcastic_1, non_sarcastic_1 = sarcastic_news['sentiment'], non_sarcastic_news['sentiment']
    sarcastic_2, non_sarcastic_2 = sarcastic_twitter['sentiment'], non_sarcastic_twitter['sentiment']

    data_a = [sarcastic_1, sarcastic_2]
    data_b = [non_sarcastic_1, non_sarcastic_2]

    ticks = ['News Headlines', 'Twitter Data']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], linestyle='--', color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
        plt.setp(bp['means'], linestyle='--', color='orange')

    plt.figure()

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.5, sym='', widths=0.8, showmeans=True, showfliers=True, meanline=True)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.8, showmeans=True, showfliers=True, meanline=True)
    set_box_color(bpl, '#D7191C')
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Sarcastic')
    plt.plot([], c='#2C7BB6', label='Non-Sarcastic')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-1.5, len(ticks) * 2)
    plt.ylim(-0.1, 4.1)
    plt.yticks(np.arange(0, 5, 1.0))
    plt.ylabel('Overall Sentiment Scores', fontsize=13)
    plt.xlabel('Datasets', fontsize=13)
    plt.tight_layout()
    plt.savefig('boxcompare.png')


def visualise_sentiment_differences(dataset1: np.ndarray, dataset2: np.ndarray):
    """
    Visualise the difference in sentiment between sarcastic and non-sarcastic instances in each class
    Larger bars show that the proportion of tokens labelled with that particular sentiment class differs more
    between sarcastic and non-sarcastic instances
    :param dataset1:
    :param dataset2:
    :return:
    """
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    index = 0.6 * np.arange(len(labels))
    bar_width = 0.2

    plt.bar(index, dataset1, bar_width,
            alpha=1,
            color='dimgrey',
            label='Sarcasm Amazon Review Corpus')

    plt.bar(index + bar_width, dataset2, bar_width,
            alpha=0.7,
            color='dimgrey',
            label='News Headlines for Sarcasm Detection')

    plt.ylabel('Absolute differences between sarcastic\n and non-sarcastic instances', fontsize=13)
    plt.xlabel('Sentiment Labels', fontsize=13)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig('Images/absolute_differences.png')


def visualise_sentiment(positive: np.ndarray, negative: np.ndarray):
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

    index = 0.6 * np.arange(len(labels))
    bar_width = 0.2

    plt.bar(index, positive, bar_width,
            alpha=1,
            color='dimgrey',
            label='Non-Sarcastic')

    plt.bar(index + bar_width, negative, bar_width,
            alpha=0.7,
            color='dimgrey',
            label='Sarcastic')

    plt.ylabel('Proportion of Tokens', fontsize=13)
    plt.xlabel('Sentiment Labels', fontsize=13)
    plt.xticks(index + bar_width, labels)
    plt.legend()
    # plt.title('Sarcasm Amazon Review Corpus', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.show()
    # plt.savefig('Images/SARC_sentimentlabels.png')


if __name__ == "__main__":
    base_path = Path(__file__).parent

    data_1 = pd.read_csv((base_path / ("../datasets/news_headlines" + "/processed_data/OriginalData.csv")).resolve(),
                       encoding="ISO-8859-1")
    feature_1 = get_feature_col(data_1, "../datasets/news_headlines", 'sentiment')
    data_1['sentiment'] = feature_1.apply(lambda x: x[-1])
    data_2 = pd.read_csv((base_path / ("../datasets/ptacek" + "/processed_data/OriginalData.csv")).resolve(),
                       encoding="ISO-8859-1")
    feature_2 = get_feature_col(data_2, "../datasets/ptacek", 'sentiment')
    data_2['sentiment'] = feature_2.apply(lambda x: x[-1])
    box_plot_visualisation(data_1, data_2)
