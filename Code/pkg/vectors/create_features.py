from warnings import filterwarnings; filterwarnings('ignore')
from pycorenlp import StanfordCoreNLP
import numpy as np
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load('en_core_web_md')


class SentimentAnnotator:
    def __init__(self):
        self.nlp_wrapper = None
        self.settings = {'annotators': 'sentiment',
                         'outputFormat': 'json',
                         'timeout': 1000000,
                         }

        # ---------------------------------------------------------------------
        # Start CoreNLP server before using sentiment annotator
        # cd stanford-corenlp-full-2018-10-05/
        # java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        # note: change the 2g to 1g if space requirements too high
        # ---------------------------------------------------------------------

        # os.chdir("./stanford-corenlp-full-2018-10-05/")
        # os.system('java -mx1g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')

    def sentence_level(self, sentence_tokens: list) -> list:
        """
        Given a sentence as a list of tokens, return the breakdown of sentiment values in the sentence
        :param sentence_tokens: A list of tokens
        :return: list of sentiment counts e.g. [0.1, 0.3, 0.2, 0.3, 0.1]
        counts[0:4] --> frequency of tokens with sentiment values 0 - 4
        """
        counts = [0] * 5

        for token in sentence_tokens:
            sentiment_val = int(self.nlp_wrapper.annotate(token,
                                                          properties=self.settings)["sentences"][0]["sentimentValue"])
            counts[sentiment_val] += 1
        return [count / len(sentence_tokens) for count in counts]

    def transform(self, string: str) -> list:
        """
        Given a string, decompose it into sentences and annotate the sentiments of each sentence
        :param string: string of data
        :return: list of sentiment count averages across all sentences
        """
        if self.nlp_wrapper is None:
            self.nlp_wrapper = StanfordCoreNLP('http://localhost:9000')

        all_sentences = self.nlp_wrapper.annotate(string, properties=self.settings)["sentences"]
        sentiment_values = []

        for sentence in all_sentences:
            token_list = [token['originalText'] for token in sentence['tokens']]
            sentiment_values.append(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])
            # print(self.sentence_level(token_list) + [int(sentence["sentimentValue"])])

        return list(np.mean(sentiment_values, axis=0))

    def fit_transform(self, data: pd.Series):
        return data.apply(lambda x: self.transform(x))


class PunctuationAnnotator:
    """
    PunctuationAnnotator must take raw data, as cleaned data will have some of these features removed (capitalisation)
    """
    def __init__(self):
        self.maximal_features = None

    @staticmethod
    def create_raw_features(string: str):
        tokens = [token.text for token in nlp(string)]
        counts = [0] * 5

        for token in tokens:
            counts[0] += 1  # sentence length
            counts[1] += token.count('!')   # number of ! tokens
            counts[2] += token.count('?')    # number of ? tokens
            counts[3] += token.count("'")   # number of ' tokens

            if not token.islower():
                counts[4] += 1  # number of tokens that have capital letters

        counts[3] = counts[3] // 2  # number of pairs of quote chars (roughly translates to number of quotes)
        return counts

    def normalise_features(self, raw_features: pd.Series):
        self.maximal_features = [-float('inf')] * 5

        for count in raw_features:
            for idx in range(len(count)):
                if count[idx] > self.maximal_features[idx]:
                    self.maximal_features[idx] = count[idx]

        return raw_features.apply(lambda counts: [raw / (maximal * np.mean(self.maximal_features))
                                                  for raw, maximal in zip(counts, self.maximal_features)])

    def transform(self, string: str) -> list:
        """
        Given a string, decompose it into sentences and annotate each sentence
        :param string: string of data
        :return: frequency of ! and ? characters in sentence
        """
        if self.maximal_features is None:
            raise ValueError('Maximal features must be initialised before strings can be transformed')

        counts = self.create_raw_features(string)
        return [raw / (maximal * np.mean(self.maximal_features)) for raw, maximal in zip(counts, self.maximal_features)]

    def fit_transform(self, data: pd.Series):
        raw_features = data.apply(lambda x: self.create_raw_features(x))
        return self.normalise_features(raw_features)


class TopicModelAnnotator:
    def __init__(self):
        self.num_topics = 10
        self.max_iterations = 1000
        self.bow_vectoriser = None
        self.lda = None

    @staticmethod
    def lemmatisation(text: str) -> str:
        return " ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in nlp(text) if
                         token.pos_ in ('NOUN', 'ADJ', 'VERB', 'ADV')])

    def transform(self, string: str) -> list:
        vector = self.bow_vectoriser.transform(string)
        topic_distribution = self.lda.transform(vector)
        return topic_distribution.tolist()

    def fit_transform(self, data: pd.Series):
        # apply lemmatisation
        lemmatised_data = data.apply(lambda x: self.lemmatisation(x))
        self.bow_vectoriser = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        vectorised_data = self.bow_vectoriser.fit_transform(lemmatised_data)

        self.lda = LatentDirichletAllocation(n_components=self.num_topics, max_iter=self.max_iterations, learning_method='online',
                                             verbose=True)
        topic_distribution = self.lda.fit_transform(vectorised_data)

        print("\n----- LDA Model ------")
        top_n = 10

        words = self.bow_vectoriser.get_feature_names()
        for topic_idx, topic in enumerate(self.lda.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                            for i in topic.argsort()[:-top_n - 1:-1]]))
        return topic_distribution.tolist()


feature_type = {'sentiment': SentimentAnnotator(), 'punctuation': PunctuationAnnotator(),
                'topic_model': TopicModelAnnotator()}


def extract_features(path_to_root: str, data: pd.DataFrame, feature: str):
    """
    Given a particular dataset and feature type, extract these features from the data and store them in a pickle file
    :param path_to_root: insert the path to the correct location to store newly generated features
    :param data: DataFrame containing column called CleanData (strings of data to extract features from)
    :param feature: feature type is a string representing the feature type to extract - valid strings include 'sentiment'
    :return: NoneType
    """
    open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'wb').close()
    store_in = open(path_to_root + "/processed_data/Features/" + feature + ".pckl", 'ab')
    annotator = feature_type[feature]

    if feature == 'punctuation':
        feature_data = annotator.fit_transform(data['text_data'])
    else:
        feature_data = annotator.fit_transform(data['clean_data'])

    pickle.dump(feature_data, store_in)
    store_in.close()

    if feature in {'topic_model', 'punctuation'}:
        annotator_file = open(path_to_root + "/processed_data/Vectorisers/" + feature + "_annotator.pckl", 'ab')
        pickle.dump(annotator, annotator_file)
        annotator_file.close()