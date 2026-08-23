import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

data_frame = pd.read_csv('../processed_data/OriginalData.csv', encoding="ISO-8859-1")
data_frame['clean_data'] = pd.read_csv('../processed_data/CleanData.csv', encoding="ISO-8859-1")

nlp = spacy.load('en_core_web_md')

print('Reading in Glove Dictionary...')
with open('../../../language_models/glove/glove.twitter.27B.50d.txt', "r", encoding="utf-8") as file:
    gloveDict = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
print('Reading Glove Dictionary complete.')

comment_column = 'text_data'
data_frame['token_data'] = data_frame[comment_column].apply(lambda x: [token.text for token in nlp(x.lower())])
data_frame['len'] = data_frame['token_data'].apply(lambda x: len(x))
data_frame['percent'] = data_frame['token_data'].apply(lambda x:  sum([1 for token in x if token in gloveDict]))
data_frame['percent'] /= data_frame['len']

non_sarcastic_data = data_frame[data_frame.sarcasm_label == 0]
sarcastic_data = data_frame[data_frame.sarcasm_label == 1]

# vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
# data_vectorized = vectorizer.fit_transform(non_sarcastic_data[comment_column])
# NUM_TOPICS = 10
# lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online',verbose=True)
# data_lda = lda.fit_transform(data_vectorized)

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % idx)
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(non_sarcastic_data[comment_column])
NUM_TOPICS = 10
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online', verbose=True)
data_lda = lda.fit_transform(data_vectorized)

print("LDA Model:")
selected_topics(lda, vectorizer)
text = ' '.join(non_sarcastic_data[comment_column])
wordcloud = WordCloud(stopwords=STOPWORDS).generate(text)
dataset_name = 'sarcastic_data'
wordcloud.to_file(dataset_name + ".png")

print('\n---------------------- Dataset Summary ----------------------')
print('Average length: ', round(data_frame.loc[:, "len"].mean(), 2))
print('Percent in glove dict: ', round((data_frame.loc[:, "percent"].mean())*100, 3))

sarcastic_comments = len(sarcastic_data)
non_sarcastic_comments = len(non_sarcastic_data)
total_comments = sarcastic_comments + non_sarcastic_comments
print('Number of comments: ', total_comments)
print('Number of sarcastic comments: ' + str(sarcastic_comments) + ' - ' +
      str(round((sarcastic_comments*100)/total_comments, 2)) + '%')
print('Number of non-sarcastic comments: ' + str(non_sarcastic_comments) + ' - ' +
      str(round((non_sarcastic_comments*100)/total_comments, 2)) + '%')
