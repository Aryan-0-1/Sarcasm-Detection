import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable tensorflow warnings
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
from os.path import isfile
from tensorflow.keras.layers import Layer
from tensorflow.keras import optimizers
from ..data_processing.helper import prepare_data, get_dataset_name
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Conv1D, Flatten, Dense, Dropout, GlobalMaxPooling1D, \
    Bidirectional, MaxPooling1D, Input
from tensorflow.keras.layers import SimpleRNN, GRU
from tensorflow.keras.layers import Embedding
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import tensorflow_hub as hub
import spacy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

nlp = spacy.load('en_core_web_md')

# ncc terminal
# ssh -D 8080 -q -C -N kgxj22@mira.dur.ac.uk


# ---------------------------- Create Embedding Layer Classes ----------------------------
class ElmoEmbeddingLayer(Layer):

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.mask = None
        self.elmo = None
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # base_path = Path(__file__).parent

        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self._trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))

        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mask=None):

        seq_len = [inputs.shape[1]] * inputs.shape[0]
        result = self.elmo(inputs={"tokens": K.cast(inputs, dtype=tf.string),
                                   "sequence_len": seq_len},
                           as_dict=True,
                           signature='tokens',
                           )['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        if not self.mask:
            return None
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.dimensions


class GloveEmbeddingLayer(Embedding):
    def __init__(self, word_index, max_seq_len, **kwargs):
        self.word_index = word_index
        self.MAX_SEQUENCE_LENGTH = max_seq_len
        self.output_dim = 50
        self.embeddings_matrix = self.compute_embedding_matrix(len(self.word_index) + 1)
        super(GloveEmbeddingLayer, self).__init__(input_dim=len(self.word_index) + 1,
                                                  output_dim=self.output_dim,
                                                  embeddings_initializer=Constant(self.embeddings_matrix),
                                                  input_length=self.MAX_SEQUENCE_LENGTH,
                                                  trainable=False,
                                                  **kwargs
                                                  )

    def get_config(self):
        return {'word_index': self.word_index,
                'max_seq_len': self.MAX_SEQUENCE_LENGTH}

    def compute_embedding_matrix(self, vocab_size):
        base_path = Path(__file__).parent
        file_path = (base_path / "../language_models/glove/glove.twitter.27B.50d.txt").resolve()

        with open(file_path, "r", encoding="utf-8") as file:
            embeddings_index = {line.split()[0]: list(map(float, line.split()[1:])) for line in file}
            del embeddings_index['0.45973']  # for some reason, this entry has 49 dimensions instead of 50

        embedding_matrix = np.zeros((vocab_size, self.output_dim))
        for word, i in self.word_index.items():
            if i >= vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


# The base code of the following attention mechanism is Copyright (c) 2017 to Ilya Ivanov - permission is granted under MIT Licence
# Adaptations were made to transform this function into an AttentionLayer
# https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
# Proposed by Yang et al. in "Hierarchical Attention Networks for Document Classification" (2016)
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        self.w_omega = None
        self.b_omega = None
        self.time_major = False
        self.u_omega = None
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        attention_size = int(input_shape[1])
        hidden_size = int(input_shape[2])  # D value - hidden size of the RNN layer
        # removed hidden_size = inputs.shape[2].value

        # Trainable parameters
        self.w_omega = tf.Variable(K.random_normal([hidden_size, attention_size], stddev=0.1))
        self.b_omega = tf.Variable(K.random_normal([attention_size], stddev=0.1))
        self.u_omega = tf.Variable(K.random_normal([attention_size], stddev=0.1))
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        if isinstance(x, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            x = tf.concat(x, 2)

        if self.time_major:
            # (T,B,D) => (B,T,D)
            x = tf.transpose(x, [1, 0, 2])

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(x, self.w_omega, axes=1) + self.b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector

        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), 1)

        return alphas, output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(AttentionLayer, self).get_config()

# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------
def pad_string(tokens: list, limit: int) -> list:
    tokens = tokens[0:limit]
    extend_by = limit - len(tokens)

    if extend_by > 0:
        tokens.extend([""] * extend_by)
    return tokens


def get_length_limit(dataset_name: str) -> int:
    # based approximately on the average number of tokens in dataset
    names = {'news_headlines': 150, 'amazon_reviews': 150, 'ptacek': 150}
    try:
        return names[dataset_name]
    except KeyError:
        raise ValueError('Dataset name "' + dataset_name + '" does not exist')


def get_batch_size(model_name: str) -> int:
    # set batch size to 1 for online learning in lstm
    batch_sizes = {'lstm': 32, 'bi-lstm': 32, 'cnn': 32, 'vanilla-rnn': 32,
                   'gru': 32, 'attention-lstm': 32, 'dcnn': 32, 'bi-gru':32,
                   "attention-bi-lstm": 32, "attention-bi-gru": 32}
    return batch_sizes[model_name]


def get_custom_layers(model_name=None, vector_type=None):
    custom_layers = {}

    if model_name and 'attention' in model_name:
        custom_layers['AttentionLayer'] = AttentionLayer

    if vector_type:
        if vector_type == 'elmo':
            custom_layers['ElmoEmbeddingLayer'] = ElmoEmbeddingLayer
        elif vector_type == 'glove':
            custom_layers['GloveEmbeddingLayer'] = GloveEmbeddingLayer
    return custom_layers


def initialize_uninitialized(sess):
    global_vars = tf.compat.v1.global_variables()
    is_not_initialized = sess.run([tf.compat.v1.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    # print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.compat.v1.variables_initializer(not_initialized_vars))


def load_model_from_file(filename: str, custom_layers: dict):
    session = tf.compat.v1.keras.backend.get_session()

    with CustomObjectScope(custom_layers):
        model = load_model(filename)
        initialize_uninitialized(session)
    return model


# -------------------------------------------- DEEP LEARNING ARCHITECTURES ---------------------------------------------
def lstm_network(model, optimiser):
    model.add(LSTM(60, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.1))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def bidirectional_lstm_network(model, optimiser):
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def deep_cnn_network(model, optimiser):
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def vanilla_rnn(model, shape, optimiser):
    model.add(SimpleRNN(50, batch_input_shape=shape, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def gru_network(model, shape, optimiser):
    model.add(GRU(50, batch_input_shape=shape, return_sequences=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def bidirectional_gru_network(model, shape, optimiser):
    model.add(Bidirectional(GRU(50, batch_input_shape=shape, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def lstm_with_attention(embedding_layer, shape, optimiser, vector_type):
    if vector_type == 'elmo':
        inputs = Input(batch_shape=shape, dtype=tf.string)
    else:
        inputs = Input(batch_shape=shape)
    x = embedding_layer(inputs)
    x = LSTM(60, return_sequences=True)(x)
    attention_weights, x = AttentionLayer()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def bidirectional_lstm_with_attention(embedding_layer, shape, optimiser, vector_type):
    if vector_type == 'elmo':
        inputs = Input(batch_shape=shape, dtype=tf.string)
    else:
        inputs = Input(batch_shape=shape)
    x = embedding_layer(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    attention_weights, x = AttentionLayer()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def bidirectional_gru_with_attention(embedding_layer, shape, optimiser, vector_type):
    if vector_type == 'elmo':
        inputs = Input(batch_shape=shape, dtype=tf.string)
    else:
        inputs = Input(batch_shape=shape)
    x = embedding_layer(inputs)
    x = Bidirectional(GRU(50, batch_input_shape=shape, return_sequences=True))(x)
    attention_weights, x = AttentionLayer()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])
    return model


def cnn_network(model, optimiser):
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# -------------------------------------------- MAIN FUNCTIONS -----------------------------------------------
def prepare_pre_vectors(text: str, vector_type: str, dataset_num: int, model_name: str):
    token_list = [t.text for t in nlp(text)]
    split_text = ' '.join(token_list)

    dataset_name = get_dataset_name(dataset_num)
    length_limit = get_length_limit(dataset_name)

    if vector_type == 'elmo':
        text = pd.DataFrame([pad_string(token_list, length_limit)]*get_batch_size(model_name))
        text = text.replace({None: ""})
        return token_list, text.to_numpy()

    elif vector_type == 'glove':
        path_to_dataset_root = "../datasets/" + dataset_name
        base_path = Path(__file__).parent
        tokeniser = pd.read_pickle((base_path / (
                    path_to_dataset_root + "/processed_data/Tokenisers/" + vector_type + "_tokeniser.pckl")).resolve())

        sequences = tokeniser.texts_to_sequences([split_text]*get_batch_size(model_name))
        # token_list = [t.text for t in nlp(tokeniser.sequences_to_texts(sequences)[0])]
        return token_list, pad_sequences(sequences, maxlen=length_limit, padding='post')
    else:
        raise ValueError('Only glove and elmo vectors supported')


def prepare_vector_embedding_layer(s_data: pd.Series, s_labels: pd.Series, dataset_name: str, vector_type: str, split: float, max_batch_size: int, limit: int):
    number_of_batches = len(s_data) // (max_batch_size * (1 / split))
    sarcasm_data = s_data[:int((number_of_batches * max_batch_size) / split)]
    sarcasm_labels = s_labels[:int((number_of_batches * max_batch_size) / split)]

    if vector_type == 'elmo':
        text = pd.DataFrame([pad_string([x.text for x in nlp(t)], limit) for t in sarcasm_data])
        text = text.replace({None: ""})
        text = text.to_numpy()
        print(text)
        return text, sarcasm_labels, Input(dtype=tf.string, batch_shape=(max_batch_size, limit)), ElmoEmbeddingLayer(batch_input_shape=(max_batch_size, limit), dtype=tf.string)

    elif vector_type == 'glove':
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(sarcasm_data)
        base_path = Path(__file__).parent
        path_to_dataset_root = "../datasets/" + dataset_name

        with open(str(base_path / (
                path_to_dataset_root + "/processed_data/Tokenisers/" + vector_type + "_tokeniser.pckl")), 'wb') as f:
            pickle.dump(tokenizer, f)

        sequences = tokenizer.texts_to_sequences(sarcasm_data)
        padded_data = pad_sequences(sequences, maxlen=limit, padding='post')
        return padded_data, sarcasm_labels, Input(batch_shape=(max_batch_size, limit)), GloveEmbeddingLayer(tokenizer.word_index, limit)
    else:
        raise TypeError('Vector type must be "elmo" or "glove"')


def visualise_results(history, file_name):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_name + '_loss.png')
    plt.show()

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    plt.plot(epochs, accuracy, label="Training accuracy")
    plt.plot(epochs, val_accuracy, label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(file_name + '_accuracy.png')
    plt.show()


def get_model(model_name: str, dataset_name: str, sarcasm_data: pd.Series, sarcasm_labels: pd.Series, vector_type: str, split: float):
    max_batch_size = get_batch_size(model_name)
    length_limit = get_length_limit(dataset_name)

    # need to deal with concatenating multiple embedding layers - difficult for time series data, as our current annotators work for sentences only
    # e.g. sentiment annotator produces a 6-dimensional vector for each sentence - not for individual words
    # sentiment may be equally valid if we decide to include sentiment per word. However, these dimensions will not match
    # if i remove the overall sentiment from the 6-dimensional vector, we could mimic the behaviour
    # e.g. [0 0 0 0 1] for each word, when averaging embeddings will be very similar

    s_data, l_data, input_layer, e_layer = prepare_vector_embedding_layer(sarcasm_data, sarcasm_labels, dataset_name, vector_type,
                                                                      split, max_batch_size, length_limit)

    new_adam = optimizers.Adam()  # lr=0.0001, decay=0.001
    model = Sequential()
    model.add(input_layer)

    e_layer.trainable = False
    model.add(e_layer)
    if model_name == 'lstm':
        model = lstm_network(model, new_adam)
    elif model_name == 'attention-lstm':
        # use batch_shape instead of model
        model = lstm_with_attention(e_layer, (max_batch_size, length_limit), new_adam, vector_type)
    elif model_name == 'attention-bi-lstm':
        # use batch_shape instead of model
        model = bidirectional_lstm_with_attention(e_layer, (max_batch_size, length_limit), new_adam, vector_type)
    elif model_name == 'bi-lstm':
        model = bidirectional_lstm_network(model, new_adam)
    elif model_name == 'dcnn':
        model = deep_cnn_network(model, new_adam)
    elif model_name == 'cnn':
        model = cnn_network(model, new_adam)
    elif model_name == 'vanilla-rnn':
        model = vanilla_rnn(model, (max_batch_size, length_limit), new_adam)
    elif model_name == 'gru':
        model = gru_network(model, (len(s_data), length_limit), new_adam)
    elif model_name == 'bi-gru':
        model = bidirectional_gru_network(model, (max_batch_size, length_limit), new_adam)
    elif model_name == 'attention-bi-gru':
        model = bidirectional_gru_with_attention(e_layer, (max_batch_size, length_limit), new_adam, vector_type)
    print(model.summary())
    return s_data, l_data, model


def evaluate_model(model_name, trained_model, testing_data, testing_labels):
    max_batch_size = get_batch_size(model_name)
    probabilities = trained_model.predict(x=np.array(testing_data), batch_size=max_batch_size)
    y_pred = np.where(probabilities > 0.5, 1, 0)
    print(testing_labels)
    print(probabilities)
    print(np.array(y_pred))
    print(y_pred)
    f1 = f1_score(np.array(testing_labels), np.array(y_pred))
    precision = precision_score(np.array(testing_labels), np.array(y_pred))
    recall = recall_score(np.array(testing_labels), np.array(y_pred))
    mcc = matthews_corrcoef(np.array(testing_labels), np.array(y_pred))
    print('F1 Score: ', f1)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('MCC: ', mcc)
    tn, fp, fn, tp = confusion_matrix(np.array(testing_labels), np.array(y_pred)).ravel()
    print('True Positives: ' + str(tp))
    print('True Negatives: ' + str(tn))
    print('False Positives: ' + str(fp))
    print('False Negatives: ' + str(fn))


def get_dl_results(model_name: str, dataset_number: int, vector_type: str, set_size=None):
    base_path = Path(__file__).parent
    dataset_name, sarcasm_labels, _, clean_data, _ = prepare_data(dataset_number, vector_type, [], set_size)

    print('Training ' + model_name.upper() + ' using ' + vector_type + ' vectors.')
    print('Dataset size: ' + str(len(clean_data)) + '\n')
    split = 0.1
    epochs = 300
    patience = 10
    max_batch_size = get_batch_size(model_name)

    with tf.compat.v1.Session() as sess:
        s_data, l_data, dl_model = get_model(model_name, dataset_name, clean_data, sarcasm_labels, vector_type, split)
        graph = tf.compat.v1.get_default_graph()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())

        custom_layer = get_custom_layers(model_name, vector_type)
        training_data, testing_data, training_labels, testing_labels = train_test_split(s_data, l_data, test_size=split, shuffle=True)

        stem = model_name + '_with_' + vector_type + '_on_' + str(dataset_number) + '.h5'
        file_name = str(base_path / ('../trained_models/' + stem))

        if isfile(file_name):
            print('Model with filename "' + stem + '" already exists - collecting results')
            dl_model = load_model_from_file(file_name, custom_layer)
            evaluate_model(model_name, dl_model, testing_data, testing_labels)
            return

        while True:
            response = input('Model with filename "' + stem + '" not found: would you like to train one? y / n\n').lower().strip()
            if response in {'y', 'n'}:
                if response == 'y':
                    break
                else:
                    print('\nCancelling training...')
                    return

        # training_data, training_labels = repeat_positive_samples(training_data, training_labels, 0.5)
        model_checkpoint = ModelCheckpoint(file_name, monitor='val_loss', mode='auto', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        model_history = dl_model.fit(x=np.array(training_data), y=np.array(training_labels), validation_data=(testing_data, testing_labels),
                                     epochs=epochs, batch_size=max_batch_size, callbacks=[early_stopping, model_checkpoint])

        dl_model = load_model_from_file(file_name, custom_layer)
        evaluate_model(model_name, dl_model, testing_data, testing_labels)

        visualise_results(model_history, str(
            base_path / ('../training_images/' + model_name + '_with_' + vector_type + '_on_' + str(dataset_number))))
    # ---------------------------------------------------------------------------------------------------------------------
