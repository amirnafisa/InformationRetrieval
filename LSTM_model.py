
from misc import *
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

class LSTMBased:
    '''LSTM based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load, n_train, n_test):
        self.load = load
        self.max_len = 50
        self.n_train = int(n_train)
        self.n_test = int(n_test)

    def load_data(self):
        X_train, y_train, X_test, y_test, self.words, self.labels = prepare_lstm_dataset(self.load, self.max_len, self.n_train, self.n_test)
        self.n_words = len(self.words)
        self.n_labels = len(self.labels)
        return X_train, y_train, X_test, y_test

    def fit(self, X, y):
        input = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.n_words, output_dim=self.max_len, input_length=self.max_len)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.n_labels, activation="softmax"))(model)  # softmax output layer

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        history = self.model.fit(X, np.array(y), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

    def predict(self, X): #dataset = Train or Test
        pred = []
        for i in range(len(X)):
            pred.extend(self.model.predict(np.array([X[i]])))

        return pred

    def cross_validate(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
        input = Input(shape=(self.max_len,))
        model = Embedding(input_dim=self.n_words, output_dim=50, input_length=self.max_len)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.n_labels, activation="softmax"))(model)  # softmax output layer

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=1, validation_split=0.1, verbose=1)

        p = model.predict(np.array([X_te[10]]))
        p = np.argmax(p, axis=-1)

        for w, pred in zip(X_te[10], p[0]):
            if self.words[w] != 'PADGARBAGE':
                print("{:15}: {}".format(self.words[w], self.labels[pred]))

    def evaluate(self, y_true, y_pred):
        y_true = np.array([np.argmax(sent,axis=-1) for sent in y_true])
        y_pred = np.array([np.argmax(sent,axis=-1) for sent in y_pred])
        y_true = np.array([y for sent in y_true for y in sent])
        y_pred = np.array([y for sent in y_pred for y in sent])
        if (sum(y_true==0) > sum(y_true==1)):
            y_true = ['O' if y==0 else 'CP' for y in list(y_true)]
            y_pred = ['O' if y==0 else 'CP' for y in list(y_pred)]
        else:
            y_true = ['CP' if y==0 else 'O' for y in list(y_true)]
            y_pred = ['CP' if y==0 else 'O' for y in list(y_pred)]

        print("Final Scores for LSTM Based Modes:")
        print(classification_report(y_pred=y_pred, y_true=y_true, labels=['CP']))

def prepare_lstm_dataset(load, max_len, n_train, n_test):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('lstm_X_train')
        y_train = tmp_load('lstm_y_train')
        X_test = tmp_load('lstm_X_test')
        y_test = tmp_load('lstm_y_test')
        vocab = tmp_load('lstm_vocab')
        labels = tmp_load('lstm_labels')

    if type(X_train) == type(None) or not load:
        print('#Loading sentences training dataset ...')
        sent, _ = load_tag_files(1,'Train', n_train)

        vocab = list(set([w[0] for s in sent for w in s]))
        vocab.append('PADGARBAGE')
        n_vocab = len(vocab)
        labels = list(set([labels for s in sent for labels in sent2labels(s) ]))
        n_labels = len(labels)

        word2idx = {w: i for i, w in enumerate(vocab)}
        label2idx = {t: i for i, t in enumerate(labels)}


        print('#Extracting features from training dataset ...')
        X_train = [[word2idx[w[0]] for w in s] for s in sent]
        X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=n_vocab - 1)
        y_train = [[label2idx[w[2]] for w in s] for s in sent]
        y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=label2idx["O"])
        y_train = [to_categorical(i, num_classes=n_labels) for i in y_train]

        print('#Extracting features from testing dataset ...')
        sent, _ = load_tag_files(1,'Test', n_test)

        X_test = [[word2idx[w[0]] if w[0] in word2idx else word2idx['PADGARBAGE'] for w in s] for s in sent]
        X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=n_vocab - 1)
        y_test = [[label2idx[w[2]] for w in s] for s in sent]
        y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=label2idx["O"])
        y_test = [to_categorical(i, num_classes=n_labels) for i in y_test]

        tmp_save(X_train, 'lstm_X_train')
        tmp_save(y_train, 'lstm_y_train')
        tmp_save(vocab, 'lstm_vocab')
        tmp_save(labels, 'lstm_labels')
        tmp_save(X_test, 'lstm_X_test')
        tmp_save(y_test, 'lstm_y_test')

    return X_train, y_train, X_test, y_test, vocab, labels
