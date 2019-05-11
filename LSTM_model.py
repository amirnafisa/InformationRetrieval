
from misc import prepare_lstm_dataset
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
        model = Embedding(input_dim=self.n_words, output_dim=50, input_length=self.max_len)(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
        out = TimeDistributed(Dense(self.n_labels, activation="softmax"))(model)  # softmax output layer

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        history = self.model.fit(X, np.array(y), batch_size=32, epochs=1, validation_split=0.1, verbose=1)

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
