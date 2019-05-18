from sklearn_crfsuite.metrics import flat_classification_report
from misc import *
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict
import re
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

stemmer = SnowballStemmer('english')

class CRFBased:
    '''CRF based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load, n_train, n_test):
        self.load = load
        self.crf = CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=False)
        self.n_train = int(n_train)
        self.n_test = int(n_test)

    def load_data(self):
        X_train, y_train, X_test, y_test = prepare_crf_dataset(self.load, self.n_train, self.n_test)
        return X_train, y_train, X_test, y_test


    def fit(self, X, y):
        self.crf.fit(X, y)
        self.labels = list(self.crf.classes_)
        self.labels.remove('O')

    def predict(self, X): #dataset = Train or Test
        pred = self.crf.predict(X)
        return pred

    def evaluate(self, y_true, y_pred):
        print("Final Scores for CRF Based Modes:")
        print(flat_classification_report(y_pred=y_pred, y_true=y_true, labels=self.labels))

def prepare_crf_dataset(load, n_train, n_test):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('crf_X_train')
        y_train = tmp_load('crf_y_train')
    if not X_train or not load:
        print('#Extracting features from training dataset ...')
        sent, _ = load_tag_files(1,'Train', n_train)
        tmp_save(sent,'crf_X_train_sent')
        X_train = [sent2features(s) for s in sent]
        y_train = [sent2labels(s) for s in sent]

        tmp_save(X_train, 'crf_X_train')
        tmp_save(y_train, 'crf_y_train')

    if load:
        X_test = tmp_load('crf_X_test')
        y_test = tmp_load('crf_y_test')
    if not X_test or not load:
        print('#Extracting features from test dataset ...')
        sent, _ = load_tag_files(1,'Test', n_test)
        tmp_save(sent,'crf_X_test_sent')
        X_test = [sent2features(s) for s in sent]
        y_test = [sent2labels(s) for s in sent]

        tmp_save(X_test, 'crf_X_test')
        tmp_save(y_test, 'crf_y_test')

    return X_train, y_train, X_test, y_test


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    tf = sent[i][4]
    features = {
    'word.lower()': word.lower(),   #word
    'stemmer.stem(word)': stemmer.stem(word), #stemmed word
    'word.isdigit()': word.isdigit(),   #if word is all numeric
    'postag': postag,   #pos tag
    'tf': tf,           #term frequency
    'len': len(word),   #number of characters in the word
    'i': i,             #position of word in sentence
    'regexp search': 1 if re.search('\W+',word) else 0, #presence of non-alphanumeric char
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),   #word to the left
            'JJ NN': 1 if postag == 'NN' and postag1 == 'JJ' else 0, #pos tag of the prev word
        })
    else:
        features['BOS'] = True  #BOS
    if i > 1:
        word1 = sent[i-2][0]
        features.update({
            '-2:word.lower()': word1.lower(),   #word to the second left
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),   #word to the right
        })
    else:
        features['EOS'] = True                  #EOS
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        features.update({
            '+2:word.lower()': word1.lower(),   #word to the second right
        })
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
