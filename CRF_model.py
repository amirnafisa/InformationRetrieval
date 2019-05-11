#!/usr/bin/env python3

from sklearn_crfsuite.metrics import flat_classification_report
from misc import prepare_crf_dataset
from sklearn_crfsuite import CRF

class CRFBased:
    '''Vector based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load=False):
        self.load = load
        self.crf = CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=False)

    def load_data(self):
        return prepare_crf_dataset(self.load)

    def fit(self, X, y):
        self.crf.fit(X, y)
        self.labels = list(self.crf.classes_)
        self.labels.remove('O')

    def predict(self, X): #dataset = Train or Test
        return self.crf.predict(X)

    def cross_validate(self, X, y):
        pred = cross_val_predict(estimator=self.crf, X=X, y=y, cv=5)
        return flat_classification_report(y_pred=pred, y_true=y_train, labels=self.labels)

    def evaluate(self, y_true, y_pred):
        print(flat_classification_report(y_pred=y_pred, y_true=y_true, labels=self.labels))
