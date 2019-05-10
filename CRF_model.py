#!/usr/bin/env python3

from sklearn_crfsuite.metrics import flat_classification_report
from misc import prepare_crf_dataset
from models import CRF_Model

def crf_model():
    X_train, y_train, X_test, y_test = prepare_crf_dataset()


    #Train and Predict CRF
    y_pred, labels = CRF_Model(X_train, y_train, X_test)

    #Evaluate
    report = flat_classification_report(y_pred=y_pred, y_true=y_test, labels = labels)
    print(report)

    #Prints only the tags/labels predicted as B-CP or I-CP
    #crf_debug_print_file(y_pred)
