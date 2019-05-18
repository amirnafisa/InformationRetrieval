#!/usr/bin/env python3
import sys
from Default_model import *
from CRF_model import *
from LSTM_model import *
from LSTM_CRF_model import *

def parse_args():
    if len(sys.argv) < 2:
        print("Command is ./experiment <model> <optional:n_train(max100)> <optional:n_test(max300)>\nModel 1: Vector\tModel2: CRF\tModel3: LSTM\tModel4: LSTMCRF")
        return None
    model = sys.argv[1]
    if model == 'Vector' or model == 'CRF' or model == 'LSTM' or  model == 'LSTMCRF':
        if len(sys.argv) == 4:
            n_train = sys.argv[2]
            n_test = sys.argv[3]
        else:
            n_train = 100
            n_test = 300
        return model, n_train, n_test

    print("Available models are: \nModel 1: Vector\tModel2: CRF\tModel3: LSTM\tModel4: LSTMCRF")
    return None, None, None

if __name__ == '__main__':

    IRClass, n_train, n_test = parse_args()

    if not IRClass:
        sys.exit(-1)

    if IRClass == 'Vector':
        IRModel = VectorBased(False, n_train, n_test)

        X_train, y_train, X_test, y_test = IRModel.load_data()

        #cv_score = IRModel.cross_validate(X_train, y_train, scoring='accuracy')

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate_NP_model(y_true=y_test, y_pred=y_pred)

        IRModel.evaluate_overall_model(y_true=IRModel.get_true_tagged_output(), y_pred=IRModel.pred_tags(y_pred))

    if IRClass == 'CRF':
        IRModel = CRFBased(False, n_train, n_test)

        X_train, y_train, X_test, y_test = IRModel.load_data()

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate(y_true=y_test, y_pred=y_pred)

    if IRClass == 'LSTM':
        IRModel = LSTMBased(False, n_train, n_test)
        X_train, y_train, X_test, y_test = IRModel.load_data()
        #IRModel.cross_validate(X_train, y_train)

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate(y_true=y_test, y_pred=y_pred)

    if IRClass == 'LSTMCRF':
        IRModel = LSTMCRFBased(True, n_train, n_test)
        X_train, y_train, X_test, y_test = IRModel.load_data()

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate(y_true=y_test, y_pred=y_pred)
