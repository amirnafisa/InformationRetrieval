#!/usr/bin/env python3
import sys
from Default_model import *
from CRF_model import *
from LSTM_model import *

def parse_args():
    if len(sys.argv) < 2:
        print("Command is ./experiment <model>\nModel 1: VectorBased\tModel2: CRFBased\tModel3: LSTMBased")
        return None
    model = sys.argv[1]
    if model == 'VectorBased' or model == 'CRFBased' or 'LSTMBased':
        return model

    print("Available models are: \nModel 1: VectorBased\tModel2: CRFBased")
    return None

if __name__ == '__main__':

    IRClass = parse_args()

    if not IRClass:
        sys.exit(-1)

    if IRClass == 'VectorBased':
        IRModel = VectorBased(load=False)

        X_train, y_train, X_test, y_test = IRModel.load_data()

        tagged_test = IRModel.get_true_tagged_output()

        #cv_score = IRModel.cross_validate(X_train, y_train, scoring='accuracy')
        #print("Cross Validation Score for Vector Based IR:\n",cv_score,"\n\n")

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate_NP_model(y_true=y_test, y_pred=y_pred)


        tagged_test_pred = IRModel.pred_tags(y_pred)
        print("evaluating final model ...")
        IRModel.evaluate_overall_model(y_true=tagged_test, y_pred=tagged_test_pred)

    if IRClass == 'CRFBased':
        IRModel = CRFBased(load=True)

        X_train, y_train, X_test, y_test = IRModel.load_data()

        #cv_score = IRModel.cross_validate(X_train, y_train, scoring='recall_macro')
        #print("Cross Validation Score for Vector Based IR:\n",cv_score,"\n\n")

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate(y_true=y_test, y_pred=y_pred)

    if IRClass == 'LSTMBased':
        IRModel = LSTMBased(load=False)
        X_train, y_train, X_test, y_test = IRModel.load_data()
        #IRModel.cross_validate(X_train, y_train)

        IRModel.fit(X_train, y_train)
        y_pred = IRModel.predict(X_test)
        IRModel.evaluate(y_true=y_test, y_pred=y_pred)
