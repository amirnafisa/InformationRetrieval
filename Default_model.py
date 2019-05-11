from models import *
from misc import *
from prep import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

class NPTrainingModel:
    '''Vector based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load, l=0.01):
        self.l = l
        self.load = load

    def load_data(self):
        return prepare_dataset(self.load)

    def fit(self, X, y):
        self.centroid_profile1 = compute_centroid(X[y==1])
        self.centroid_profile2 = compute_centroid(X[y==0])

    def predict(self, X): #dataset = Train or Test
        dif = np.array([cosine_sim(vec, self.centroid_profile1)-cosine_sim(vec, self.centroid_profile2) for vec in X])

        y = np.ones_like(dif)
        y[dif <= 0] = 0
        return y

    def get_params(self, deep = False):
        return {'l':self.l}

    def cross_validate(self, X, y, scoring = 'accuracy'):
        return cross_val_score(self, X, y, cv=5, scoring=scoring)

    def evaluate(self, y_true, y_pred):
        print(accuracy_score(y_true, y_pred))


class VectorBased:
    '''Vector based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load=False):
        self.NPModel = NPTrainingModel(load=load)

    def load_data(self):
        X_train, y_train, X_test, y_test, self.tagged_test_sent, self.tagged_test_output, self.retrieved_test_NPs = self.NPModel.load_data()
        return X_train, y_train, X_test, y_test

    def get_true_tagged_output(self):
        return self.tagged_test_output

    def fit(self, X, y):
        self.NPModel.fit(X, y)

    def predict(self, X): #dataset = Train or Test
        y_pred = self.NPModel.predict(X)
        return y_pred

    def pred_tags(self, y):
        print("collecting predicted tags ...")
        pred_NPs = get_predicted_NPs(y, self.retrieved_test_NPs)
        print("tagging the documents")
        return tag_pred_labels(self.tagged_test_sent, pred_NPs)

    def cross_validate_NP_model(self, X, scoring = 'accuracy'):
        print("NP Training Cross Validation")
        return self.NPModel.cross_validate(X, self.y, scoring)

    def evaluate_NP_model(self, y_true, y_pred):
        print("Raw NP model score:")
        self.NPModel.evaluate(y_true, y_pred)

    def evaluate_overall_model(self, y_true, y_pred):
        print("Final Scores for Vector Based Modes:")
        print(flat_classification_report(y_pred=y_pred, y_true=y_true))
