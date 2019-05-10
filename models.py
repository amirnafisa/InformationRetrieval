from misc import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from precrf import *
from sklearn_crfsuite import CRF
from sklearn.model_selection import cross_val_predict

from eli5.sklearn_crfsuite import explain_weights_sklearn_crfsuite

epsilon = 0.2

def Default(doc_vecs_tr, doc_vecs_tt, Y_tr):
    centroid_profile1 = compute_centroid(doc_vecs_tr[Y_tr==1])
    centroid_profile2 = compute_centroid(doc_vecs_tr[Y_tr==0])

    dif = np.array([cosine_sim(vec, centroid_profile1)-cosine_sim(vec, centroid_profile2) for vec in doc_vecs_tt])

    Y_hat = np.ones_like(dif)
    Y_hat[dif <= 0] = 0
    return Y_hat

def KNeighbours(doc_vecs_tr, doc_vecs_tt, Y_tr):
    pca = PCA(n_components=10, svd_solver='arpack')
    X = pca.fit_transform(np.concatenate((doc_vecs_tr,doc_vecs_tt)))
    #X = np.concatenate((doc_vecs_tr,doc_vecs_tt))
    clf = KNeighborsClassifier(8)
    clf.fit(X[:len(Y_tr)], Y_tr)
    Y_hat = clf.predict(X[len(Y_tr):])
    return Y_hat

def CRF_Model(X_train, y_train, X_test):
    crf = tmp_load('crf.mdl')
    if not crf:
        print('Creating CRF model ...')
        crf = CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=False)

        print('Cross validating CRF ...')
        ##Cross validation
        pred = cross_val_predict(estimator=crf, X=X_train, y=y_train, cv=5)
        report = flat_classification_report(y_pred=pred, y_true=y_train)
        print(report)
        tmp_save(crf, 'crf.mdl')

    print('Training CRF ...')
    crf.fit(X_train, y_train)

    #Show weights top 30 weights features
    #print('Top 20 weighted CRF features ...')
    #print(explain_weights_sklearn_crfsuite(crf, top=30, target_names=None, targets=None, feature_re=None, feature_filter=None))

    #Predict and Evaluate
    print('Predicting and Evaluating CRF on Test data...')
    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)

    return y_pred, labels
