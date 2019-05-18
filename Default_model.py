from misc import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

class NPTrainingModel:
    '''Vector based information retrieval.
    The model is similar to the Default model
    used in homework 2 and 3'''

    def __init__(self, load, n_train, n_test, l=0.01):
        self.l = l
        self.load = load
        self.n_train = int(n_train)
        self.n_test = int(n_test)

    def load_data(self):
        return prepare_VectorBased_dataset(self.load, self.n_train, self.n_test)

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

    def __init__(self, load, n_train, n_test):
        self.NPModel = NPTrainingModel(load, n_train, n_test)

    def load_data(self):
        X_train, y_train, X_test, y_test, self.test_doc_tokens, self.true_test_output, self.retrieved_candidate_NPs = self.NPModel.load_data()
        return X_train, y_train, X_test, y_test

    def get_true_tagged_output(self):
        return self.true_test_output

    def fit(self, X, y):
        self.NPModel.fit(X, y)

    def predict(self, X): #dataset = Train or Test
        y_pred = self.NPModel.predict(X)
        return y_pred

    def pred_tags(self, y):
        pred_NPs = y2NP(y, self.retrieved_candidate_NPs)
        pred_test_output = tag_pred_labels(self.test_doc_tokens, pred_NPs)
        return pred_test_output

    def cross_validate_NP_model(self, X, scoring = 'accuracy'):
        print("\nNP Training Cross Validation")
        return self.NPModel.cross_validate(X, self.y, scoring)

    def evaluate_NP_model(self, y_true, y_pred):
        print("\nRaw NP model score:")
        self.NPModel.evaluate(y_true, y_pred)

    def evaluate_overall_model(self, y_true, y_pred):
        print("Final Scores for Vector Based Modes:")
        print(classification_report(y_pred=y_pred, y_true=y_true, labels=['CP']))


def prepare_VectorBased_dataset(load, n_train, n_test):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('VB_X_train')
        y_train = tmp_load('VB_y_train')
        doc2vec_model = get_model(Doc2Vec,'cur_doc_model.mdl')
        word2vec_model = get_model(Word2Vec, 'cur_model.mdl')
    if isinstance(X_train, type(None)) or not load:
        print('#Extracting features from training dataset ...')
        tagged_train_sent, true_CPs = load_tag_files(1,'Train', n_train)

        doc_tokens = sent2doc_tokens(tagged_train_sent)

        vocab = list(set([w for i, sents in doc_tokens.items() for s in sents for w in s]))

        tagged_train_output = [l for s in tagged_train_sent for l in sent2labels(s)]

        doc2vec_model  = create_doc2vec_model(doc_tokens)
        word2vec_model = create_word2vec_model(vocab)

        print("#Extracting noun phrases ...")
        retrieved_train_NPs = get_NP(doc_tokens)

        X_train, y_train = get_doc2vec_NP2vec(doc_tokens, retrieved_train_NPs, true_CPs, doc2vec_model, word2vec_model)

        tmp_save(X_train, 'VB_X_train')
        tmp_save(y_train, 'VB_y_train')

    if load:
        X_test = tmp_load('VB_X_test')
        y_test = tmp_load('VB_y_test')
        doc_tokens = tmp_load('VB_test_doc_tokens')
        tagged_test_output = tmp_load('VB_tagged_test_output')
        retrieved_test_NPs = tmp_load('VB_retrieved_test_NPs')

    if isinstance(X_test, type(None)) or not load:
        print('\n#Extracting features from test dataset ...')
        tagged_test_sent, true_CPs = load_tag_files(1,'Test', n_test)

        doc_tokens = sent2doc_tokens(tagged_test_sent)

        tagged_test_output = [l for s in tagged_test_sent for l in sent2labels(s)]

        print("#Extracting noun phrases ...")
        retrieved_test_NPs = get_NP(doc_tokens)

        X_test, y_test = get_doc2vec_NP2vec(doc_tokens, retrieved_test_NPs, true_CPs, doc2vec_model, word2vec_model)

        tmp_save(X_test, 'VB_X_test')
        tmp_save(y_test, 'VB_y_test')
        tmp_save(doc_tokens, 'VB_test_doc_tokens')
        tmp_save(tagged_test_output, 'VB_tagged_test_output')
        tmp_save(retrieved_test_NPs, 'VB_retrieved_test_NPs')

    return X_train, y_train, X_test, y_test, doc_tokens, tagged_test_output, retrieved_test_NPs


def sent2doc_tokens(sents):
    doc_tokens = defaultdict()
    for sent in sents:
        for i, [token, postag, label, doc_idx, tf] in enumerate(sent):
            if doc_idx in doc_tokens:
                if i == 0:
                    doc_tokens[doc_idx].append([token])
                else:
                    doc_tokens[doc_idx][-1].append(token)
            else:
                doc_tokens[doc_idx] = [[token]]
    return doc_tokens


def y2NP(y, retrieved_test_NPs):
    pred_NPs = set()
    j = 0
    for i, NPs in retrieved_test_NPs.items():
        for NP in NPs:
            if y[j] == 1:
                pred_NPs.update([NP])
            j+=1
    pred_NPs = list(map(lambda t: t.split(' '), list(pred_NPs)))
    return list(pred_NPs)


def tag_pred_labels(doc_tokens, pred_NPs):

    pred_output = []
    for i, sents in doc_tokens.items():
        for j, tokens in enumerate(sents):
            for k, token in enumerate(tokens):
                pred_output.append('O')
                filter_CP = list(filter(lambda t: token in t, pred_NPs))
                filter_CP = list(filter(lambda t: t == tokens[k-t.index(token):k-t.index(token)+len(t)], filter_CP))
                pred_output[-1] = 'CP' if filter_CP else 'O'

    return pred_output


def get_NP(doc_tokens):
    NPs = defaultdict()
    for doc_idx, doc in doc_tokens.items():
        for sent in doc:
            NPs_ret = list(set(get_noun_phrases(sent).keys()))
            if NPs_ret:
                if doc_idx in NPs:
                    NPs[doc_idx].extend( NPs_ret )
                else:
                    NPs[doc_idx] = list(set(get_noun_phrases(sent).keys()))
    return NPs


def get_doc2vec_NP2vec(doc_tokens, retrieved_NPs, true_CPs, doc2vec_model, word2vec_model):
    set_of_NP_X, total_vec = [], []
    for i in range(len(retrieved_NPs)):
        doc_vec = get_vector_embedding_for_docs(doc_tokens[i], doc2vec_model)

        for phrase in retrieved_NPs[i]:
            set_of_NP_X.append(phrase)
            phrase_vec = get_vector_embedding_for_NP(phrase,word2vec_model)
            total_vec.append(np.concatenate((phrase_vec, doc_vec)))

    X = np.array(total_vec)

    y = np.array(list(map(lambda t: 1 if t in true_CPs else 0, set_of_NP_X)))

    return X, y
