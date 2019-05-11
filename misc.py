import os
import sys

import nltk
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import defaultdict
import pickle



VEC_SIZE = 20
DOC_VEC_SIZE = 5
proj_dir = 'FIRE2017-IRLeD-track-data'
models_dir = 'models'


def get_file_name(task=1, train_mode='Train', doc_type='docs', idx=0):
    if doc_type == 'docs':
        filename = 'Task_' + str(task) + '/' + train_mode + '_docs/case_' + str(idx) + '_statement.txt'
    elif doc_type == 'catches':
        filename = 'Task_' + str(task) + '/' + train_mode+'_catches/case_' + str(idx) + '_catchwords.txt'
    elif doc_type == 'current_cases':
        filename = 'Task_' + str(task) + '/Current_Cases/current_case_' + f"{idx:04d}" + '.txt'
    elif doc_type == 'prior_cases':
        filename = 'Task_' + str(task) + '/Prior_Cases/prior_case_' + f"{idx:04d}" + '.txt'
    elif doc_type == 'irled-qrel':
        filename = 'Task_' + str(task) + '/' + doc_type + '.txt'
    return os.path.join(proj_dir,filename)

def get_noun_phrases(doc_tokens):
    '''
    Reads the corpus into a list of Documents
    '''
    noun_phrases = defaultdict(int)
    my_parser = nltk.RegexpParser (
            r"""
                NP: {<JJ|NN.*>+}             # Chunk sequences of JJ, NN
                NP: {<NN>*<NNS>*<NNP>*}      # Chunk sequences of NN and NNS
            """
        )

    word_tagged_text = pos_tag(doc_tokens)
    phrase_tagged_text = my_parser.parse(word_tagged_text)

    for subtree in phrase_tagged_text.subtrees(filter=lambda t: t.label()=='NP'):
        noun_phrases[' '.join(leaf[0] for leaf in subtree.leaves())] += 1
    for (word,tag) in filter(lambda tag: tag=='NN' or tag=='NNP' or tag=='NNS',word_tagged_text):
        noun_phrases[word] += 1

    return noun_phrases


def get_model(model, file):
    file = os.path.join(models_dir,file)
    if os.path.isfile(file):
        return model.load(file)
    return None


def save_model(model, file):

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    model.save(os.path.join(models_dir,file))


def create_word2vec_model(vocab):

    model = Word2Vec(vocab, min_count=1, size=VEC_SIZE, window=5) #CBOW Model from gensim
    save_model(model, 'cur_model.mdl')
    return model

def create_doc2vec_model(documents):

    tag_documents = [TaggedDocument([word for flatten_doc in doc for word in flatten_doc], [i]) for i, doc in documents.items()]
    model = Doc2Vec(tag_documents, vector_size=DOC_VEC_SIZE, window=2, min_count=1, workers=4)
    save_model(model, 'cur_doc_model.mdl')
    return model

def get_vector_embedding_for_docs(doc, model):
    flatten_doc = [word for flatten_doc in doc for word in flatten_doc]
    return model.infer_vector(flatten_doc)

def compute_centroid(vecs):
    return np.mean(vecs,axis=0)

def compute_acc(Y_hat,Y):
    return round(np.mean(Y_hat==Y), 4)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    if norm(x) == 0 or norm(y) == 0:
        return 0
    return np.dot(x.T,y)/(norm(x) * norm(y))

def get_vector_embedding_for_NP(phrase, model):

    vec = np.zeros((VEC_SIZE,))
    for word in phrase:
        if word in model.wv:
            vec += model.wv[word]
    vec /= len(phrase)

    return vec


def tmp_save(item, file):
    dir = 'tmp_files'
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir,file+'.lst'), 'wb') as fp:
        pickle.dump(item, fp, protocol=pickle.HIGHEST_PROTOCOL)


def tmp_load(file, load=True):
    if not load:
        return None
    dir = 'tmp_files'
    file = os.path.join(dir,file+'.lst')
    if os.path.isfile(file):
        with open(file,'rb') as fp:
            return pickle.load(fp)
    return None

def crf_debug_print_file(y_pred):

    sent = tmp_load('crf_X_test_sent')
    with open(proj_dir+'/tmp_files/crf_pred_debug.tsv','w') as f:
        for i in range(len(sent)):
            for j in range(len(sent[i])):
                if y_pred[i][j] == 'CP' and j == 0:
                    f.write(sent[i][j][0])
                elif y_pred[i][j] == 'CP' and j > 0:
                    f.write(' '+sent[i][j][0])
                elif j > 0:
                    if y_pred[i][j-1] == 'CP':
                        f.write(', ')
            f.write('\n')

def prepare_crf_dataset(load=True):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('crf_X_train')
        y_train = tmp_load('crf_y_train')
    if not X_train or not load:
        print('Extracting features from training dataset ...')
        sent, _ = load_tag_files(1,'Train')
        tmp_save(sent,'crf_X_train_sent')
        X_train = [sent2features(s) for s in sent]
        y_train = [sent2labels(s) for s in sent]

        tmp_save(X_train, 'crf_X_train')
        tmp_save(y_train, 'crf_y_train')

    if load:
        X_test = tmp_load('crf_X_test')
        y_test = tmp_load('crf_y_test')
    if not X_test or not load:
        print('Extracting features from test dataset ...')
        sent, _ = load_tag_files(1,'Test')
        tmp_save(sent,'crf_X_test_sent')
        X_test = [sent2features(s) for s in sent]
        y_test = [sent2labels(s) for s in sent]

        tmp_save(X_test, 'crf_X_test')
        tmp_save(y_test, 'crf_y_test')

    return X_train, y_train, X_test, y_test

def load_tag_files(task, mode):
    sent = []
    CP_set = set()
    if mode == 'Train':
        start_idx = 0
        n_files = 100
    else:
        start_idx = 100
        n_files = 300
    for i in range(start_idx,start_idx+n_files):

        train_file = get_file_name(1, mode, 'docs', i)
        catch_file = get_file_name(1, mode, 'catches', i)

        catch_phrases = []
        with open(catch_file,encoding = "ISO-8859-1") as f:
            for line in f:
                for phrase in line.strip().split(','):

                    CP_set.update([phrase.strip().lower()])
                    catch_phrases.append(word_tokenize(phrase.strip().lower()))

        with open(train_file,encoding = "ISO-8859-1") as f:
            for line in f:
                sentences = sent_tokenize(line)
                for j, sentence in enumerate(sentences):
                    sent.append([])
                    words = word_tokenize(sentence)
                    POS_tags = pos_tag(words)
                    next_label = 0

                    for k, [word, tag] in enumerate(zip(words, POS_tags)):
                        filter_CP = list(filter(lambda t: word in t, catch_phrases))
                        filter_CP = list(filter(lambda t: t == words[k-t.index(word):k-t.index(word)+len(t)], filter_CP))
                        label = 'CP' if filter_CP else 'O'

                        sent[-1].append((*tag, label, i-start_idx))
    return sent, list(CP_set)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
    'postag': postag,
    'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label, doc_idx in sent]

def sent2tokens(sent):
    return [token for token, postag, label, doc_idx in sent]

def sent2doc_tokens(sents):
    doc_tokens = defaultdict()
    vocab = set()
    for sent in sents:
        for i, [token, postag, label, doc_idx] in enumerate(sent):
            vocab.update(token)
            if doc_idx in doc_tokens:
                if i == 0:
                    doc_tokens[doc_idx].append([token])
                else:
                    doc_tokens[doc_idx][-1].append(token)
            else:
                doc_tokens[doc_idx] = [[token]]
    return doc_tokens, vocab

def get_predicted_NPs(y, retrieved_test_NPs):
    pred_NPs = set()
    j = 0
    for i in range(len(retrieved_test_NPs)):
        for phrase in retrieved_test_NPs[i]:
            if y[j] == 1:
                pred_NPs.update(phrase)
            j+=1
    pred_NPs = list(map(lambda t: t.split(' '), list(pred_NPs)))
    return list(pred_NPs)

def tag_pred_labels(tagged_test_sents, pred_NPs):
    pred_output = []
    for j, sent in enumerate(tagged_test_sents):
        pred_output.append([])
        tokens = sent2tokens(sent)
        for k, token in enumerate(tokens):
            filter_CP = list(filter(lambda t: token in t, pred_NPs))
            filter_CP = list(filter(lambda t: t == tokens[k-t.index(token):k-t.index(token)+len(t)], filter_CP))
            label = 'CP' if filter_CP else 'O'
            pred_output[j].append(label)
    return pred_output
