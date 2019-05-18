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
from collections import defaultdict, Counter
import pickle
import time
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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


def load_tag_files(task, mode, n_files):
    sentences = []
    CP_set = set()
    doc_freq = Counter()
    if mode == 'Train':
        start_idx = 0
    else:
        start_idx = 100
    for i in range(start_idx,start_idx+n_files):

        train_file = get_file_name(1, mode, 'docs', i)
        catch_file = get_file_name(1, mode, 'catches', i)

        catch_phrases = []
        with open(catch_file,encoding = "ISO-8859-1") as f:
            for line in f:
                for phrase in line.strip().split(','):

                    CP_set.update([phrase.strip().lower()])
                    catch_phrases.append(word_tokenize(phrase.strip().lower()))

        doc = []
        bag_of_words = {}
        with open(train_file,encoding = "ISO-8859-1") as f:
            for line in f:
                for sentence in sent_tokenize(line):
                    tokens = list(map(lambda t: t.lower(), word_tokenize(sentence)))
                    doc.append(tokens)
                    for token in tokens:
                        if token in bag_of_words:
                            bag_of_words[token] += 1
                        else:
                            bag_of_words[token] = 1
                            if token in doc_freq:
                                doc_freq[token] += 1
                            else:
                                doc_freq[token] = 1
        for words in doc:
            sentences.append([])
            POS_tags = pos_tag(words)
            next_label = 0
            for k, [word, tag] in enumerate(zip(words, POS_tags)):
                filter_CP = list(filter(lambda t: word.lower() in t, catch_phrases))
                filter_CP = list(filter(lambda t: t == words[k-t.index(word):k-t.index(word)+len(t)], filter_CP))
                label = 'CP' if filter_CP else 'O'

                sentences[-1].append([*tag, label, i-start_idx, bag_of_words[word]])

    sentences = [[[*t[:4],t[4]*np.log2(n_files/doc_freq[t[0]])] for t in sentence] for sentence in sentences]

    return sentences, list(CP_set)

def sent2labels(sent):
    return [label for token, postag, label, doc_idx, tf in sent]
