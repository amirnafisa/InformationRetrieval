import os
import sys

import nltk
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from gensim.models import Word2Vec
from collections import defaultdict
import pickle

VEC_SIZE = 100
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


def read_docs(start_idx, n_files, task=1, train_mode='Train', doc_type='docs'):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = []
    vocab = []


    for i in range(n_files):

        file = get_file_name(task, train_mode, doc_type, start_idx+i)

        with open(file,encoding = "ISO-8859-1") as f:
            docs.append([])
            for line in f:
                sentences = sent_tokenize(line) if doc_type == 'docs' else line.split(',')

                for sentence in sentences:
                    word_tokens = word_tokenize(sentence) if doc_type == 'docs' else [sentence]
                    if word_tokens:
                        word_tokens = list(map(lambda x:x.lower().strip(),word_tokens))
                        docs[i].extend(word_tokens)
                        vocab.extend([word_tokens])

    return docs, vocab


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


def get_model(file):
    file = os.path.join(models_dir,file)
    if os.path.isfile(file):
        return Word2Vec.load(file)
    return None


def save_model(model, file):

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    model.save(os.path.join(models_dir,file))


def get_vector_embedding_for_NPs(vocab, phrases, model=True):

    doc_vecs = np.zeros((len(phrases),VEC_SIZE))

    if model:
        model = get_model('cur_model.mdl')
    if not model:

        model = Word2Vec(vocab, min_count=1, size=VEC_SIZE, window=5) #CBOW Model from gensim
        save_model(model, 'cur_model.mdl')

    for i, phrase in enumerate(phrases):
        doc_vecs[i] = get_vector_embedding(model, phrase)

    return doc_vecs


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

def get_vector_embedding(model, phrase):

    vec = np.zeros((VEC_SIZE,))
    for word in phrase.split(' '):
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


def tmp_load(file, load):
    if not load:
        return None
    dir = 'tmp_files'
    file = os.path.join(dir,file+'.lst')
    if os.path.isfile(file):
        with open(file,'rb') as fp:
            return pickle.load(fp)
    return None

def write_output(Y_test_hat,test_X_doc,test_set_of_NP_X,start_idx,ext='Train'):
    print("Printing Task1 Outputs ...")

    dir = 'output/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    dir = 'output/'+ext+'/'
    if not os.path.exists(dir):
        os.mkdir(dir)

    print(Y_test_hat.shape)
    for i in range(len(test_X_doc)):
        f = open(dir+'case_'+str(int(i+start_idx))+'_catchwords.txt','w')

        for noun_phrase in test_X_doc[i]:
            j = test_set_of_NP_X.index(noun_phrase)

            if Y_test_hat[j] == 1:
                f.write(str(noun_phrase)+', ')
        f.close()

def evaluate_task1(start_idx, n_files, folder):
    precision = np.zeros((n_files))
    recall = np.zeros_like(precision)

    for j in range(n_files):
        i = j+start_idx
        file_true = get_file_name(1, folder, 'catches', i)
        file_hat = 'output/'+folder+'/case_'+str(i)+'_catchwords.txt'

        with open(file_true) as f:
            lines = f.readline()
            list_true = set(lines.strip().lower().split(','))


        with open(file_hat) as f:
            lines = f.readline()
            list_hat = set(lines.strip().lower().split(','))

        precision[j], recall[j] = compute_prec_recall(list_true, list_hat)

        print(j, precision[j], recall[j], sep='\t')

    return precision, recall


def compute_prec_recall(golden_truth, prediction):

    common_set = set(golden_truth).intersection(set(prediction))
    print(common_set)

    precision = round(len(common_set)/len(prediction), 4)

    recall = round(len(common_set)/len(golden_truth), 4)



    return precision, recall
