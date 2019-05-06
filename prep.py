from misc import *

def load_data_structure(mode, start_idx, n_files, load):

    docs_tokens_in = tmp_load(mode+'_docs_tokens_in', load)
    learn_mode = 'Test' if mode == 'test' else 'Train'
    if not docs_tokens_in:
        docs_tokens_in, vocab = read_docs(start_idx, n_files, 1, learn_mode, 'docs')
        tmp_save(docs_tokens_in, mode+'_docs_tokens_in')
        if mode == 'train':
            tmp_save(vocab,'vocab')

    docs_tokens_ot = tmp_load(mode+'_docs_tokens_ot', load)
    if not docs_tokens_ot:
        docs_tokens_ot, _      = read_docs(start_idx, n_files, 1, learn_mode, 'catches')
        tmp_save(docs_tokens_ot, mode+'_docs_tokens_ot')

def get_numpy_vectors(mode, n_files, doc2vec_model, word2vec_model, load):
    X_doc   = tmp_load('X_doc_' + mode, load)
    X       = tmp_load('X_'     + mode, load)
    Y       = tmp_load('Y_'     + mode, load)

    if not X_doc:
        X_doc = defaultdict()
        total_vec, set_of_NP_X = [], []
        docs_tokens_in = tmp_load(mode+'_docs_tokens_in', True)
        docs_tokens_ot = tmp_load(mode+'_docs_tokens_ot', True)

        for i in range(n_files):
            X_doc[i] = list(set(get_noun_phrases(docs_tokens_in[i]).keys()))

            doc_vec = get_vector_embedding_for_docs(docs_tokens_in[i], doc2vec_model)

            for phrase in X_doc[i]:
                set_of_NP_X.append(phrase)
                phrase_vec = get_vector_embedding_for_NP(phrase,word2vec_model)
                total_vec.append(np.concatenate((phrase_vec, doc_vec)))

        X = np.array(total_vec)

        set_of_NP_Y = list(set([phrase for doc in docs_tokens_ot for phrase in doc]))
        Y = np.array(list(map(lambda t: 1 if t in set_of_NP_Y else 0, set_of_NP_X)))

        tmp_save(X_doc, 'X_doc_' + mode)
        tmp_save(X,     'X_'     + mode)
        tmp_save(Y,     'Y_'     + mode)

    return X, Y

def prepare_dataset(task1_n_train = 100,n_test = 300,train_dev_split=0.7,load=True):
    print("Reading Task1 Docs ...")

    n_train = int(task1_n_train*train_dev_split)
    n_dev = task1_n_train - n_train

    load_data_structure('train', 0, n_train, load)
    load_data_structure('dev', n_train, n_dev, load)
    load_data_structure('test', task1_n_train, n_test, load)

    print("Extracting noun phrases ...")

    doc2vec_model  = create_doc2vec_model(tmp_load('train_docs_tokens_in'),model=False)
    word2vec_model = create_word2vec_model(tmp_load('vocab'), model=False)

    X_train, Y_train = get_numpy_vectors('train', n_train, doc2vec_model, word2vec_model, load)
    X_dev,   Y_dev   = get_numpy_vectors('dev',   n_dev,   doc2vec_model, word2vec_model, load)
    X_test , Y_test  = get_numpy_vectors('test',  n_test,  doc2vec_model, word2vec_model, load)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
