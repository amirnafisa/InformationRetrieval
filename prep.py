from misc import *

def prepare_dataset(task1_n_train = 100,task1_n_test = 300,train_dev_split=0.7,load=True):
    print("Reading Task1 Docs ...")

    n_train = int(task1_n_train*train_dev_split)
    n_dev = task1_n_train - n_train

    vocab = tmp_load('vocab', load)

    train_docs_tokens_in = tmp_load('train_docs_tokens_in', load)
    if not vocab or not train_docs_tokens_in:
        train_docs_tokens_in, vocab = read_docs(start_idx=0, n_files=n_train, task=1, train_mode='Train', doc_type='docs')
        tmp_save(vocab,'vocab')
        tmp_save(train_docs_tokens_in, 'train_docs_tokens_in')

    train_docs_tokens_ot = tmp_load('train_docs_tokens_ot', load)
    if not train_docs_tokens_ot:
        train_docs_tokens_ot, _     = read_docs(start_idx=0, n_files=n_train, task=1, train_mode='Train', doc_type='catches')
        tmp_save(train_docs_tokens_ot, 'train_docs_tokens_ot')

    dev_docs_tokens_in = tmp_load('dev_docs_tokens_in', load)
    if not dev_docs_tokens_in:
        dev_docs_tokens_in, _      = read_docs(start_idx=n_train, n_files=n_dev, task=1, train_mode='Train', doc_type='docs')
        tmp_save(dev_docs_tokens_in, 'dev_docs_tokens_in')

    dev_docs_tokens_ot = tmp_load('dev_docs_tokens_ot', load)
    if not dev_docs_tokens_ot:
        dev_docs_tokens_ot, _      = read_docs(start_idx=n_train, n_files=n_dev, task=1, train_mode='Train', doc_type='catches')
        tmp_save(dev_docs_tokens_ot, 'dev_docs_tokens_ot')

    test_docs_tokens_in = tmp_load('test_docs_tokens_in', load)
    if not test_docs_tokens_in:
        test_docs_tokens_in, _      = read_docs(start_idx=task1_n_train, n_files=task1_n_test, task=1, train_mode='Test', doc_type='docs')
        tmp_save(test_docs_tokens_in, 'test_docs_tokens_in')

    test_docs_tokens_ot = tmp_load('test_docs_tokens_ot', load)
    if not test_docs_tokens_ot:
        test_docs_tokens_ot, _      = read_docs(start_idx=task1_n_train, n_files=task1_n_test, task=1, train_mode='Test', doc_type='catches')
        tmp_save(test_docs_tokens_ot, 'test_docs_tokens_ot')

    print("Extracting noun phrases ...")

    train_X_doc         = tmp_load('train_X_doc', load)
    train_set_of_NP_X   = tmp_load('train_set_of_NP_X', load)
    train_Y_doc         = tmp_load('train_Y_doc', load)
    train_set_of_NP_Y   = tmp_load('train_set_of_NP_Y', load)

    dev_X_doc           = tmp_load('dev_X_doc', load)
    dev_set_of_NP_X     = tmp_load('dev_set_of_NP_X', load)
    dev_Y_doc           = tmp_load('dev_Y_doc', load)
    dev_set_of_NP_Y     = tmp_load('dev_set_of_NP_Y', load)

    test_X_doc          = tmp_load('test_X_doc', load)
    test_set_of_NP_X    = tmp_load('test_set_of_NP_X', load)
    test_Y_doc          = tmp_load('test_Y_doc', load)
    test_set_of_NP_Y    = tmp_load('test_set_of_NP_Y', load)

    if not train_X_doc:
        train_X_doc = defaultdict()
        train_Y_doc = defaultdict()
        dev_X_doc   = defaultdict()
        dev_Y_doc   = defaultdict()
        test_X_doc  = defaultdict()
        test_Y_doc  = defaultdict()

        train_set_of_NP_X = set()
        train_set_of_NP_Y = set()
        dev_set_of_NP_X   = set()
        dev_set_of_NP_Y   = set()
        test_set_of_NP_X  = set()
        test_set_of_NP_Y  = set()


        for i in range(n_train):
            train_X_doc[i] = set(get_noun_phrases(train_docs_tokens_in[i]).keys())
            train_set_of_NP_X.update(train_X_doc[i])
            train_Y_doc[i] = set(train_docs_tokens_ot[i])
            train_set_of_NP_Y.update(train_Y_doc[i])

        for i in range(n_dev):
            dev_X_doc[i] = set(get_noun_phrases(dev_docs_tokens_in[i]).keys())
            dev_set_of_NP_X.update(dev_X_doc[i])
            dev_Y_doc[i] = set(dev_docs_tokens_ot[i])
            dev_set_of_NP_Y.update(dev_Y_doc[i])

        for i in range(task1_n_test):
            test_X_doc[i] = set(get_noun_phrases(test_docs_tokens_in[i]).keys())
            test_set_of_NP_X.update(test_X_doc[i])
            test_Y_doc[i] = set(test_docs_tokens_ot[i])
            test_set_of_NP_Y.update(test_Y_doc[i])

        #Convert from unordered to ordered collection
        train_set_of_NP_X = list(train_set_of_NP_X)
        train_set_of_NP_Y = list(train_set_of_NP_Y)
        dev_set_of_NP_X   = list(dev_set_of_NP_X)
        dev_set_of_NP_Y   = list(dev_set_of_NP_Y)
        test_set_of_NP_X  = list(test_set_of_NP_X )
        test_set_of_NP_Y  = list(test_set_of_NP_Y)

        tmp_save(train_X_doc, 'train_X_doc')
        tmp_save(train_set_of_NP_X, 'train_set_of_NP_X')
        tmp_save(train_Y_doc, 'train_Y_doc')
        tmp_save(train_set_of_NP_Y, 'train_set_of_NP_Y')
        tmp_save(dev_X_doc, 'dev_X_doc')
        tmp_save(dev_set_of_NP_X, 'dev_set_of_NP_X')
        tmp_save(dev_Y_doc, 'dev_Y_doc')
        tmp_save(dev_set_of_NP_Y, 'dev_set_of_NP_Y')
        tmp_save(test_X_doc, 'test_X_doc')
        tmp_save(test_set_of_NP_X, 'test_set_of_NP_X')
        tmp_save(test_Y_doc, 'test_Y_doc')
        tmp_save(test_set_of_NP_Y, 'test_set_of_NP_Y')

    print("Creating vector embeddings ...")

    X_train = tmp_load('X_train', load)
    if type(X_train).__name__ == 'NoneType':
        X_train = get_vector_embedding_for_NPs(vocab, train_set_of_NP_X) #Option1 ignoring the frequency of NPs
        tmp_save(X_train, 'X_train')

    X_dev = tmp_load('X_dev', load)
    if type(X_dev).__name__ == 'NoneType':
        X_dev = get_vector_embedding_for_NPs(vocab, dev_set_of_NP_X) #Option1 ignoring the frequency of NPs
        tmp_save(X_dev, 'X_dev')

    X_test = tmp_load('X_test', load)
    if type(X_test).__name__ == 'NoneType':
        X_test  = get_vector_embedding_for_NPs(vocab, test_set_of_NP_X ) #Option1 ignoring the frequency of NPs
        tmp_save(X_test, 'X_test')

    Y_train = tmp_load('Y_train', load)
    if type(Y_train).__name__ == 'NoneType':
        Y_train = np.zeros((X_train.shape[0],))
        for i, phrase in enumerate(train_set_of_NP_X):
            if phrase in train_set_of_NP_Y:
                Y_train[i] = 1

        tmp_save(Y_train, 'Y_train')

    Y_dev = tmp_load('Y_dev', load)
    if type(Y_dev).__name__ == 'NoneType':
        Y_dev = np.zeros((X_dev.shape[0],))
        for i, phrase in enumerate(dev_set_of_NP_X):
            if phrase in dev_set_of_NP_Y:
                Y_dev[i] = 1
        tmp_save(Y_dev, 'Y_dev')

    Y_test = tmp_load('Y_test', load)
    if type(Y_test).__name__ == 'NoneType':
        Y_test = np.zeros((X_test.shape[0],))
        for i, phrase in enumerate(test_set_of_NP_X):
            if phrase in test_set_of_NP_Y:
                Y_test[i] = 1
        tmp_save(Y_test, 'Y_test')

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test
