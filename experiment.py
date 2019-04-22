from misc import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    task1_n_train = 100
    task1_n_test = 300

    print("Reading Task1 Docs ...")
    vocab = tmp_load('vocab')
    train_docs_tokens_in = tmp_load('train_docs_tokens_in')

    if not vocab or not train_docs_tokens_in:
        train_docs_tokens_in, vocab = read_docs(0, task1_n_train, task=1, train_mode='Train', doc_type='docs')
        tmp_save(vocab,'vocab')
        tmp_save(train_docs_tokens_in, 'train_docs_tokens_in')

    train_docs_tokens_ot = tmp_load('train_docs_tokens_ot')
    if not train_docs_tokens_ot:
        train_docs_tokens_ot, _     = read_docs(0, task1_n_train, task=1, train_mode='Train', doc_type='catches')
        tmp_save(train_docs_tokens_ot, 'train_docs_tokens_ot')

    test_docs_tokens_in = tmp_load('test_docs_tokens_in')
    if not test_docs_tokens_in:
        test_docs_tokens_in, _      = read_docs(task1_n_train, task1_n_test, task=1, train_mode='Test', doc_type='docs')
        tmp_save(test_docs_tokens_in, 'test_docs_tokens_in')

    test_docs_tokens_ot = tmp_load('test_docs_tokens_ot')
    if not test_docs_tokens_ot:
        test_docs_tokens_ot, _      = read_docs(task1_n_train, task1_n_test, task=1, train_mode='Test', doc_type='catches')
        tmp_save(test_docs_tokens_ot, 'test_docs_tokens_ot')

    print("Extracting noun phrases ...")

    train_X_doc         = tmp_load('train_X_doc')
    train_set_of_NP_X   = tmp_load('train_set_of_NP_X')
    train_Y_doc         = tmp_load('train_Y_doc')
    train_set_of_NP_Y   = tmp_load('train_set_of_NP_Y')
    test_X_doc          = tmp_load('test_X_doc')
    test_set_of_NP_X    = tmp_load('test_set_of_NP_X')

    if not train_X_doc:
        train_X_doc = defaultdict()
        train_Y_doc = defaultdict()
        test_X_doc  = defaultdict()

        train_set_of_NP_X = set()
        train_set_of_NP_Y = set()
        test_set_of_NP_X  = set()


        for i in range(task1_n_train):
            train_X_doc[i] = set(get_noun_phrases(train_docs_tokens_in[i]).keys())
            train_set_of_NP_X.update(train_X_doc[i])
            train_Y_doc[i] = set(get_noun_phrases(train_docs_tokens_ot[i]).keys())
            train_set_of_NP_Y.update(train_Y_doc[i])

        for i in range(task1_n_test):
            test_X_doc[i] = set(get_noun_phrases(test_docs_tokens_in[i]).keys())
            test_set_of_NP_X.update(test_X_doc[i])

        #Convert from unordered to ordered collection
        train_set_of_NP_X = list(train_set_of_NP_X)
        train_set_of_NP_Y = list(train_set_of_NP_Y)
        test_set_of_NP_X  = list(test_set_of_NP_X )

        tmp_save(train_X_doc, 'train_X_doc')
        tmp_save(train_set_of_NP_X, 'train_set_of_NP_X')
        tmp_save(train_Y_doc, 'train_Y_doc')
        tmp_save(train_set_of_NP_Y, 'train_set_of_NP_Y')
        tmp_save(test_X_doc, 'test_X_doc')
        tmp_save(test_set_of_NP_X, 'test_set_of_NP_X')

    print("Creating vector embeddings ...")

    X_train = tmp_load('X_train')
    if type(X_train).__name__ == 'NoneType':
        X_train = get_vector_embedding_for_NPs(vocab, train_set_of_NP_X) #Option1 ignoring the frequency of NPs
        tmp_save(X_train, 'X_train')

    X_test = tmp_load('X_test')
    if type(X_test).__name__ == 'NoneType':
        X_test  = get_vector_embedding_for_NPs(vocab, test_set_of_NP_X ) #Option1 ignoring the frequency of NPs
        tmp_save(X_test, 'X_test')

    Y_train = tmp_load('Y_train')
    if type(Y_train).__name__ == 'NoneType':
        Y_train = np.zeros((X_train.shape[0],))
        for i, phrase in enumerate(train_set_of_NP_X):
            if phrase in train_set_of_NP_Y:
                Y_train[i] = 1
        tmp_save(Y_train, 'Y_train')


    print("Training Task1 ...")

    #Start Training (X,Y)
    Y_test_hat = tmp_load('Y_test_hat')
    if type(Y_test_hat).__name__ == 'NoneType':
        Y_test_hat = Default(X_train, X_test, Y_train)
        tmp_save(Y_test_hat, 'Y_test_hat')

    print("Printing Task1 Outputs ...")

    f = open('tmp_files/output_task1.tsv','w')
    for i in range(task1_n_test):
        f.write(str(i)+':\t')

        for noun_phrase in test_X_doc[i]:
            j = test_set_of_NP_X.index(noun_phrase)

            if Y_test_hat[j] == 1:
                f.write(str(noun_phrase)+', ')
        f.write('\n')
    f.close()
