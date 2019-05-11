from misc import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict

def get_NP(doc_tokens):
    NPs = defaultdict()
    for doc_idx, doc in doc_tokens.items():
        for sent in doc:
            NPs_ret = list(set(get_noun_phrases(sent).keys()))
            if NPs_ret:
                if doc_idx in NPs:
                    NPs[doc_idx].append( NPs_ret )
                else:
                    NPs[doc_idx] = list(set(get_noun_phrases(sent).keys()))
    return NPs

def get_doc2vec_NP2vec(doc_tokens, retrieved_NPs, retrieved_CPs, doc2vec_model, word2vec_model):
    set_of_NP_X, total_vec = [], []
    for i in range(len(retrieved_NPs)):
        doc_vec = get_vector_embedding_for_docs(doc_tokens[i], doc2vec_model)

        for phrase in retrieved_NPs[i]:
            set_of_NP_X.append(phrase)
            phrase_vec = get_vector_embedding_for_NP(phrase,word2vec_model)
            total_vec.append(np.concatenate((phrase_vec, doc_vec)))

    X = np.array(total_vec)

    y = np.array(list(map(lambda t: 1 if t in retrieved_CPs else 0, set_of_NP_X)))

    return X, y


def prepare_dataset(load):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('VB_X_train')
        y_train = tmp_load('VB_y_train')
        doc2vec_model = get_model(Doc2Vec,'cur_doc_model.mdl')
        word2vec_model = get_model(Word2Vec, 'cur_model.mdl')
    if isinstance(X_train, type(None)) or not load:
        print('Extracting features from training dataset ...')
        tagged_train_sent, retrieved_CPs = load_tag_files(1,'Train')

        doc_tokens, vocab = sent2doc_tokens(tagged_train_sent)

        tagged_train_output = [sent2labels(s) for s in tagged_train_sent]

        print("DOC TOKENS",len(doc_tokens))
        doc2vec_model  = create_doc2vec_model(doc_tokens)
        word2vec_model = create_word2vec_model(vocab)

        print("Extracting noun phrases ...")
        retrieved_train_NPs = get_NP(doc_tokens)

        print('Creating vector embedding training dataset ...')
        X_train, y_train = get_doc2vec_NP2vec(doc_tokens, retrieved_train_NPs, retrieved_CPs, doc2vec_model, word2vec_model)

        tmp_save(X_train, 'VB_X_train')
        tmp_save(y_train, 'VB_y_train')

    if load:
        X_test = tmp_load('VB_X_test')
        y_test = tmp_load('VB_y_test')
        tagged_test_sent = tmp_load('VB_tagged_test_sent')
        tagged_test_output = tmp_load('VB_tagged_test_output')
        retrieved_test_NPs = tmp_load('VB_retrieved_test_NPs')

    if isinstance(X_test, type(None)) or not load:
        print('Extracting features from test dataset ...')
        tagged_test_sent, retrieved_CPs = load_tag_files(1,'Test')

        doc_tokens, _ = sent2doc_tokens(tagged_test_sent)

        tagged_test_output = [sent2labels(s) for s in tagged_test_sent]

        retrieved_test_NPs = get_NP(doc_tokens)

        X_test, y_test = get_doc2vec_NP2vec(doc_tokens, retrieved_test_NPs, retrieved_CPs, doc2vec_model, word2vec_model)

        tmp_save(X_test, 'VB_X_test')
        tmp_save(y_test, 'VB_y_test')
        tmp_save(tagged_test_sent, 'VB_tagged_test_sent')
        tmp_save(tagged_test_output, 'VB_tagged_test_output')
        tmp_save(retrieved_test_NPs, 'VB_retrieved_test_NPs')

    return X_train, y_train, X_test, y_test, tagged_test_sent, tagged_test_output, retrieved_test_NPs


def prepare_lstm_dataset(load=True, max_len=50):
    #Extract features and create dataset
    X_train, X_test = None, None
    if load:
        X_train = tmp_load('lstm_X_train')
        y_train = tmp_load('lstm_y_train')
    if not X_train or not load:


        sent = tmp_load('sent_train')
        if type(sent) == type(None):
            print('Loading sentences training dataset ...')
            sent, _ = load_tag_files(1,'Train')
            tmp_save(sent,'sent_train')

        vocab = list(set([w[0] for s in sent for w in s]))
        vocab.append('PADGARBAGE')
        n_vocab = len(vocab)
        labels = list(set([labels for s in sent for labels in sent2labels(s) ]))
        n_labels = len(labels)

        word2idx = {w: i for i, w in enumerate(vocab)}
        label2idx = {t: i for i, t in enumerate(labels)}

        X_train = tmp_load('lstm_X_train')
        y_train = tmp_load('lstm_y_train')
        if type(X_train) == type(None):
            print('Extracting features from training dataset ...')
            X_train = [[word2idx[w[0]] for w in s] for s in sent]
            X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=n_vocab - 1)
            y_train = [[label2idx[w[2]] for w in s] for s in sent]
            y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=label2idx["O"])

            y_train = [to_categorical(i, num_classes=n_labels) for i in y_train]

            tmp_save(X_train, 'lstm_X_train')
            tmp_save(y_train, 'lstm_y_train')


        X_test = tmp_load('lstm_X_test')
        y_test = tmp_load('lstm_y_test')
        if type(X_test) == type(None):
            print('Extracting features from testing dataset ...')
            sent, _ = load_tag_files(1,'Test')

            X_test = [[word2idx[w[0]] if w[0] in word2idx else word2idx['PADGARBAGE'] for w in s] for s in sent]
            X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=n_vocab - 1)
            y_test = [[label2idx[w[2]] for w in s] for s in sent]
            y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=label2idx["O"])

            y_test = [to_categorical(i, num_classes=n_labels) for i in y_test]
            tmp_save(X_test, 'lstm_X_test')
            tmp_save(y_test, 'lstm_y_test')
    return X_train, y_train, X_test, y_test, vocab, labels
