from misc import *

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
