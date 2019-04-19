from misc import *

if __name__ == '__main__':

    task1_n_train = 100
    task1_n_test = 300

    #Load Noun Phrases if already extracted from the training docs for task1
    noun_phrases = tmp_load('noun_phrases')
    vocab = tmp_load('vocab')
    if not noun_phrases:

        #Extract Noun phrases from the training docs for task1
        noun_phrases = defaultdict(int)

        docs_tokens, vocab = read_docs(task1_n_train, task=1, train_mode='Train', doc_type='docs')

        for i in range(task1_n_train):
            noun_phrases = get_noun_phrases(docs_tokens[i], noun_phrases)

        tmp_save(noun_phrases,'noun_phrases')
        tmp_save(vocab, 'vocab')

    #Create vectors for training for task 1
    X = get_vector_embedding_for_NPs(vocab, noun_phrases.keys()) #Option1 ignoring the frequency of NPs

    #Load golden phrases if already saved
    golden_phrases = tmp_load('golden_phrases')

    if not golden_phrases:
        #Extract Golden phrases from the training catches for task1
        golden_phrases = set()

        docs_tokens, _ = read_docs(task1_n_train, task=1, train_mode='Train', doc_type='catches')

        for i in range(task1_n_train):
            golden_phrases.update(docs_tokens[i])

        tmp_save(golden_phrases,'golden_phrases')

    Y = np.zeros((X.shape[0],))
    for i, phrase in enumerate(noun_phrases.keys()):
        if phrase in golden_phrases:
            Y[i] = 1


    print(X.shape,Y.shape)

    #Start Training (X,Y)

    #Load Test Dataset and extract noun_phrases

    #Predict for noun phrases in test Dataset

    #Add the predictions in the golden file

    #Evaluate by computing precision recall (Add functions)
