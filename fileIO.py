import os

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize


def get_file_name(proj_dir, task=1, train_mode='Train', doc_type='docs'):
    if doc_type == 'docs':
        filename = 'Task_' + str(task) + '/' + train_mode + '_docs/case_' + str(i) + '_statement.txt'
    elif doc_type == 'catches':
        filename = 'Task_' + str(task) + '/' + train_mode+'_catches/case_' + str(i) + '_catchwords.txt'
    elif doc_type == 'current_cases':
        filename = 'Task_' + str(task) + '/Current_Cases/current_case_' + f"{i:04d}" + '.txt'
    elif doc_type == 'prior_cases':
        filename = 'Task_' + str(task) + '/Prior_Cases/prior_case_' + f"{i:04d}" + '.txt'
    elif doc_type == 'irled-qrel':
        filename = 'Task_' + str(task) + '/' + doc_type + '.txt'
    return os.path.join(proj_dir,filename)

def read_docs(file, mode='case'):
    '''
    Reads the corpus into a list of Documents
    '''
    NP = []
    NNP = []
    grammar = r"""
        NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
        NP: {<NN>*<NNS>*<NNP>*}      # Chunk sequences of NN and NNS
    """
    my_parser = nltk.RegexpParser(grammar)

    with open(file) as f:
        for line in f:
            line = line.strip()
            sentences = sent_tokenize(line)
            for sentence in sentences:
                print("LINE:\n",sentence,"\n\n")
                text = word_tokenize(sentence)
                print("TEXT:\n",text,"\n\n")
                word_tagged_text = nltk.pos_tag(text)
                print("TAGS:\n",word_tagged_text,"\n\n")
                phrase_tagged_text = my_parser.parse(word_tagged_text)
                print("PHRS:\n",phrase_tagged_text,"\n\n")
                NP += [[leaf[0] for leaf in subtree.leaves()] for subtree in phrase_tagged_text.subtrees(filter=lambda t: t.label()=='NP')]
                NNP += [word for (word,tag) in word_tagged_text if tag=='NN' or tag=='NNP' or tag=='NNS']

    noun_phrases = NP+list(set(NNP))
    print("NPs:\n",noun_phrases,"\n\n")
    print()

    return noun_phrases

if __name__ == '__main__':
    proj_dir = 'FIRE2017-IRLeD-track-data'

    task1_n_train = 100
    task1_n_test = 300

    for i in range(task1_n_train):
        read_docs(get_file_name(proj_dir, task=1, train_mode='Train', doc_type='docs'))
        break
