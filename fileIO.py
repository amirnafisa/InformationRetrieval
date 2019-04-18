def get_file_name(file, ext):
    return file + '-' + ext + '.tsv'

def read_docs(file, tokenize):
    '''
    Reads the corpus into a list of Documents
    '''
    docs,labels = [],[]

    with open(file) as f:
        for line in f:
            line = line.strip()
            if tokenize:
                words = word_tokenize(line)
            else:
                words = re.split('\t| ',line)

            for j, word in enumerate(words):
                if j == 0:      #New document
                    docs.append([])
                elif j == 1:    #Sense of the word in the document
                    labels.append(int(word))
                elif j > 1:     #Text/Features of the document
                    docs[-1].append(word)

    return docs,labels
