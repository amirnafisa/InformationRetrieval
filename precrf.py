def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
    'postag': postag,
    'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def extract_features(task, mode):
    sent = []
    if mode == 'Train':
        start_idx = 0
        n_files = 100
    else:
        start_idx = 100
        n_files = 300
    for i in range(start_idx,start_idx+n_files):

        train_file = get_file_name(1, mode, 'docs', i)
        catch_file = get_file_name(1, mode, 'catches', i)

        catch_phrases = []
        with open(catch_file,encoding = "ISO-8859-1") as f:
            for line in f:
                for phrase in line.strip().split(','):
                    catch_phrases.append(word_tokenize(phrase.strip().lower()))

        with open(train_file,encoding = "ISO-8859-1") as f:
            for line in f:
                sentences = sent_tokenize(line)
                for j, sentence in enumerate(sentences):
                    sent.append([])
                    words = word_tokenize(sentence)
                    POS_tags = pos_tag(words)
                    next_label = 0
                    for k, [word, tag] in enumerate(zip(words, POS_tags)):
                        [label, next_label, labeled_already] = ['I-CP', next_label-1, True] if next_label > 0 else ['O', 0, False]

                        if next_label == 0 and not labeled_already:
                            filter_CP = list(filter(lambda t: word in t, catch_phrases))
                            max_len = 0
                            for CP in filter_CP:

                                l = len(CP)
                                idx = CP.index(word)
                                [label, max_len, next_label] = ['B-CP', l, l-1] if idx == 0 and l > max_len and CP == words[k-idx:k-idx+l] else [label, max_len, next_label]

                        sent[-1].append((*tag, label))
    return sent
