# Catchphrase Extraction from Legal Documents

[Code](https://github.com/amirnafisa/InformationRetrieval)

### ABSTRACT
In this paper we analyze Vector based models, Conditional Random Field (CRF) based models, Recurrent Neural Net- work (RNN) based models (Long Short Term Memory Net- work) for the task of catchphrase extraction in the legal docu- ments. We run experiments and analyze techniques and fea- tures that influence our design and scores. We evaluate using the standard precision, recall and F1 score measure. We cul- minate by proposing a model that combines both LSTM and CRF.

### Keywords
Vector based; Conditional Random Field; Recurrent Neural Network; Long Short Term Memory; catchphrase extraction; legal.

### INTRODUCTION
With the constant increase in the number of legal doc- uments, the need for automatically processing legal docu- ments has only increased. One of the tasks involved in build- ing such systems is catchphrase extraction. Catchphrases give a gist of what the legal document is about. This later helps other law researchers decide if the document is rele- vant to them or not.
Legal documents are generally long and complicated. Au- tomatically extracting catchphrases is thus a complex task and requires models to truly understand what the document
is about. We have used the IRLeD dataset[1] that contains cases from Indian Supreme Court with are 100 training and 300 test documents (annotated). Catchphrase extraction is known to have been done mostly on corpuses from Wikipedia[2] or research and scientific documents[3] since it is more eco- nomic to get their annotated datasets. These works though not in the legal domain have guided us in our project through various techniques in information retrieval (IR).
In this project, we analyze various models and propose the one with the best expected F1 score. In the next section we describe briefly our models. Then we succinctly mention the results. In the conclusion section, we briefly analyze the results. Eventually we propose some tasks which given more time, we would have liked to include in this project but were not able to.

### PROJECT PIPELINE
We implemented Vector based IR, IR using conditional random field (CRF), IR using Long Short Term Memory (LSTM) based recurrent neural network (RNN) and IR us- ing LSTM and CRF combined.

##### Vector based IR
Running instruction is python3 experiment.py Vector 100 50. Here the argument Vector specifies the vector based model, the arguments to its right (100 50) specify number of training files (max 100) and number of test files (max 300). We recommend not to use all 300 test files in the vector based IR since the post-processing step of converting the labels to tags per document is bottleneck in terms of time.
The motivation for using Vector based IR came from the homeworks and is used to establish a baseline to compare and illustrate the need for other models described in follow- ing section. Vector based IR is implemented as follows:
• Extracted noun phrases (NPs) using parts of speech tagger provided in NLTK toolkit and a regexp parser from all training documents
• The grammar rules used for NP extraction are: – NP−><JJ|NN.∗>+
– NP− >< NN > ∗ < NNS > ∗ < NNP > ∗ • Created a list of catch phrases (CP) as marked in the
training documents (mark others as O)
• Embedded each noun phrase as a vector (NP_vec) by averaging the vectors obtained for each word using gen- sim model
• Embedded each document as a vector (doc_vec) using gensim model
• Appended doc_vec to each NP_vec
• Separated the vectors into two sets, one marked with
CP and the other with O
• Computed centroids for each set
• Using cosine similarity between the centroids and the test vectors, predict the label (CP/O) for each of the test noun phrase vector

##### IR using CRF
Running instruction is python3 experiment.py CRF 100 50.
The task of CP extraction here and in the following models is treated as named entity recognition (NER) task. The se- quence of words in each document is maintained and tagged with corresponding parts of speech tag, labels (CP/O), doc- ument index and term weights. The label is used for super- vised training only.
Letx1,x2,,xN bethesequenceofwordsandt1,t2,,tN be corresponding labels where ∀i, ti ∈ [CP, O]. The CRF models p(t1, t2, , tN |x1, x2, , xN ) by a set of features defined as follows:
• Word
• Stemmed word (SnowballStemmer from NLTK)
• 1 if word is numeric only else 0
• Parts of speech tag
• Term weight (TF-logIDF)
• Length of the word
• Position of the word in the sentence
• 1 if any non-alphanumeric character present in the word else 0
• Word to the left
• Second word to the left
• Word to the right
• Second word to the right
• True if Beginning of Sentence (BOS)
• True if End of Sentence (EOS)
• 1 if Previous word’s POS tag is ‘JJ’ and current word’s POS tag is ‘NN’ else 0
While developing CRF we played with various features and used the ones that had significant impact. The top 10 feature weights for CRF are:

| Feature key        | Feature Value | Pred  | Weight  |
| ------------------ |:-------------:| -----:| -------:|
| word.lower()       | accessories   | CP    | 5.92    |
| +2:word.lower()    | sumac         | CP    | 5.57    |
| word.lower()       | minorities    | CP    | 5.32    |
| word.lower()       | nomination    | CP    | 5.25    |
| word.lower()       | financing     | CP    | 5.21    |
| -2:word.lower()    | adventure     | CP    | 5.03    |
| -1:word.lower()    | novopan       | CP    | 4.98    |
| word.lower()       | constituency  | CP    | 4.96    |
Table 1: High weighted features

##### IR using LSTM
Running instruction is python3 experiment.py LSTM 100 50.
The recurrent neural networks have been proven to be suc- cessful for sequence tagging problems with significant im- provements using long short term memory networks. Here we have used the keras LSTM package and implemented as follows:

- Padded each sentence with âA ̆ŸPADGARBAGEâA ̆Z ́ to make all sentences of uniform length of 50
- Converted all words to numerical indices (unique index for a unique word)
- Converted all labels to numerical indices (unique index for a unique label)
- Createdtheneuralnetworkmodelcontainingfollowing layers:
    – Added first layer with input dimension as many as number of unique words and output dimension of size 50
    – Added a dropout layer (dropout of 0.1)
    – Optimizer used is RMS Prop
    – Loss function used is Cross Entropy loss

##### IR using LSTM and CRF combined
Running instruction is python3 experiment.py LSTM- CRF 100 50.
We used keras packages for adding an extra layer of keras CRF after the LSTM as mentioned in section 2.3 to create a new model. However in the keras CRF, we have not used our custom feature set but relied on the keras utils to extract the essential features.

### RESULTS AND EVALUATION
| Model        | Precision  | Recall  | F1 Score  |
| ------------- |:-------------:| -----:|-----:|
| Vector        | 0.10          | 0.93  | 0.18 |
| CRF           | 0.62          | 0.36  | 0.46 |
| LSTM          | 0.69          | 0.21  | 0.33 |
| LSTM+CRF      | 0.53          | 0.36  | 0.43 |
Table 2: Performance of various models 

### CONCLUSION
We find that statistical features along with deep learning models perform the best. The baseline model has the best recall with the lowest precision. Since, extracting based on NPs lead to the extraction of many more false CPs along with the true CPs. This alone is therefore not a reliable way of CP extraction. CRF significantly improves precision and F1 score. LSTM can improve precision significantly by being more accurate in predicted the true CPs. However its recall is somewhat lower than the CRF. The LSTM and CRF com- bined yields better results compared to LSTM alone, how- ever f1 score is very close to that of CRF.

### FUTURE WORK
We find that a certain fraction of the catchphrases are not extracted by our Noun Phrase extractor. This is because the NP rules we use are not extensive. A good next step would be using SyntaxNet or Stanford CoreNLP parser to extract Noun Phrases.
Another extension would be to use a better approach to model Noun Phrases. Currently, we average the word em- beddings of all words present in the NP. Instead, we could map all of the embeddings to a fixed vector space using BERT.
With respect to CRF and LSTM based model, we pro- pose to use custom CRF features to the model which we be- lieve should significantly improve the F1 score suprpassing all other models so far since the results prove the effective- ness of the custom features included in the CRF. Secondly we would like to play around with LSTM model by chang- ing hyper-parameters some more.
The second task in the work is that of precedence retrieval which can make use of our catchphrase extraction technique. Future work should also attempt to broaden the range of ap- plications of the current methodology in the legal IR.

### REFERENCES
[1] Mandal, Arpan, et al. Overview of the FIRE 2017 IRLeD Track: Information Retrieval from Legal Documents. FIRE, 2017
[2] Arora, S., Liang, Y., Ma, T. A simple but tough-to-beat baseline for sentence embeddings. 2016
[3] Hulth, Anette Improved automatic keyword extraction given more linguistic knowledge. Association for Computational Linguistics, 2003
