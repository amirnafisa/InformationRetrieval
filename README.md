# InformationRetrieval
Date: April 18, 2019

Issues: 1. Removal of isolated punctuations needed before tagging or noun phrase extraction.
        2. Multiple sentences in a line. (Using nltk sentence tokenizer)
        3. find grammar rules for NP extraction.
        4. Should we remove suffixes 's' or 'es' or 'd' or  'ed'???
############################################################3
Date: April 18, 2019

Dataset: FIRE 2017 Legal information retrieval

Task1: Catchphrase extraction (Train on 100 cases and test on 300 cases)

Step1- Noun phrase extraction using POS tagger

Create Training set as follows:
  Option1:
            Let X be the training data of shape N by M, where N is the number of noun phrases in all the training documents and M is length of phrase feature vector for noun phrase.

            Terms = Bag of noun phrases extracted
            For each term in Terms:
              Compute phrase feature vector = average of word embeddings (words in the phrase)

  Option2:
            Slight deviation from option1,
            Terms = bag of noun-phrases along with frequency with which they appear in a document.
            Eg, Terms = [land-1, land-2, high-court-2, proceedings-4, proceedings-6]

            For each term in Terms:
              Compute phrase feature vector = [average of word embeddings] + [freq]


Step2: Train the model
      Use a machine learning method to output the probability of a phrase/term to be a Catchphrase candidate

Step3: Validation/Test method
      For each document d in the validation set:
            Extract noun-phrases (with frequency for option2) from d;
                  For each term in noun phrase/withfreq:
                        pred = model.predict(term)

                        if the term is predicted as a candidate then append to the list of Catchphrases[d]

Step 4: Evaluation:
      Let G be the set of true catchphrases for document i,
      Let O be the set of predicted catchphrases for the document i.

      Precision[i] = #{O intersection G}/#{O}
      Recall[i] = #{O intersection G}/#{G}
      F1[i] = 2 * Precision[i] * Recall[i] / (Precision[i]+Recall[i])

      Overall Precision = mean(Precision)
      Overall Recall = mean(Recall)
      Overall F1 = mean(F1)


Task2: Preceedence Retrieval (200 cases and 2000 prior cases to look up)

Step 1: Catchphrase extraction of the input cases (queries) and prior cases (documents)

Step 2: Compute vectors for all documents and queries

Step 3: For each query:
            compute similarity of vectors
            Rank based on similarity
            Extract 5 best ranks
            Use  machine learning to improve the ranking

Reference-
Overview of the FIRE 2017 IRLeD Track: Information Retrieval from Legal Documents
Improved Automatic Keyword Extraction Given More Linguistic Knowledge
A Simple but Tough-to-Beat Baseline for Sentence Embeddings

############################################################3
#Older Proposals -

1. Legal IR - (Presidential)
Dataset: https://www.presidency.ucsb.edu/documents/app-categories/presidential?page=7387
Not annotated: No evaluation, but interesting

2. Legal IR - (Terms and Conditions)
----- Couldn't find annotated dataset ------

3. Bilingual Dataset
http://www.cs.jhu.edu/~kevinduh/a/wikiclir2018/
https://pdfs.semanticscholar.org/2b6c/68e0cb1a6a6e4092cc335cc0110d3583e035.pdf


Task:
INPUT TO THE MODEL: Documents = <Foreign/English Language Documents>
INPUT TO THE MODEL: Query = <Foreign/English Language Query>

OUTPUT OF THE MODEL: Rank of documents for the query

EVALUATION: Supervised Dataset

MODEL: Supervised/Semisupervised learning
      1. Compare dictionary based and translation based models
      2. Compare RNN vs Vector based models

DEPENDENCY:
      1. Dataset: CLIR JHU (Kevin Duh)
      2. Bilingual Dictionary: Wikipedia based

MODEL EITHER CASE:
      1. Base model (similar to assignments)
      2. Additional (compare RNN based and vector based models)
      3. compare semi-supervised and supervised learning evaluation
      4. Extending model for Tnc in another domain
      5.
