#!/usr/bin/env python3
from misc import *
from prep import *
from models import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    preload = True
    task1_n_train = 100
    task1_n_test = 300
    train_dev_split = 0.7

    X_train,Y_train,X_dev,Y_dev,X_test,Y_test = prepare_dataset(task1_n_train,task1_n_test,train_dev_split,preload)

    print("Training Task1 ...")
    #Start Training (X,Y) Use Development dataset for training
    classifier = Default #Could be Default
    Y_dev_hat = classifier(X_train, X_dev, Y_train)
    tmp_save(Y_dev_hat, 'Y_dev_hat')

    #Compute training accuracy for choosing candidate noun phrases irrespective of document
    print("Raw Noun Phrases: Dev Accuracy:",compute_acc(Y_dev,Y_dev_hat))

    #Write corresponding predicted catchphrases for development set
    dev_X_doc = tmp_load('dev_X_doc', True)
    dev_set_of_NP_X = tmp_load('dev_set_of_NP_X', True)
    write_output(Y_dev_hat,dev_X_doc,dev_set_of_NP_X,task1_n_train*train_dev_split,ext='Train')

    #Evaluate the development predictions
    precision, recall = evaluate_task1(start_idx=int(task1_n_train*train_dev_split),n_files=int(task1_n_train*(1-train_dev_split)), folder='Train')
    print("\n\n\n",round(np.mean(precision), 4),round(np.mean(recall), 4))



    ##Test the model
    #classifier = Default #Could be Default
    #Y_test_hat = classifier(X_train, X_test, Y_train)
    #tmp_save(Y_test_hat, 'Y_test_hat')

    ##Compute training accuracy for choosing candidate noun phrases irrespective of document
    #print("Raw Noun Phrases: Test Accuracy:",compute_acc(Y_test,Y_test_hat))

    ##Write corresponding predicted catchphrases for development set
    #test_X_doc = tmp_load('test_X_doc', True)
    #test_set_of_NP_X = tmp_load('test_set_of_NP_X', True)
    #write_output(Y_test_hat,test_X_doc,test_set_of_NP_X,task1_n_train,ext='Test')

    ##Evaluate the development predictions
    #precision, recall = evaluate_task1(start_idx=task1_n_train,n_files=task1_n_test, folder='Test')
    #print("\n\n\n",round(np.mean(precision), 4),round(np.mean(recall), 4))
