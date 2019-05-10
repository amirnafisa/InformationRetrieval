from models import *
from misc import *
from prep import *

def base_model():
    preload = True
    task1_n_train = 100
    task1_n_test = 300
    train_dev_split = 0.7

    X_train,Y_train,X_dev,Y_dev,X_test,Y_test = prepare_dataset(task1_n_train,task1_n_test,train_dev_split,preload)

    print(X_train.shape)

    print("Training Task1 ...")
    #Start Training (X,Y) Use Development dataset for training
    classifier = Default #Could be Default
    Y_dev_hat = classifier(X_train, X_dev, Y_train)
    tmp_save(Y_dev_hat, 'Y_dev_hat')

    #Compute training accuracy for choosing candidate noun phrases irrespective of document
    #print("Raw Noun Phrases: Dev Accuracy:",compute_acc(Y_dev,Y_dev_hat))

    #Write corresponding predicted catchphrases for development set
    X_doc_dev = tmp_load('X_doc_dev', True)
    write_output(Y_dev_hat,X_doc_dev,task1_n_train*train_dev_split,ext='Train')

    #Evaluate the development predictions
    print("IDX","precision","recall",sep='\t')
    precision, recall = evaluate_task1(start_idx=int(task1_n_train*train_dev_split),n_files=int(task1_n_train*(1-train_dev_split)), folder='Train')
    print("\n\n\n",round(np.mean(precision), 4),round(np.mean(recall), 4))



    ##Test the model
    #classifier = Default #Could be Default
    #Y_test_hat = classifier(X_train, X_test, Y_train)
    #tmp_save(Y_test_hat, 'Y_test_hat')

    ##Compute training accuracy for choosing candidate noun phrases irrespective of document
    #print("Raw Noun Phrases: Test Accuracy:",compute_acc(Y_test,Y_test_hat))

    ##Write corresponding predicted catchphrases for development set
    #X_doc_test = tmp_load('X_doc_test', True)
    #write_output(Y_test_hat,X_doc_test,task1_n_train,ext='Test')

    ##Evaluate the development predictions
    #print("IDX","precision","recall",sep='\t')
    #precision, recall = evaluate_task1(start_idx=task1_n_train,n_files=task1_n_test, folder='Test')
    #print("\n\n\n",round(np.mean(precision), 4),round(np.mean(recall), 4))
