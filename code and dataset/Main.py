'''
witter User Gender Classification¶
Predict user gender based on text posted
The dataset from : https://www.kaggle.com/crowdflower/twitter-user-gender-classification
'''

#import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import mglearn
from sklearn.svm import SVC
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

def load_dataset():
    # Load dataset
    tweet_text = pd.read_csv("Tweet_data.csv")
    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    x_vali = pd.read_csv("x_validation.csv")
    y_vali = pd.read_csv("y_validation.csv")
    x_test = pd.read_csv("x_test.csv")
    y_test = pd.read_csv("y_test.csv")

    return tweet_text, x_train, y_train, x_vali, y_vali, x_test, y_test

def eda(df):
    print(df.head())
    print()

def visuallized(df):
    df.plot(kind="box", subplots = True, sharex=False, sharey=False)
    df.hist()
    plt.show()

def correlation(df):
    #Create correlation matrix
    corr_matrix = df.corr().abs()

    #Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    #Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

    #Drop Marked Features
    #df.drop(df[to_drop], axis=1)
    print (to_drop)

def svc(x_train, y_train, x_vali, y_vali):

    best_score = 0
    #Using Grid Search
    #to improve the model's generalization performance by tunning its parameters
    for gamma in [0.1, 1, 10]:
        for C in [0.1, 1, 10]:
            #for each combination of parameters, train an SVC
            svm = SVC(gamma=gamma, C=C)
            svm.fit(x_train, y_train)
            # evaluate the SVC on the test set
            score = svm.score(x_vali, y_vali)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}

    # We use validation to adjust its parameter
    print("Best score on validation: {:.2f}".format((best_score)))
    print("Best parameters: {}".format(best_parameters))

    #rebuild a model on the combined training and validation set,
    #and evaluate it on the test set

    svm = SVC(**best_parameters)
    svm.fit(x_train, y_train)
    test_score = svm.score(x_test, y_test)
    print("Test set score with best parameters: {:.2f}".format(test_score))

    ''' Best score on validation: 0.44
        Best parameters: {'C': 1, 'gamma': 0.1}
        Test set score with best parameters: 0.45'''

def nb(x_train, y_train, x_vali, y_vali, x_test, y_test):
    #fit the training dataset on the NB classifier
    nb_clf = naive_bayes.MultinomialNB()
    nb_clf.fit(x_train, y_train)

    # predict the labels on validation dataset
    nb_predict = nb_clf.predict(x_vali)

    # Accuracy
    print("Training accuracy :", nb_clf.score(x_train, y_train))
    print("Validation accuracy :", nb_clf.score(x_vali, y_vali))
    print("Testing accuracy :", nb_clf.score(x_test, y_test))

    # Evaluate predictions
    # use accuracy_score to get accuracy
    nb_score = accuracy_score(nb_predict, y_vali)*100
    print("Naive Bayes Accuracy Score: ", nb_score)
    #Naive Bayes Accuracy Score:  43.35839598997494
    print("This is Confusion matrix")
    print(confusion_matrix(y_vali, nb_predict))
    print("This is Classification report")
    print(classification_report(y_vali, nb_predict))

def svm_best(x_train, y_train, x_vali, y_vali, x_test, y_test):
    svm_clf = LinearSVC(C=10)
    svm_clf.fit(x_train, y_train)

    #Accuracy
    print("Training accuracy :", svm_clf.score(x_train, y_train))
    print("Validation accuracy :", svm_clf.score(x_vali, y_vali))
    print("Testing accuracy :", svm_clf.score(x_test, y_test))


    # predict the labels on validation dataset
    svm_predict = svm_clf.predict(x_test)

    # use accuracy_score to get accuracy
    svm_score = accuracy_score(svm_predict, y_test) * 100
    print("SVM Accuracy Score: ", svm_score)
    print("This is Confusion matrix")
    print(confusion_matrix(y_test, svm_predict))
    print("This is Classification report")
    print(classification_report(y_test, svm_predict))

def tunning_svm(Xtr, ytr, Xva, yva, Xts, yts):

    cp = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    acc_train = []
    acc_val = []
    acc_test = []
    best_score = 0

    for c in cp:
        linear_kernel = LinearSVC(C=c)
        linear_kernel.fit(Xtr, ytr) # fit the model
        acc_train.append(linear_kernel.score(Xtr, ytr))  #training accuracy
        acc_val.append(linear_kernel.score(Xva, yva))    #validation accuracy
        acc_test.append(linear_kernel.score(Xts, yts))   #testing accuracy
        score = linear_kernel.score(Xtr, ytr)
        if score > best_score:
            best_score = score
            best_parameters = {'C': c }

    print("Best score:", best_score)
    print("Best parameters: ", best_parameters)

    namec = ('0.001', '0.01', '0.1', '1', '10', '100', '1000')
    plt.plot(namec, acc_train, Label="Training")
    plt.plot(namec, acc_test, Label="Testing")
    plt.plot(namec, acc_val, Label="Validation")
    plt.ylabel("The Accuracy")
    plt.xlabel("C Parameter")
    plt.title("The Accuracy VS C plot")
    plt.legend()
    plt.show()

    print("Training Accuracy :", acc_train)
    print("Validation Accuracy :", acc_val)
    print("Testing Accuracy :", acc_test)

def lr(x_train, y_train, x_vali, y_vali, x_test, y_test):
    lr_clf = LogisticRegression(penalty="l1", C=1)
    lr_clf.fit(x_train, y_train)

    # predict the labels on validation dataset
    lr_predict =lr_clf.predict(x_vali)

    # Accuracy
    print("Training accuracy :", lr_clf.score(x_train, y_train))
    print("Validation accuracy :", lr_clf.score(x_vali, y_vali))
    print("Testing accuracy :", lr_clf.score(x_test, y_test))


    # use accuracy_score to get accuracy
    lr_score = accuracy_score(lr_predict, y_vali) * 100
    print("LogisticRegression Accuracy Score: ", lr_score)
    print("This is Confusion matrix")
    print(confusion_matrix(y_vali, lr_predict))
    print("This is Classification report")
    print(classification_report(y_vali, lr_predict))

def tunning_lr(x_train, y_train):
    logistic = LogisticRegression()
    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter space
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    # Create hyperparameter options

    hyperparameters = dict(C=C, penalty=penalty)

    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

    # Fit grid search
    best_model = clf.fit(x_train, y_train)

    # View best hyperparameters
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])

if __name__ == '__main__':
    tweet_text, x_train, y_train, x_vali, y_vali, x_test, y_test = load_dataset()

    #print("Class distribution of y_train")
    #print(y_train.groupby('LABLE').size())
    #print("Class distribution of y_vali")
    #print(y_vali.groupby('LABLE').size())
    #print("Class distribution of y_test")
    #print(y_test.groupby('LABLE').size())

    x_train = np.asarray(x_train, dtype=np.int64)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_vali = np.asarray(x_vali, dtype=np.int64)
    y_vali = np.asarray(y_vali, dtype=np.int64)
    x_test = np.asarray(x_test, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    print("X_train shape : ",x_train.shape)
    print("X_vali shape : ",x_vali.shape)
    print("X_test shape : ",x_test.shape)

    #svc(x_train, y_train, x_vali, y_vali)
    nb(x_train, y_train, x_vali, y_vali, x_test, y_test)
    #Naive Bayes Accuracy Score: 43.35839598997494
    #svm_best(x_train, y_train, x_vali, y_vali, x_test, y_test)
    #SVM Accuracy Score: 44.06015037593985
    #lr(x_train, y_train, x_vali, y_vali, x_test, y_test)
    #LogisticRegression Accuracy Score:  44.11027568922306
    #tunning_svm(x_train, y_train, x_vali, y_vali, x_test, y_test)
    #tunning_lr(x_train, y_train)
