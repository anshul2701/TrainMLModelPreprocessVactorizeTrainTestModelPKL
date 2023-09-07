import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn import metrics
# from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from collections import Counter
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

import collections 
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# import nltk 
nltk.download('punkt')
from nltk.corpus import stopwords 
nltk.download('stopwords')

from io import StringIO

def read_data():

    df = pd.read_excel('C:\python CSV\commonRTTMS\DataAnalysis2latest.xlsx')

    df.set_index('Comments and Work notes')
    df.fillna('')
    df.reset_index(inplace = True)
    df1 = df['Comments and Work notes']
    df2 = df['Response'].replace(['Yes','No'],[1,0])
    indices = df['index']
    # print(indices)
    return df1, df2, indices

# read_data()

def preprocess(Worknote):
    
    stop_words = stopwords.words('english')
    stop_words.extend(['u', 'wa', 'ha', 'would', 'com'])

# --------------------------------------------

    Worknote = Worknote.str.replace(r'[^\w\s]', '')
    print("removing special characters...")
    print(Worknote)
# ---------------------------------------------
    
    Worknote = Worknote.str.replace('\d*','')
    print("Remove digits....")
    print(Worknote)

# ---------------------------------------------

    Worknote = Worknote.replace("www", "") 
    print("Removing www ...")  
    print(Worknote) 

# ---------------------------------------------

    Worknote = Worknote.replace("http\S+", "")
    print("Removing https ...")
    print(Worknote)

# ---------------------------------------------

    Worknote = Worknote.replace('\s+', ' ')
    print("Removing Single Spacing...")
    print(Worknote)

# ---------------------------------------------

    Worknote = Worknote.replace(r'\s+[a-zA-Z]\s+', '')
    print("Removing all single characters...")
    print(Worknote)

    print("*********************************")
    print(Worknote)
    print("*********************************")
    # Worknote = Worknote.split() 
    Worknote = Worknote.astype(str)
    filtered_words = Worknote.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)])).astype(str)
    
    # print(df1)
    # filtered_words = [word for word in Worknote if word not in stop_words] 
    
    print("Print the filtered text")
    print(filtered_words)
    return filtered_words

# --------------------Preprocess function End-----------------------

    # url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def TFIDFvectorization(fil_word):
    vectorizer = TfidfVectorizer()
    #feature extraction method
    X_tfidf = vectorizer.fit_transform(fil_word)
    pickle.dump(vectorizer, open('vectorizerCT10.pkl', 'wb'))
    print(X_tfidf)
    return X_tfidf


def traintestsplit(X_tfidf,df2,indices):
    X_train,X_test,y_train,y_test,indices_train,indices_test = train_test_split(X_tfidf,df2,indices,test_size = .2, random_state =42)
    return X_train,X_test,y_train,y_test,indices_train,indices_test

def modelTrain(X_train,X_test,y_train,y_test):

    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X_train,y_train)

    class_weights = {0:100,
                     1:.5}
    
    svm_model = SVC(class_weight=class_weights, kernel= 'linear')
    svm_model.fit(X_res, y_res)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(accuracy)
    print(f1)
    print(precision)
    print(recall)

    print(metrics.classification_report(y_test, y_pred))
    
    return svm_model

def pklSaveFile(svm_model):
    with open('C:\python CSV\commonRTTMS\label10Updated.pkl', 'wb') as file:
        # Dump the data into the file
        pickle.dump(svm_model, file)

df1, df2, indices = read_data()

filtered_words = preprocess(df1)

X_tfidf = TFIDFvectorization(filtered_words)

X_train = traintestsplit(X_tfidf, df2, indices)[0]
X_test = traintestsplit(X_tfidf, df2, indices)[1]
y_train = traintestsplit(X_tfidf, df2, indices)[2]
y_test = traintestsplit(X_tfidf, df2, indices)[3]

svm_model = modelTrain(X_train,X_test,y_train,y_test)
pklSaveFile(svm_model)
