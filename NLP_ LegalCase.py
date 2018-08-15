import pandas as pd
import numpy as np 
from numpy import *
from lxml import objectify
import io

import fileinput
import nltk as nl
from nltk.corpus import stopwords
set(stopwords.words('english'))
from nltk.tokenize import sent_tokenize,word_tokenize

#library for sentiment analysis 
import textblob

#PreProcessing and Modelling
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# imports needed and set up logging
import gensim as gm
import logging

import matplotlib.pyplot as plt
## comment the below line out in a python terminal or spyder,Optional
%matplotlib inline

def create_dict(keys, values):
    return dict(zip(keys, values + [None] * (len(keys) - len(values))))

## Dataset Preparation

basePath = '/Users/abhirammaddali/Downloads/'
nextPath = 'corpus1/fulltext/'
numeric='06_'
#sec2='06_4'
form = '.xml'

def folderLoader(basePath,nextPath,numeric,form,nofile):
    listfiles=[]
    df=pd.DataFrame(columns=('name','AustLII','sentenceID','sentence'))
    final=pd.DataFrame(columns=('name','AustLII','sentenceID','sentence'))
    num=[i for i in range(1,nofile)]
    nm=[i for i in range(1,nofile)]
    print (nm)
    for i in num:
        filePath = basePath + nextPath+ numeric+str(nm[i-1])+form
        listfiles.append(filePath)
    for path in listfiles:
        try:
            xml = objectify.parse(open(path))
            root = xml.getroot()
            length_of_file=len(root.getchildren()[3].getchildren())
            for i in range(0,length_of_file):
                list1=[]
                name=root.getchildren()[0].text
                list1.append(name)
                Austlii=root.getchildren()[1].text
                list1.append(Austlii)
                list1.append(i+1)
                r1=root.getchildren()[3].getchildren()[i].text
                list1.append(r1)
                dictr=create_dict(('name','AustLII','sentenceID','sentence'),list1)
                row_s = pd.Series(dictr)
                df=df.append(row_s,ignore_index=True)
            final=final.append(df)
            #print(final)
        except FileNotFoundError:
                  pass
    return final

u_06=folderLoader(basePath=basePath,nextPath=nextPath,numeric=numeric,form=form,nofile=136)

print(u_06.head())

numeric='07_'
u_07=folderLoader(basePath=basePath,nextPath=nextPath,numeric=numeric,form=form,nofile=136)

print(u_07.head())

## testing data 09
basePath = '/Users/abhirammaddali/Downloads/'
nextPath = 'corpus1/fulltext/'
numeric='09_'
#sec2='06_4'
form = '.xml'
test_09=folderLoader(basePath=basePath,nextPath=nextPath,numeric=numeric,form=form,nofile=15)

print(test_09.head())

# Data Cleaning & Tokenization 

df_train=pd.DataFrame()
df_train=df_train.append(u_06,ignore_index=True)

df_train=df_train.append(u_07,ignore_index=True)

print(df_train.info())

def correct_text(dataframe, remove_stopwords=True,toke_sent=False):
    ##Converting into Lower case 
    dataframe['sentence'] = dataframe['sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Deleting puntuation
    dataframe['sentence'] = dataframe['sentence'].str.replace('[^\w\s]','')
    # Tokenizing words
    dataframe['tokenized_words'] = dataframe.apply(lambda row: nl.word_tokenize(row['sentence']), axis=1)
    if remove_stopwords:
        ## setting the stop words
        stop = stopwords.words('english')
        dataframe['tokenized_words']=dataframe['tokenized_words'].apply(lambda x: [item for item in x if item not in stop])
    if toke_sent:
        #tokenizing sentenses
        df_text['tokenized_sents'] = df_text.apply(lambda row: nl.sent_tokenize(row['sentence']), axis=1)
    return dataframe

# Classification Dataset Preparation 

df_text=correct_text(df_train)

def sentiment_analysis(row_data):
    blb = textblob.TextBlob(row_data)
    return blb.sentiment.polarity

df_text['sentiment'] = df_text.loc[:,'sentence'].apply(sentiment_analysis)

df_text['sentiment_class']  =df_text.sentiment.map(lambda x: 'Negative' if x < 0 else ('Positive' if x > 0 else 'Neutral'))

print(df_text.head())

### Word2vec model 

documents= list(df_text.tokenized_words)

# Importing the built-in logging module
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Creating the model and setting values for the various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-5 # (0.0001) Downsample setting for frequent words

model = gm.models.Word2Vec(documents,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

words = list(model.wv.vocab)
print(words)

X = model[model.wv.vocab]

#### PCA analysis 

pca = PCA(n_components=100)
result = pca.fit_transform(X)
print(result)

plt.scatter(result[:, 0], result[:, 1])
plt.show()
### Converstion of traning data into feature matrix

# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))  
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1  
    return reviewFeatureVecs

# Calculating average feature vector for training set
train_reviews = []
for sentence in df_text['sentence']:
    train_reviews.append(sentence)
trainData = getAvgFeatureVecs(train_reviews, model, num_features)

#### Cleaning and vectorizing the test data 

df_test=correct_text(test_09,remove_stopwords=True)
print(df_test)

# Calculating average feature vector for training set
test_reviews = []
for sentence in df_test['sentence']:
    test_reviews.append(sentence)
    
testData = getAvgFeatureVecs(test_reviews, model, num_features)

# Making null values to 0 in traning data
where_NaNs = isnan(trainData)
trainData[where_NaNs] = 0

# Making null values to 0 in testing data
where_NaNs = isnan(testData)
testData[where_NaNs] = 0
#Scalar conversion of traning data 
scalar=StandardScaler()
x_enc=scalar.fit_transform(trainData)
#encoder = MultiColumnLabelEncoder()

# Data Modelling 

#### Rndom Forest Classifier

forest=RandomForestClassifier(n_estimators = 100)
forest = forest.fit(trainData, df_text["sentiment_class"])
pred = forest.predict(testData)
output = pd.DataFrame(data={"id":df_test["sentence"], "sentiment":pred})
output.to_csv("/Users/abhirammaddali/Downloads/result.csv")