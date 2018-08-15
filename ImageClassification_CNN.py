# This is a use case for image recognition. The primary goal is to recognize the letters from 0-9, A-Z and a-z. The dataset for the same is available @ http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/. Download the english character images ( 64 classes in all ) from the website.

import os #To read length of the files
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Image reading library used in this implementation is PIL
from PIL import Image

# Importing all libraries for processing the data before it can be modeled.
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Importing all the classifiers that we are going to be implementing in this demo.
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

# Importing the metrics we are going to be using to measure the results of the model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

### Performing CNN Required Libraries
#from keras.utils.np_utlis import to_categorical
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D, MaxPool2D


# Although image imports can be automated using 'os' package, the following function semi-automates it as it is processing the image each time it is imported.
def folderLoader(basePath, numFiles, nextPath='', numeric='', section2='',imageFormat=''):
    listFiles = []
    for i in range(numFiles):
        if i < 10:
            section3 = '-0000'
        elif i < 100:
            section3 = '-000'
        elif i < 1000:
            section3 = '-00'
        elif i < 10000:
            section3 = '-0'
        else:
            section3 = '-'
        filePath = basePath + nextPath + numeric + section2 + numeric + section3 + str(i) + imageFormat
        listFiles.append(filePath)
    listFiles = listFiles[1:]
    finalList = []
    for path in listFiles:
        im = Image.open(path, 'r')
        resizedIm = im.resize([50,50])
        #convimg=resizedIm.convert('L') ### conversion of Image
        image = np.asarray(resizedIm)
        imageThreshold = threshold(image)
        #plt.imshow(imageThreshold)
        #plt.show()
        imageRow = imageThreshold.flatten()
        # print(imageRow[5:10])
        #imageRowFinal = np.array(list(imageRow).extend([targetLabel]))
        #print(imageRowFinal[5:10])
        finalList.append(imageRow)
    df = pd.DataFrame(finalList)
    #finalDF = df.T
    return df
    
def numfile(basepath, nextpath,numeric):
    filePath = basePath + nextPath+ numeric
    #filepath='/Users/abhirammaddali/Documents/Itlirogram /Python/text-recgonition /EnglishImg/Img/GoodImg/Bmp'
    lt = os.listdir(filePath) # dir is your directory path
    return (len(lt))

# The following function is a threshold calculator which calculates the mean for each pixel and if the pixel value is greater than the threshold then assigns it white color and if less than the mean then black. Primarily it is coverting multi color images to binary color images.
def threshold(imageArray):
    balanceAr = []
    for eachRow in imageArray:
        for eachPix in eachRow:
            avgNum = mean(eachPix[:3])
            balanceAr.append(avgNum)
    balance = mean(balanceAr)
    matimh=[]
    for eachRow in imageArray:
        newrow=[]
        for eachPix in eachRow:
            if mean(eachPix[:3]) > balance:
                newrow.append(255)
            else:
                newrow.append(0)
        matimh.append(newrow)
    return np.array(matimh)
    
# The following section is where each folder is being traversed and the functions defined above are being called to get the data sourced.
df = pd.DataFrame()
basePath = '/Users/abhirammaddali/Documents/Itlirogram /Python/text-recgonition /EnglishImg'
nextPath = '/Img/GoodImg/Bmp/Sample'
numeric='001'
sec2 = '/img'
form = '.png'

# Reading all files for Alphabet A
numeric='011'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'A'
df = df.append(dfin, ignore_index=True)

## Printing the A image 
im = np.array(df.drop(['Class'], axis=1))
plt.figure(figsize=(15,15))
y_axis, x_axis = 5,4
for i in range(y_axis*x_axis):
    plt.subplot(y_axis, x_axis, i+1)
    rand=np.random.randint(0,im.shape[0],1)
    img=np.reshape(im[rand],(50,50))
    plt.imshow(img,cmap='gray')
plt.show()
df.shape


# Reading all files for Alphabet B
numeric='012'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'B'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet C
numeric='013'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'C'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet D
numeric='014'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'D'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet E
numeric='015'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'E'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet F
numeric='016'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'F'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet G
numeric='017'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'G'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet H
numeric='018'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'H'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet I
numeric='019'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'I'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet J
numeric='020'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'J'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet K
numeric='021'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'K'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet L
numeric='022'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'L'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet M
numeric='023'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'M'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet N
numeric='024'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'N'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet O
numeric='025'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'O'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet P
numeric='026'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'P'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet Q
numeric='027'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'Q'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet R
numeric='028'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'R'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet S
numeric='029'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'S'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet T
numeric='030'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'T'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet U
numeric='031'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'U'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet V
numeric='032'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'V'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet W
numeric='033'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'W'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet X
numeric='034'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'X'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet Y
numeric='035'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'Y'
df = df.append(dfin, ignore_index=True)

# Reading all files for Alphabet Z
numeric='036'
dfin = folderLoader(basePath, numFiles=numfile(basePath,nextPath,numeric), nextPath=nextPath, numeric=numeric, section2=sec2, imageFormat=form)
dfin['Class'] = 'Z'
df = df.append(dfin, ignore_index=True)

print(df.info())
## saving the file to disk for easy read later.
df.to_csv("/Users/abhirammaddali/Documents/Itlirogram /Python/final2.csv")
df.tail()

## Preparing for Training and Testing 
df2=pd.DataFrame()
df2=df
X = np.array(df.drop(['Class'], axis=1))
y = np.array(df['Class'])


# Scaling the predictors for better model performance.
scaler= StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
## Making Class categorical to apply CNN
#y=keras.utils.np_utils.to_categorical(y_enc, num_classes=26)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2)

kfold = KFold(n_splits=5)

# 2. XGB classifier
xgb=XGBClassifier(max_depth=3, n_estimators=400,learning_rate=0.01,silent=0,objective='softmax',num_class=26,subsample=0.4)
xgb.fit(X_train, y_train)
y_preds = xgb.predict(X_test)
print("\n\n XGB boosting Accuracy score is: ", accuracy_score(y_test,y_preds))
print("\n XGB boosting Confusion matrix is: \n",confusion_matrix(y_test,y_preds))

# 2. MLP classifier tuned for better prediction performance
nn = MLPClassifier(learning_rate='adaptive', learning_rate_init=0.001, alpha=0.0001, max_iter=10000, hidden_layer_sizes=(1000,1000))
scores = cross_val_score(nn, X_train, y_train, cv=kfold, n_jobs=-1)
print ("\n\nScores MLP with tuning",scores)
nn.fit(X_train, y_train)
y_preds = nn.predict(X_test)
print("\n\n Neural Network with tuning accuracy score is: \n", accuracy_score(y_test,y_preds))
print("\n Neural Network with tuning confusion matrix is: \n",confusion_matrix(y_test,y_preds))

## Diving 
X1= np.array(df2.drop(['Class'], axis=1))
y1= np.array(df2['Class'])


## Scaling the predictors for better model performance.
#scaler= StandardScaler()
#X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y1)
## Making Class categorical to apply CNN
y2=keras.utils.np_utils.to_categorical(y_enc, num_classes=26)

## Checking th shape of CNN 
print ("Shape of X: ",X1.shape)
print ("Shape of y: ",y1.shape)

#Spliting the data
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y2, test_size=0.2)

## Normilizing X_Train 
X1_train=X1_train/255
X1_train 

## Reshaping X_train to Input into CNN 
X1_train=X1_train.reshape(-1,50,50,1)
print ("Shape of X_train: ",X1_train.shape)
print ("Shape of y_trian: ",y1_train.shape)


## Reshaping X_test to Input into CNN 
X1_test=X1_test.reshape(-1,50,50,1)
print ("Shape of X_train: ",X1_test.shape)
print ("Shape of y_trian: ",y1_test.shape)

## Creation of CNN model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu', input_shape = (50,50,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(26, activation = "softmax"))
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X1_train, y1_train, batch_size = 95, epochs = 20, validation_data = (X1_test,y1_test))
model.evaluate(X_test,y_test)

#### Tuned CNN model 
model2 = Sequential()
##First Convulution Layer 
model2.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu', input_shape = (50,50,1)))
##Second Convulution Layer
model2.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
##Third Convulution Layer
model2.add(Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
##ANN Start point 
model2.add(Flatten())
model2.add(Dense(256, activation = "relu"))
model2.add(Dropout(0.50))
##Output Layer 
model2.add(Dense(26, activation = "softmax"))
model2.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

model2.summary()

model2.fit(X1_train, y1_train, batch_size = 95, epochs = 20, validation_data = (X1_test,y1_test))

model2.evaluate(X1_test,y1_test)

## Testing the model
A_val=[]
def testDFCreator(filename):
    tf = Image.open(filename, 'r')
    rs_T = tf.resize([50,50])
    iarT = np.asarray(rs_T)
    iarT_thr = threshold(iarT)
    plt.imshow(iarT_thr,cmap='gray')
    plt.show()
    X_ft_a = iarT_thr.flatten()
    return X_ft_a

#Image.open('images.png','r')
A_val.append(testDFCreator(filename='image9.jpeg'))
A_val.append(testDFCreator(filename='images.jpeg'))
A_val.append(testDFCreator(filename='images-2.jpeg'))
A_val.append(testDFCreator(filename='B3.jpeg'))
A_val.append(testDFCreator(filename='B6.jpeg'))
A_val.append(testDFCreator(filename='B5.jpeg'))
A_val.append(testDFCreator(filename='C5.jpeg'))
A_val.append(testDFCreator(filename='C3.jpeg'))
A_val.append(testDFCreator(filename='C1.jpeg'))
A_val=np.asarray(A_val)


# Reshaping the model to Input into CNN
A_val=A_val/255
print(A_val,"\n\n")
A_val=A_val.reshape(-1,50,50,1)
print (A_val)

predictions = model2.predict_classes(A_val, verbose=1)
print (predictions)


