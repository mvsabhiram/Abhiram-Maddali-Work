import pandas as pd 
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from numpy.linalg import inv
from math import log
df=pd.read_csv("iris.csv", names=["seplen","sepwid","petlen","petwid","class"])
def f(row):
    if row['class'] == 'Iris-setosa':
        val = 1
    elif row['class'] == 'Iris-versicolor':
        val = 2
    else:
        val = 3
    return val
df['newcol'] = df.apply(f, axis=1)
df['counter'] = range(1, len(df) + 1)   
def Discrement(Train_overall,Test_overall):
    clas= Train_overall[["newcol"]]
    train=Train_overall[['seplen','sepwid','petlen','petwid']]
    t=df[['seplen','sepwid','petlen','petwid']]
    cl= np.unique(clas)
    for i in cl:
        inx= df.loc[df['newcol']== i]
        fea = inx[['seplen','sepwid','petlen','petwid',"newcol"]]
        mu=df.mean().mean()
        vai=t.cov()
        ivai=inv(vai)
        l=fea["seplen"].count()
        l1=fea["newcol"].count()
        prio= l/l1
        Wi=0.5*ivai
        Wii=ivai*np.transpose(mu)
        wio=-0.5*np.transpose(mu)*Wii- (0.5* abs(ivai))+ log(prio);
        si=Test_overall["seplen"].count() 
        return si
        for j in si :
            g=(np.transpose(Test_overall.loc[j])*Wi)+(np.transpose(Wii)*(Test_overall.loc[j]))+(wio)  
def plit_dat(ds,n) :
    xlist=ds[["newcol"]]
    cla=np.unique(xlist)
    for i in cla:
        inx= ds.loc[ds['newcol']== 1]
        inx2= ds.loc[ds['newcol']== 2]
        inx3= ds.loc[ds['newcol']== 3]
    inx2_e=pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])
    inx2_e= inx2
    inx2_e.index=pd.RangeIndex(len(inx2.index))
    inx2_e.index=range(len(inx2_e.index))
    inx3_e=pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])
    inx3_e= inx3
    inx3_e.index=pd.RangeIndex(len(inx3.index))
    inx3_e.index=range(len(inx3_e.index))  
#empty Datasets for Training and Testing 
    Train_over=pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","newcol"])
    Test_over=pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","newcol"])  
#class 1
    t1,tes1=n_split(inx,n)
    t_1=t1[["seplen","sepwid","petlen","petwid","newcol"]]
    tes1=tes1[["seplen","sepwid","petlen","petwid","newcol"]];
    Train_overall=Train_over.append(t_1)
    Test_overall=Test_over.append(tes1) 
#class 2    
    t2,tes2=n_split(inx2_e,n)
    t_2=t2[["seplen","sepwid","petlen","petwid","newcol"]]
    tes2=tes2[["seplen","sepwid","petlen","petwid","newcol"]]
    Train_overall=Train_over.append(t_2)
    Test_overall=Test_over.append(tes2)
    
#class 3
    t3,tes3=n_split(inx3_e,n)
    t_3=t3[["seplen","sepwid","petlen","petwid","newcol"]]
    tes3=tes3[["seplen","sepwid","petlen","petwid","newcol"]]
    Train_over=Train_overall.append(t_3)
    Test_over=Test_overall.append(tes3)
#NDiscriminant Function 
    pred=Discrement(Train_over,Test_over)
    gnb = GaussianNB()
    Y_train = list(Train_overall.newcol.values)
    Y_test=list(Test_overall.newcol.values)
    gnb.fit(Train_overall[["seplen","sepwid","petlen","petwid"]],Y_train)
    predic=gnb.predict(Test_overall[["seplen","sepwid","petlen","petwid"]])
#Accuracies Predicted 
    a_score=accuracy_score(Y_test,predic)
##Confusion Matrix
    confusmat=confusion_matrix(Y_test,predic)
    return pred
    print Test_overall
def n_split(r,s):
    l=r["seplen"].count()  
    d=np.floor(l/s)
    d=int(d)
    g=0
    j=0
    y_over= pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])
    p_over= pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])   
    test_d= pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])
    train_d= pd.DataFrame(columns=["seplen","sepwid","petlen","petwid","counter","newcol"])
    while(g<s):
        for i in range((g*d),((g*d)+d)):
            test_d.loc[j] = r.loc[i]
            j=j+1;  
        train_d= r[~r.counter.isin(test_d.counter)]
        g=g+1
        j=0
        y_over=y_over.append(train_d)
        p_over=p_over.append(test_d)
    return y_over,p_over
accuracy=plit_dat(df,5)
print "accuracy=",accuracy,"\n\n\nConfusion Matrix=\n",confusin