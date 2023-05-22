from headers import *


def logistic(Xtrain,Xverify,Ytrain,Yverify,W):
    LR=LogisticRegression(class_weight=W)
    LR.fit(Xtrain,Ytrain)
    y_val_pred = LR.predict(Xverify)
    precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
    return precision[1],recall[1],fscore[1]


def LRtrasfer(Xtrain,Xval,ytrain,yval):
    W={}
    W[0]=1
    W[1]=0
    iterations=1
    Wlist=[]
    precisionlist=[]
    recalllist=[]
    fscorelist=[]
    while iterations <500:
        W[1]=iterations
        Wlist.append(iterations)
        precision,recall,fscore=logistic(Xtrain,Xval,ytrain,yval,W)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.xlabel("class weights")
    plt.ylabel("Logistic regression-Precision,recall,fscore on CV set")
    plt.legend()
    plt.show()


def LRcashout(Xtrain,Xval,ytrain,yval):
    W={}
    W[0]=1
    W[1]=0
    iterations=1
    Wlist=[]
    precisionlist=[]
    recalllist=[]
    fscorelist=[]
    while iterations <500:
        W[1]=iterations
        Wlist.append(iterations)
        precision,recall,fscore=logistic(Xtrain,Xval,ytrain,yval,W)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.xlabel("class weights")
    plt.ylabel("Logistic regression-Precision,recall,fscore on CV set")
    plt.legend()
    plt.show()