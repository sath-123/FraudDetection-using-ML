from headers import *

def LinearSVM(Xtrain,Xverify,Ytrain,Yverify,cw):
    ls = LinearSVC(class_weight = cw, dual = False ,tol=1e-05,max_iter = 1000)
    ls.fit(Xtrain,Ytrain)
    y_val_pred = ls.predict(Xverify)
    precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
    return precision[1],recall[1],fscore[1]


def ltransfer(Xtrain,Xval,ytrain,yval):
    cw={}
    cw[0]=1
    cw[1]=1
    iterations=1
    Wlist=[]
    precisionlist=[]
    recalllist=[]
    fscorelist=[]
    while iterations <500:
        cw[1]=iterations
        Wlist.append(iterations)
        precision,recall,fscore=LinearSVM(Xtrain,Xval,ytrain,yval,cw)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.legend()
    plt.show()


def lcashout(Xtrain,Xval,ytrain,yval):
    cw={}
    cw[0]=1
    cw[1]=1
    iterations=1
    Wlist=[]
    precisionlist=[]
    recalllist=[]
    fscorelist=[]
    while iterations <500:
        cw[1]=iterations
        Wlist.append(iterations)
        precision,recall,fscore=LinearSVM(Xtrain,Xval,ytrain,yval,cw)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.legend()
    plt.show()
