from headers import *

def SVM(Xtrain,Xverify,Ytrain,Yverify,cw):
    print("hh")
    rbf = SVC(kernel='rbf',class_weight = cw,tol=1e-03,cache_size = 200)
    rbf.fit(Xtrain,Ytrain)
    y_val_pred = rbf.predict(Xverify)
    precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
    return precision[1],recall[1],fscore[1]
    

def rbf(Xtrain,Xverify,Ytrain,Yverify):
    cw={}
    cw[0]=1
    cw[1]=1
    iterations=0
    Wlist=[]
    precisionlist=[]
    recalllist=[]
    fscorelist=[]
    while iterations <1:
        cw[1]=iterations
        Wlist.append(iterations)
        precision,recall,fscore=SVM(Xtrain,Xverify,Ytrain,Yverify,cw)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.legend()
    plt.show()