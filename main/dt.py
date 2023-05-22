from headers import *

def random_forest_classifier(Xtrain,Xverify,Ytrain,Yverify,w):
    RFC=RandomForestClassifier(n_estimators=30, class_weight=w, max_depth=3)
    RFC.fit(Xtrain,Ytrain)
    y_val_pred = RFC.predict(Xverify)
    precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
    return precision[1],recall[1],fscore[1]
    

def tree(Xtrain,Xverify,Ytrain,Yverify):
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
        precision,recall,fscore=random_forest_classifier(X1_train,X1_val,y1_train,y1_val,cw)
        precisionlist.append(precision)
        recalllist.append(recall)
        fscorelist.append(fscore)
        iterations=1+iterations
        
    plt.plot(Wlist,precisionlist,label = "precision",linestyle="--")
    plt.plot(Wlist,recalllist,label = "recall",linestyle="--")
    plt.plot(Wlist,fscorelist,label = "fscore",linestyle="--")
    plt.xlabel("class weights")
    plt.title("Random forest -recall,presicion,fscore on transfer data")
    plt.legend()
    plt.show()
