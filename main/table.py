from header import *

def AUPRC(Xtrain,Xverify,Ytrain,Yverify,cw,algorithm):
    if algorithm=="linear":
        ls = LinearSVC(class_weight = cw, dual = False ,tol=1e-05,max_iter = 1000)
        ls.fit(Xtrain,Ytrain)
        y_val_pred = ls.predict(Xverify)
        precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
        y_val_pred = ls.decision_function(Xverify)
    #     print(y_val_pred.shape)
        precision, recall, thresholds = precision_recall_curve(Yverify, y_val_pred,pos_label = 1,sample_weight=None)
    #     print(thresholds)
        area = auc(recall,precision)
        area = round(area,4)
        return precision[1],recall[1],fscore[1],area
    elif algorithm=="LR":
        LR=LogisticRegression(class_weight=cw)
        LR.fit(Xtrain,Ytrain)
        y_val_pred = LR.predict(Xverify)
        precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
        y_val_pred = LR.decision_function(Xverify)
        pre_svm , rec_svm, thresh_svm = precision_recall_curve(Yverify, y_val_pred,pos_label = 1,sample_weight=None)
        area = auc(rec_svm, pre_svm)
        area = round(area,4)
        return precision[1],recall[1],fscore[1],area
    elif algorithm=="rbf":
        rbf = SVC(kernel='rbf',class_weight = cw,tol=1e-03,cache_size = 200)
        rbf.fit(Xtrain,Ytrain)
        y_val_pred = rbf.predict(Xverify)
        precision,recall,fscore,support=score(Yverify,y_val_pred,average=None)
        y_val_pred = rbf.decision_function(Xverify)
        pre_kernel_svm , rec_kernel_svm, thresh_kernel_svm = precision_recall_curve(Yverify, y_val_pred,pos_label = 1 )
        area = auc(rec_kernel_svm, pre_kernel_svm)
        area = round(area,4)
        return precision[1],recall[1],fscore[1],area
        

# for tabulating precision ,recall ,fscore
def table(data,algorithm,Xtrain,Xverify,Ytrain,Yverify,w):
    
    dummy=[]
    precision,recall,fscore,area=AUPRC(Xtrain,Xverify,Ytrain,Yverify,w,algorithm)
    dummy.append(algorithm)
    dummy.append(precision)
    dummy.append(recall)
    dummy.append(fscore)
    dummy.append(area)
    data.append(dummy)
    print("kk")
    
data=[]
c1={}
c1[1]=70
c1[0]=1
c2={}
c2[1]=39
c2[0]=1
c3={}
c3[1]=16
c3[0]=1

head = ["Algorithm", "precision","recall","fscore","AUPRC"]
table(data,"LR",X1_train,X1_val,y1_train,y1_val,c1)
table(data,"linear",X1_train,X1_val,y1_train,y1_val,c2)
table(data,"rbf",X1_train,X1_val,y1_train,y1_val,c3)
print(tabulate(data, headers=head, tablefmt="grid"))
