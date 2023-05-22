from headers import *

def PRCurve(Xtrain,Xverify,Ytrain,Yverify,cw,W,w):
    ls = LinearSVC(class_weight = cw, dual = False ,tol=1e-05,max_iter = 1000)
    ls.fit(Xtrain,Ytrain)
    y_val_pred = ls.decision_function(Xverify)
#     print(y_val_pred.shape)
    precision, recall, thresholds = precision_recall_curve(Yverify, y_val_pred,pos_label = 1,sample_weight=None)
#     print(thresholds)
    area = auc(recall,precision)
    area = round(area,4)
    print('Logistic Regression - Area under PRC' , area)
    plt.plot(recall, precision, linestyle='--' , color = 'r',label = 'Logistic Regression - AUPRC - '  + str(area))
    
    LR=LogisticRegression(class_weight=W)
    LR.fit(Xtrain,Ytrain)
    y_val_pred = LR.decision_function(Xverify)
    pre_svm , rec_svm, thresh_svm = precision_recall_curve(Yverify, y_val_pred,pos_label = 1,sample_weight=None)
    area = auc(rec_svm, pre_svm)
    area = round(area,4)
    print('Linear SVM - Area under PRC' , area)
    plt.plot(rec_svm, pre_svm, linestyle='--' , color = 'b',label = 'Linear SVM - AUPRC - '  + str(area))
    
    rbf = SVC(kernel='rbf',class_weight = w,tol=1e-03,cache_size = 200)
    rbf.fit(Xtrain,Ytrain)
    y_val_pred = rbf.decision_function(Xverify)
    pre_kernel_svm , rec_kernel_svm, thresh_kernel_svm = precision_recall_curve(Yverify, y_val_pred,pos_label = 1 )
    area = auc(rec_kernel_svm, pre_kernel_svm)
    area = round(area,4)

    print('RBF Precision ' , pre_kernel_svm)
    print('RBF Recall ' , rec_kernel_svm)
   
    print('SVM with RBF kernel - Area under PRC' , area)
    plt.plot(rec_kernel_svm, pre_kernel_svm, linestyle='--' , color = 'g',label = 'SVM RBF kernel - AUPRC - '  + str(area))
    plt.xlabel("Recall")
    plt.ylabel("Precision")