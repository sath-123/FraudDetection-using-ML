from headers import *
from pca import *
from lr import *
from rbf import *
from Linearsvm import *
from dt import *
from AUPRC import *
from table import *

# code for extracting the data from csv file and sepersting transfer and cashout data
# reading csv files
data =  pd.read_csv('transactions_train.csv')
# print(data)
# print(data.iloc[0].values)
data1 = data[data['type'] == 'TRANSFER']
data2= data[data['type'] == 'CASH_OUT']
data3=data[data['type'] == 'CASH_IN']
data4=data[data['type'] == 'DEBIT']
data5=data[data['type'] == 'PAYMENT']
X1=data1[['amount','oldbalanceOrig','newbalanceOrig','oldbalanceDest','newbalanceDest']].values
y1=data1.iloc[:,9].values
X2=data2[['amount','oldbalanceOrig','newbalanceOrig','oldbalanceDest','newbalanceDest']].values
y2=data2.iloc[:,9].values
X3=data3[['amount','oldbalanceOrig','newbalanceOrig','oldbalanceDest','newbalanceDest']].values
y3=data3.iloc[:,9].values
X4=data4[['amount','oldbalanceOrig','newbalanceOrig','oldbalanceDest','newbalanceDest']].values
y4=data4.iloc[:,9].values
X5=data5[['amount','oldbalanceOrig','newbalanceOrig','oldbalanceDest','newbalanceDest']].values
y5=data5.iloc[:,9].values
pca(X1,y1)
pca(X2,y2)
pca(X3,y3)
pca(X4,y4)
pca(X5,y5)




#code for spliting the data into 70%-Training data,15%-Validation data,15%-Testing data

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,stratify=y1 , test_size=0.30, random_state=42)
X1_val, X1_test, y1_val, y1_test = train_test_split(X1_test, y1_test,stratify=y1_test , test_size=0.50, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,stratify=y2 , test_size=0.30, random_state=42)
X2_val, X2_test, y2_val, y2_test = train_test_split(X2_test, y2_test,stratify=y2_test , test_size=0.50, random_state=42)
print(y2_val.shape)

LRtrasfer(X1_train,X1_val,y1_train,y1_val)
LRcashout(X2_train,X2_val,y2_train,y2_val)


ltransfer(X1_train,X1_val,y1_train,y1_val)
lcashout(X2_train,X2_val,y2_train,y2_val)

rbf(X1_train,X1_val,y1_train,y1_val)
rbf(X2_train,X2_val,y2_train,y2_val)

tree(X1_train,X1_val,y1_train,y1_val)
tree(X2_train,X2_val,y2_train,y2_val)






