from headers import *
scaler = preprocessing.StandardScaler()
def pca(X,y):
    X = scaler.fit_transform(X)
    principal=PCA(n_components=2)
    principal=principal.fit(X)
    X_new=principal.transform(X)

    col=[]
    for j in y:
        if(j==0):
            col.append('#1f77b4')
        elif j==1:
            col.append('#ff7f0e')
    plt.scatter(X_new[:,0],X_new[:,1].real,c=col)
    plt.show()