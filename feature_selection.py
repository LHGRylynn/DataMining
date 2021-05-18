from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

wine = load_wine()

# Standardization
X=StandardScaler().fit(wine.data).transform(wine.data)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,wine.target,test_size=0.3,random_state=0)

# Discretization
df=pd.DataFrame(wine.data,columns=wine.feature_names)
for col_name in wine.feature_names:
    df[col_name]=pd.qcut(df[col_name],6,labels=False)
# print(df.values)

# Original Data
print("Original Data:")
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print("acc:%f"%score)
print()

# Variance Filtering
print("Variance Filtering:")
Xtrain,Xtest,Ytrain,Ytest=train_test_split(VarianceThreshold(threshold=2).fit_transform(df.values),wine.target,test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print("acc:%f"%score)
print()

# Chi test
print("Chi test:")
Xtrain,Xtest,Ytrain,Ytest=train_test_split(SelectKBest(chi2, k=4).fit_transform(df.values, wine.target),wine.target,test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print("acc:%f"%score)
print()

# Wrapper:Recursive feature elimination
print("Wrapper:Recursive feature elimination:")
Xtrain,Xtest,Ytrain,Ytest=train_test_split(RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=4).fit_transform(X, wine.target),wine.target,test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print("acc:%f"%score)
print()
