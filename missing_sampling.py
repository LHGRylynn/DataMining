import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

info = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
heart = pd.read_csv("cleveland.data.csv", header=None, names=info)
heart.target = heart.target.where(heart.target == 0, 1)
# print(heart)

# plt.figure(figsize=(12, 10))
# corr = heart.corr()
# sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
# plt.show()

cp_dummies = pd.get_dummies(heart['cp'], prefix='cp')
restecg_dummies = pd.get_dummies(heart['restecg'], prefix='restecg')
slope_dummies = pd.get_dummies(heart['slope'], prefix='slope')
thal_dummies = pd.get_dummies(heart['thal'], prefix='thal')

# 将原数据中经过独热编码的列删除
heart_new = heart.drop(['cp', 'restecg', 'slope', 'thal'], axis=1)
heart_new = pd.concat(
    [heart_new, cp_dummies, restecg_dummies, slope_dummies, thal_dummies], axis=1)

label = heart_new['target']
data = heart_new.drop('target', axis=1)

standardScaler = StandardScaler()
standardScaler.fit(data)
data = standardScaler.transform(data)
'''
缺失值处理
缺失值占比小于2%（6/303），此处直接删除
'''

train_X, test_X, train_y, test_y = train_test_split(
    data, label, random_state=3)

knn = KNeighborsClassifier()

knn.fit(train_X, train_y)

# knn_pred_y = knn.predict(test_X)
# print(knn_pred_y)

acc = knn.score(test_X, test_y)

print("acc:%f" % acc)

'''
分层抽样

'''
X_sample, X_drop_sample, y_sample, y_drop_sample = train_test_split(
    data, label, test_size=0.2, stratify=label, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=3)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

# knn_pred_y = knn.predict(X_test)

acc_new = knn.score(X_test, y_test)

acc_drop_sample=knn.score(X_drop_sample,y_drop_sample)

print("acc_new:%f" % acc_new)
print("acc_drop_sample:%f" % acc_drop_sample)
