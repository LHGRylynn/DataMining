import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import seaborn as sns
import collections

diabetes=pd.read_csv(r"q1/dataset_diabetes/diabetic_data.csv")

# missing values processing
diabetes.replace("?",np.NaN,inplace=True)
cleaned_diabetes = diabetes.dropna(axis=1,thresh=len(diabetes)*0.7)
cleaned_diabetes = cleaned_diabetes.dropna(axis=0,thresh=cleaned_diabetes.shape[1])


# print(cleaned_diabetes['time_in_hospital'].values)
# db_df=pd.DataFrame({'time_in_hospital':cleaned_diabetes['time_in_hospital'].values,
#                  'num_lab_procedures':cleaned_diabetes['num_lab_procedures'].values,
#                  'num_procedures':cleaned_diabetes['num_procedures'].values,
#                  'num_medications':cleaned_diabetes['num_medications'].values,
#                  'number_outpatient':cleaned_diabetes['number_outpatient'].values,
#                  'number_emergency':cleaned_diabetes['number_emergency'].values,
#                  'number_inpatient':cleaned_diabetes['number_inpatient'].values,
#                  'number_diagnoses':cleaned_diabetes['number_diagnoses'].values
# })
# print(db_df)


'''
按列统计相异的值

for col in cleaned_diabetes.columns[19:]:
    print("%s:"%col)
    dic = collections.Counter(cleaned_diabetes[col])
    for key in dic:
        print(key,dic[key])
    print()

'''

#根据统计找出有重复值大于93%的列进行删除，并将ID两列和diag_3一并删除
cleaned_diabetes.drop(cleaned_diabetes.columns[[0,1,15,16,17,19,22,23,24,25,26,29,30,31,32,33,34,35,36,37,39,40,41,42,43]], axis=1, inplace=True)
#print(cleaned_diabetes.shape)


#对中文属性列进行整数编码
dummies=['race','gender','age','A1Cresult','metformin','glipizide','glyburide','insulin','change','diabetesMed','readmitted']

for dummy in dummies:
    label_encoder = LabelEncoder()
    cleaned_diabetes[dummy] = label_encoder.fit_transform(cleaned_diabetes[dummy])

'''
热图查看相关系数
plt.figure(figsize=(12, 12))
corr = cleaned_diabetes.corr()
sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
plt.show()
'''

# 'time_in_hospital'与'num_medications'相关度较高，删去'time_in_hospital'
cleaned_diabetes.drop(['time_in_hospital'],axis=1,inplace=True)
data=cleaned_diabetes.values

#K-means
# sse={}
# for k in range(2,10):
#     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
#     sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Number of cluster")
# plt.ylabel("SSE")
# plt.title("SSE-K")
# plt.show()

# K-means_result.to_csv
kmeans = KMeans(n_clusters=3, max_iter=1000).fit(data)
cleaned_diabetes["kmeans_label"]=kmeans.labels_


'''
# DBscan
res = []
for eps in range (5,16,1):
    dbscan = DBSCAN(eps=eps, min_samples=21).fit(data)
    n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
    outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
    stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
    res.append({'eps':eps,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})
df = pd.DataFrame(res)
print(df.loc[df.n_clusters == 3, :])
'''
# DBscan_result.to_csv
dbscan = DBSCAN(eps=6, min_samples=21).fit(data)
cleaned_diabetes["dbscan_label"]=dbscan.labels_
cleaned_diabetes.to_csv('clusters.csv',index = False)

'''
# Hierarchical
from sklearn.cluster import AgglomerativeClustering #导入sklearn的层次聚类函数
model = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
model.fit(data) #训练模型
#详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别

from scipy.cluster.hierarchy import linkage,dendrogram
#这里使用scipy的层次聚类函数
Z = linkage(data, method = 'ward', metric = 'euclidean') #谱系聚类图
P = dendrogram(Z, 0) #画谱系聚类图
plt.show()
'''

result=pd.read_csv(r"q1/clusters.csv")

print("kmeans result:")
print(result["kmeans_label"].value_counts())
print()
print("dbscan result:")
print(result["dbscan_label"].value_counts())

# validity index
cluster_score_si = metrics.silhouette_score(data, result["kmeans_label"])
print("cluster_score_si:", cluster_score_si)

cluster_score_si = metrics.silhouette_score(data, result["dbscan_label"])
print("cluster_score_si:", cluster_score_si)