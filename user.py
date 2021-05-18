import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

live=pd.read_csv(r"q2/user/用户常看直播频道记录样例.csv")
live["STBID"]=live["STBID"].apply(str)
live.drop(["OPK","L_CHANNEL_NAME","PT_TIME","RN"],axis=1,inplace=True)
live=live[live["CNT"]>200]

collection=pd.read_csv(r"q2/user/收藏记录.csv")
collection["STBID"]=collection["STBID"].apply(str)
collection.drop(["ID","CODE","ITEMCODE","MD5","NAME","PORTAL_VER","STATUS","TIME","TYPE","URL","FOLDERCODE"],axis=1,inplace=True)
user=pd.merge(collection,live,on=['STBID'])  # merge
original_data=user.copy(deep=True)  # original data
user['CNT']=pd.qcut(user['CNT'],5,labels=False)  # qcut_CNT

# one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(user["SID"])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# combine original data with one-hot encoded data
code=pd.DataFrame(onehot_encoded)
data=user.join(code)
data.drop(["SID","STBID"],axis=1,inplace=True)# data to train

# PCA
pca = PCA(n_components=2)
X = pca.fit(data).transform(data)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("PCA")
plt.show()

# SSE
sse={}
for k in range(10,30):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title("SSE-K")
plt.show()

# result of Kmeans
kmeans = KMeans(n_clusters=20, max_iter=1000).fit(X)
res=pd.DataFrame(X,columns=["Dimension1","Dimension2"])
res["label"]=kmeans.labels_

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(0,20):
    ax.scatter(res[(res["label"]==i)]["Dimension1"],res[(res["label"]==i)]["Dimension2"],label=i)

colormap = plt.cm.gist_ncar 
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]       

for t,j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.title("Result")
plt.show()

# validity index
cluster_score_si = metrics.silhouette_score(res, res["label"])
print("cluster_score_si:", cluster_score_si)

cluster_score_ch = metrics.calinski_harabasz_score(res, res["label"])
print("cluster_score_ch:", cluster_score_ch)

cluster_score_DBI = metrics.davies_bouldin_score(res, res["label"])
print("cluster_score_DBI:", cluster_score_DBI)

# # original data to csv
# original_data["label"]=kmeans.labels_
# original_data.to_csv(r"q2\clusters.csv",index=False)
