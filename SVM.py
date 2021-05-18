import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
plt.rcParams['axes.unicode_minus']=False

data1=pd.read_csv(r"data1.csv",encoding='GBK')
data2=pd.read_csv(r"data2.csv",encoding='GBK')
data3=pd.read_csv(r"data3.csv",encoding='GBK')

def CM(y,y_predict,title):
    class_names = ['no','yes']
    # create confusion matrix 创建混淆矩阵
    matrix = confusion_matrix(y, y_predict)

    # create pandas dataframe   创建数据集
    dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

    # create heatmap 绘制热力图
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt="d")
    plt.title("Confusion Matrix(%s)"%title)
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()

def SVM(original):
    #对性别属性列进行独热编码
    sex_dummies = pd.get_dummies(original['sex'], prefix='sex')

    # 将原数据中经过独热编码的列删除
    original.drop(['sex'], axis=1,inplace=True)
    original = pd.concat([original, sex_dummies], axis=1)
    # print(original)

    # # 利用热图查看相关度，筛选重要特征
    # plt.figure(figsize=(15, 15))
    # corr = original.corr()
    # # print(corr)
    # sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
    # plt.show()

    # 划分数据集
    drop=['病人id','birthDay','examineDate','参数记录','开始时间','记录时间']
    X=original.drop(['真实标签'],axis=1)
    y=original['真实标签']
    X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    X_train, X_vali, y_train, y_vali = train_test_split(X_train_vali, y_train_vali, test_size = 0.25, random_state = 0)
    X_train=X_train.drop(drop,axis=1)
    X_vali=X_vali.drop(drop,axis=1)
    X_test_return=X_test.copy(True)
    X_test=X_test.drop(drop,axis=1)

    # 标准化
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_vali = ss.transform(X_vali)
    X_test = ss.transform(X_test)

    # 建立模型
    pca = PCA(n_components=11, whiten=True, random_state=0)
    svc = SVC(kernel='linear', class_weight='balanced')
    model = make_pipeline(pca, svc)

    # 网格搜索，选出最优参数C
    param_grid = {'svc__C':[1,2,5,10,12,15]}
    grid = GridSearchCV(model,param_grid,n_jobs=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    model=grid.best_estimator_

    svm_y_vali_predict=model.predict(X_vali)
    print(classification_report(y_vali, svm_y_vali_predict))
    CM(y_vali,svm_y_vali_predict,'Vali')
    
    svm_y_test_predict=model.predict(X_test)
    print(classification_report(y_test, svm_y_test_predict))
    CM(y_test,svm_y_test_predict,'Test')

    X_test_return['预测标签']=svm_y_test_predict
    drop=['sex_女','sex_男','birthDay','age','examineDate','参数记录','透前体重','透析前收缩压','开始时间','体重变化','收缩压数值','超滤率','体温','电导度','透析液温度']
    return X_test_return.drop(drop,axis=1)

def SVM_data(data):
    drop=['sex','birthDay','age','examineDate','参数记录','透前体重','透析前收缩压','开始时间','体重变化','收缩压数值','超滤率','体温','电导度','透析液温度']
    data_res=pd.merge(SVM(data.copy(True)),data.drop(drop,axis=1),on=['病人id','记录时间'])
    data_res.loc[data_res['预测标签']==False,['预测标签']]=0
    data_res.loc[data_res['预测标签']==True,['预测标签']]=1
    print(data_res)
    return data_res

SVM_data(data1).to_csv('data1_SVM.csv',index=False,encoding='ANSI')
SVM_data(data2).to_csv('data2_SVM.csv',index=False,encoding='ANSI')
SVM_data(data3).to_csv('data3_SVM.csv',index=False,encoding='ANSI')
