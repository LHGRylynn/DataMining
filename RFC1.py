import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import sklearn.ensemble as ensemble


plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
plt.rcParams['axes.unicode_minus']=False

data1=pd.read_csv(r"data1.csv",encoding='GBK')
data2=pd.read_csv(r"data2.csv",encoding='GBK')
data3=pd.read_csv(r"data3.csv",encoding='GBK')


def RFC(original):
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

    # 建立模型并网格搜索寻优
    param_grid = {
        'criterion':['entropy','gini'],
        'max_depth':[2,3,4,5],    # 深度：这里是森林中每棵决策树的深度
        'n_estimators':[21,23,25],  # 决策树个数-随机森林特有参数
        'max_features':[0.1,0.2,0.3],  # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
        'min_samples_split':[4,8,12,16]  # 叶子的最小拆分样本量
    }

    rfc = ensemble.RandomForestClassifier(class_weight='balanced')
    rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                        scoring='roc_auc', cv=4, n_jobs=5)
    rfc_cv.fit(X_train, y_train)
    print(rfc_cv.best_params_)

    # 使用随机森林对验证集进行预测
    rfc_y_vali_predict = rfc_cv.predict(X_vali)
    print(classification_report(y_vali,rfc_y_vali_predict))

    fpr_vali, tpr_vali, th_vali = roc_curve(y_vali,rfc_y_vali_predict)
    # 构造 roc 曲线
    plt.plot(fpr_vali, tpr_vali, lw=2, alpha=0.3)
    plt.show()
    print('AUC = %.4f' %metrics.auc(fpr_vali, tpr_vali))

    # 使用随机森林对测试集进行预测
    rfc_y_test_predict = rfc_cv.predict(X_test)
    print(classification_report(y_test,rfc_y_test_predict))

    fpr_test, tpr_test, th_test = roc_curve(y_test,rfc_y_test_predict)
    # 构造 roc 曲线
    print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


    X_test_return['预测标签']=rfc_y_test_predict
    X_test_return.loc[X_test_return['预测标签']==False,['预测标签']]=0
    X_test_return.loc[X_test_return['预测标签']==True,['预测标签']]=1
    drop=['sex_女','sex_男','birthDay','age','examineDate','参数记录','透前体重','透析前收缩压','开始时间','体重变化','收缩压数值','超滤率','体温','电导度','透析液温度']
    X_test_return.drop(drop,axis=1,inplace=True)
    print(X_test_return)
    return X_test_return

# RFC(data1).to_csv('data1_RFC.csv',index=False,encoding='ANSI')
RFC(data1)

