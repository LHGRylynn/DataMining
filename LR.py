import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
plt.rcParams['axes.unicode_minus']=False

data1=pd.read_csv(r"data1.csv",encoding='GBK')
data2=pd.read_csv(r"data2.csv",encoding='GBK')
data3=pd.read_csv(r"data3.csv",encoding='GBK')


def LR(original,ts):
    #对性别属性列进行独热编码
    sex_dummies = pd.get_dummies(original['sex'], prefix='sex')

    # 将原数据中经过独热编码的列删除
    original.drop(['sex'], axis=1,inplace=True)
    original = pd.concat([original, sex_dummies], axis=1)
    # print(original)

    # 利用热图查看相关度，筛选重要特征
    plt.figure(figsize=(15, 15))
    corr = original.corr()
    # print(corr)
    sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
    plt.show()


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

    # LR模型对训练集训练，并应用于验证集
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train,y_train)
    lr_y_vali_predict=lr.predict(X_vali)

    class_names = ['no','yes']
    # 使用LR模型自带的评分函数score获得模型在验证集上的准确性结果。
    print ('Vali Accuracy of LR Classifier:', lr.score(X_vali, y_vali))
    # 利用classification_report模块获得LR其他三个指标的结果。
    print (classification_report(y_vali, lr_y_vali_predict, target_names=class_names))


    # 查看验证集混淆矩阵
    # 创建混淆矩阵
    matrix = confusion_matrix(y_vali, lr_y_vali_predict)
    # 创建数据集
    dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
    # 绘制热力图
    plt.figure()
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt="d")
    plt.title("Confusion Matrix(Vali)"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()


    lr_y_vali_predict_probability = lr.predict_proba(X_vali)

    #调整sigmoid函数阈值以提高召回率
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    j=1
    for i in thresholds:
        y_vali_predict_recall=lr_y_vali_predict_probability[:,1] > i
        plt.subplot(3,3,j)
        matrix = confusion_matrix(y_vali, y_vali_predict_recall)
        
        # print('Recall metric in the testing dataset while threshold=%s:'%i,matrix[1,1]/(matrix[1,0]+matrix[1,1]))
        recall=matrix[1,1]/(matrix[1,0]+matrix[1,1])
        # create pandas dataframe   创建数据集
        dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

        # create heatmap 绘制热力图
        sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt="d")
        plt.title("Confusion Matrix(Threshold>=%s,recall=%s)"%(i,recall))
        plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        j+=1
    plt.show()



    # 在0.7-0.8的范围内进一步调参
    j=1
    for i in np.arange(0.71,0.79,0.01):
        y_vali_predict_recall=lr_y_vali_predict_probability[:,1] > i
        plt.subplot(3,3,j)
        matrix = confusion_matrix(y_vali, y_vali_predict_recall)
        
        # print('Recall metric in the testing dataset while threshold=%s:'%i,matrix[1,1]/(matrix[1,0]+matrix[1,1]))
        recall=matrix[1,1]/(matrix[1,0]+matrix[1,1])
        # create pandas dataframe   创建数据集
        dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)

        # create heatmap 绘制热力图
        sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt="d")
        plt.title("Confusion Matrix(Threshold>=%s,recall=%s)"%(i,recall))
        plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        j+=1
    plt.show()

    
    # 将训练好的模型应用于测试集
    # lr_y_test_predict=lr.predict(X_test)
    lr_y_test_predict_probability = lr.predict_proba(X_test)
    lr_y_test_predict=lr_y_test_predict_probability[:,1] > ts   # 最终选择阈值0.72
    # lr_y_test_predict=lr_y_test_predict_probability[:,1] > 0.75   # 最终选择阈值0.72

    # 使用LR模型自带的评分函数score获得模型在测试集上的准确性结果。
    print ('Test Accuracy of LR Classifier:', lr.score(X_test, y_test))
    # 利用classification_report模块获得LR其他三个指标的结果。
    print (classification_report(y_test, lr_y_test_predict, target_names=class_names))

    # 查看测试集混淆矩阵
    matrix = confusion_matrix(y_test, lr_y_test_predict)
    dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
    plt.figure()
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues",fmt="d")
    plt.title("Confusion Matrix(Test)"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()

    X_test_return['预测标签']=lr_y_test_predict
    drop=['sex_女','sex_男','birthDay','age','examineDate','参数记录','透前体重','透析前收缩压','开始时间','体重变化','收缩压数值','超滤率','体温','电导度','透析液温度']
    return X_test_return.drop(drop,axis=1)

def LR_data(data,ts):
    drop=['sex','birthDay','age','examineDate','参数记录','透前体重','透析前收缩压','开始时间','体重变化','收缩压数值','超滤率','体温','电导度','透析液温度']
    data_res=pd.merge(LR(data.copy(True),ts),data.drop(drop,axis=1),on=['病人id','记录时间'])
    data_res.loc[data_res['预测标签']==False,['预测标签']]=0
    data_res.loc[data_res['预测标签']==True,['预测标签']]=1
    print(data_res)
    return data_res

LR_data(data1,0.75).to_csv('data1_LR.csv',index=False,encoding='ANSI')
LR_data(data2,0.5).to_csv('data2_LR.csv',index=False,encoding='ANSI')
LR_data(data3,0.5).to_csv('data3_LR.csv',index=False,encoding='ANSI')
