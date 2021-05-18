from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris=datasets.load_iris()
A=iris.data
H=iris.feature_names

df = pd.DataFrame(A,columns=H)
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')
plt.show()

print("Pearson correlation coefficient:")
print(df.corr("pearson"))
print()

print("Spearman correlation coefficient:")
print(df.corr("spearman"))
print()

print("Kendall correlation coefficient:")
print(df.corr("kendall"))
print()
