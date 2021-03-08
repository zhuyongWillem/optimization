# Description：
# Author：朱勇
# Email: yong_zzhu@163.com
# Time：2021/3/8 20:41

import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#加载数据及可视化
data = pd.read_csv('task2_data.csv')
x = data.drop(['y'],axis=1)
y = data.loc[:,'y']
fig1 = plt.figure()
plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0],marker='x',s=150,label='bad')
plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1],marker='o',facecolor='none',edgecolors='red',s=150,label='good')
plt.title('chip data')
plt.xlabel('size 1')
plt.ylabel('size 2')
plt.legend()
plt.show()
#寻找异常点
#计算均值方差
x1 = data.loc[:,'x1'][y==0]
x2 = data.loc[:,'x2'][y==0]
u1 = x1.mean()
sigma1 = x1.std()
u2 = x2.mean()
sigma2 = x2.std()
#计算高斯分布
p1 = 1/sigma1/math.sqrt(2*math.pi)*np.exp(-np.power((x1-u1),2)/2/math.pow(sigma1,2))
p2 = 1/sigma2/math.sqrt(2*math.pi)*np.exp(-np.power((x2-u2),2)/2/math.pow(sigma2,2))
p = np.multiply(p1,p2)
print('max p',max(p))
print('min p',min(p))
print('max/min',max(p)/min(p))
ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(x[y==0])
y_predict_bad = ad_model.predict(x[y==0])
#异常数据点
fig2 = plt.figure()
plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0],marker='x',s=150,label='bad')
plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1],marker='o',facecolor='none',edgecolors='red',s=150,label='good')
plt.scatter(x.loc[:,'x1'][y==0][y_predict_bad==-1],x.loc[:,'x2'][y==0][y_predict_bad==-1],marker='o',s=150)
plt.title('chip data')
plt.xlabel('size 1')
plt.ylabel('size 2')
plt.legend()
plt.show()
#剔除异常点，更新数据
data.drop(index=35)
x = data.drop(['y'],axis=1)
y = data.loc[:,'y']
#主成分分析
x_norm = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
x_reduced = pca.fit_transform(x_norm)
var_ratio = pca.explained_variance_ratio_
print(var_ratio)
fig3 = plt.figure()
plt.bar([1,2],var_ratio)
plt.xticks([1,2],['pc1','pc2'])
plt.show()
#数据分离
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.4)
#建立模型
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(x_train,y_train)
#模型预测
y_train_predict = knn_3.predict(x_train)
y_test_predict = knn_3.predict(x_test)
print(accuracy_score(y_train_predict,y_train),accuracy_score(y_test_predict,y_test))
#结果可视化
xx,yy = np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range = np.c_[xx.ravel(),yy.ravel()]
y_range_predict = knn_3.predict(x_range)
fig4 = plt.figure()
plt.scatter(x_range[:,0][y_range_predict==0],x_range[:,1][y_range_predict==0],label='bad_p')
plt.scatter(x_range[:,0][y_range_predict==1],x_range[:,1][y_range_predict==1],label='good_p')
plt.scatter(x.loc[:,'x1'][y==0],x.loc[:,'x2'][y==0],marker='x',s=150,label='bad')
plt.scatter(x.loc[:,'x1'][y==1],x.loc[:,'x2'][y==1],marker='o',facecolor='none',edgecolors='red',s=150,label='good')
plt.title('chip data')
plt.xlabel('size 1')
plt.ylabel('size 2')
plt.legend()
plt.show()
