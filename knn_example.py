# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:19:53 2020

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from warnings import filterwarnings
import matplotlib.pyplot as plt
xor=pd.read_csv("original.csv")
df=xor.copy()
df=df.drop(df.columns[0],axis=1)
y = df["sales"]
x = df.drop(["sales"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=45)
knn_model=KNeighborsRegressor().fit(x_train,y_train)
knn_params={'n_neighbors': np.arange(1,135,1)}
knn=KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn,knn_params,cv=10)
knn_cv_model.fit(x_train,y_train)
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(x_train,y_train)
drmse0=np.sqrt(-1*cross_val_score(knn_model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()
drmse1=np.sqrt(-1*cross_val_score(knn_tuned,x_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()
rmse0=np.sqrt(mean_squared_error(y_test,knn_model.predict(x_test)))
rmse1=np.sqrt(mean_squared_error(y_test,knn_tuned.predict(x_test)))
print("test hatası:"+str(("%.3f\n")%rmse0)+" doğrulanmış test hatası:"+str(("%.3f\n")%drmse0))
print("model tunning sonrası test hatası:"+str(("%.3f\n")%rmse1)+" model tunning sonrası doğrulanmış test hatası:"+str(("%.3f")%drmse1))
tv=input("TV reklam sayısı: ")
radyo=input("radyo reklam sayısı: ")
gazete=input("gazete reklam sayısı: ")
veri=[[tv,radyo,gazete]]
filterwarnings('ignore')
print("\n tahmini satış: "+str(("%d")%knn_tuned.predict(veri)))
r2=r2_score(y_test,knn_tuned.predict(x_test))







