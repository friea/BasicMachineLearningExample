# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 05:53:27 2020

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from warnings import filterwarnings
xor=pd.read_excel("xor.xlsx")
df=xor.copy()
y = df["y"]
x = df.drop(["y"],axis=1)
knn_model=KNeighborsRegressor(n_neighbors=2).fit(x,y)
knn_params={'n_neighbors': np.arange(1,4,1)}
knn=KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn,knn_params,cv=3)
knn_cv_model.fit(x,y)
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(x,y)
drmse0=np.sqrt(-1*cross_val_score(knn_model,x,y,cv=2,scoring="neg_mean_squared_error")).mean()
drmse1=np.sqrt(-1*cross_val_score(knn_tuned,x,y,cv=2,scoring="neg_mean_squared_error")).mean()
rmse0=np.sqrt(mean_squared_error(y,knn_model.predict(x)))
rmse1=np.sqrt(mean_squared_error(y,knn_tuned.predict(x)))
print("test hatası:"+str(("%.3f\n")%rmse0)+" doğrulanmış test hatası:"+str(("%.3f\n")%drmse0))
print("model tunning sonrası test hatası:"+str(("%.3f\n")%rmse1)+" model tunning sonrası doğrulanmış test hatası:"+str(("%.3f")%drmse1))
filterwarnings('ignore')
r2=r2_score(y,knn_tuned.predict(x))
x1=input("ikilik tabanda birinci sayı")
x2=input("ikilik tabanda ikinci sayı")
yeni_veri=[[x1,x2]]
print("tahmin edilen y değeri: "+ str(("%d")%knn_tuned.predict(yeni_veri)))