import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
veri = pd.read_csv("original.csv")
veri = veri.drop(veri.columns[[0]],axis=1)
X=veri.drop(veri.columns[[-1]],axis=1)
Y=veri["sales"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=60)
lm = LinearRegression()
model=lm.fit(X_train,Y_train)
rmse=np.sqrt(mean_squared_error(Y_train,model.predict(X_train)))
rmset=np.sqrt(mean_squared_error(Y_test,model.predict(X_test)))
Y_pred=model.predict(X_train)
r2= r2_score(Y_train,Y_pred)
print("eğitim hatası= "+str(("%.2f\n")%rmse)+"test hatası= "+str(("%.2f\n")%rmset))
dr2=cross_val_score(model,X,Y,cv=10,scoring="r2").mean()
deh=np.sqrt((-1)*(cross_val_score(model,X_train,Y_train,cv=10,scoring="neg_mean_squared_error").mean()))
dth=np.sqrt((-1)*(cross_val_score(model,X_test,Y_test,cv=10,scoring="neg_mean_squared_error").mean()))
print("doğrulanmış eğitim hatası= "+str(("%.2f\n")%deh)+"doğrulanmış test hatası= "+str(("%.2f\n")%dth)+"doğrulanmış r2 skoru= "+str(("%.2f\n")%dr2))
yeni_veri[0]=input("TV reklam sayısı girin:\n")
yeni_veri[1]=input("Radyo reklam sayısı girin:\n")
yeni_veri[2]=input("Gazete reklam sayısı girin:\n")
print("üründeki tahmin edilen satış = "+ str(("%d")%model.predict(yeni_veri)))