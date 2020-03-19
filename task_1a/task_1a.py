#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
data = pd.read_csv("train.csv")


# In[29]:


feature_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']
X = data[feature_cols]
y = data['y']
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

regu_params = [0.01,0.1,1,10,100]
result = []
for param in regu_params:
    tot_RMSE = 0
    for train,test in kf.split(data):
        X_train = data.iloc[train][feature_cols]
        y_train = data.iloc[train]['y']
        X_test = data.iloc[test][feature_cols]
        y_test = data.iloc[test]['y']
        ridge_reg = linear_model.Ridge(param)
        ridge_reg.fit(X_train, y_train)
        y_pred = ridge_reg.predict(X_test)
        RMSE = mean_squared_error(y_test, y_pred)**0.5
        tot_RMSE = tot_RMSE + RMSE
    result.append(tot_RMSE/10)

df = pd.DataFrame({'':result})
df.to_csv("result.csv",index=0,header=0)
    

