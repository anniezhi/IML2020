### IML Task 1 ###
# Xiaoying Zhi
# 21 Mar 2020
# cross-validation for ridge regression

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

## load csv ##
all_data = pd.read_csv('train.csv')
#all_id = np.array(all_data['Id'])
all_data = all_data.drop('Id', axis=1)
#print(all_data.head())

## separate y and x ##
all_y = np.array(all_data.y)
#print(all_y.shape)
all_x = np.array(all_data.iloc[:,all_data.columns!='y'])
#print(all_x.shape)

## ridge regression ##
alpha_set = np.array([0.01, 0.1, 1, 10, 100])
rmse_avg_all = []
for alpha in alpha_set:
	#K folds
	kf = KFold(n_splits=10, shuffle=True, random_state=42)
	rmse_total = 0
	for train_index, test_index in kf.split(all_x):
		train_x, test_x = all_x[train_index], all_x[test_index]
		train_y, test_y = all_y[train_index], all_y[test_index]
		reg_model = Ridge(alpha=alpha, tol=1e-4).fit(train_x, train_y)
		pred_y = reg_model.predict(test_x)
		rmse = mean_squared_error(test_y, pred_y, squared=False)
		rmse_total += rmse
	rmse_avg = rmse_total / 10
	rmse_avg_all.append(rmse_avg)

## export output ##
output = pd.DataFrame(rmse_avg_all)
print(output)
output.to_csv('result.csv', index=False, header=False)

