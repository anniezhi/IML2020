### IML Task 1b ###
# Xiaoying Zhi
# 21 Mar 2020
# linear regression with feature transformers

import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Lasso, Ridge, LinearRegression
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
all_x_linear = np.array(all_data.iloc[:,all_data.columns!='y'])
#print(all_x.shape)

## transform ##
transformer_quadr = FunctionTransformer(np.square)
all_x_quadr = transformer_quadr.transform(all_x_linear)
#print(all_x_quadr)
transformer_exp = FunctionTransformer(np.exp)
all_x_exp = transformer_exp.transform(all_x_linear)
#print(all_x_exp)
transformer_cos = FunctionTransformer(np.cos)
all_x_cos = transformer_cos.transform(all_x_linear)
#print(all_x_cos)
all_x_ones = np.ones((len(all_y),1))

all_x = np.concatenate((all_x_linear,all_x_quadr,all_x_exp,all_x_cos,all_x_ones),axis=1)

## feature selection ##
sel = SelectKBest(k=18)
all_x_sel = sel.fit_transform(all_x, all_y)
sel_result = sel.get_support(indices=False)   #list of selected features
all_index = np.arange(0,21)
sel_in_index = all_index[sel_result]
sel_out_index = all_index[~sel_result]
print(sel_in_index)
print(sel_out_index)

## ridge regression ##
#alpha_set = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
rmse_avg_all = []
alpha = 10
#K folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_total = 0
models = []
rmses = []
for train_index, test_index in kf.split(all_x_sel):
	train_x, test_x = all_x_sel[train_index], all_x_sel[test_index]
	train_y, test_y = all_y[train_index], all_y[test_index]
	reg_model = Ridge(alpha=alpha, tol=1e-4).fit(train_x, train_y)
	models.append(reg_model)
	pred_y = reg_model.predict(test_x)
	rmse = mean_squared_error(test_y, pred_y, squared=False)
	rmses.append(rmse)
	#print(rmse)
	#rmse_total += rmse
#rmse_avg = rmse_total / 10
#rmse_avg_all.append(rmse_avg)

models = np.array(models)
rmses = np.array(rmses)
best_model_idx = np.argmin(rmses)
ridge_coefs = models[best_model_idx].coef_
print(ridge_coefs)
print(rmses[best_model_idx])


## export output ##
all_coefs = np.zeros((21,))
all_coefs[sel_in_index] = ridge_coefs
print(all_coefs)

output = pd.DataFrame(all_coefs)
#print(output)
output.to_csv('result_2.csv', index=False, header=False)

