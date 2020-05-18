# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:21:00 2020

@author: mehme
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd







data = pd.read_csv('TSLA_1.csv')

data=data.iloc[0:,:]

values = data.values

num_f = len(data.columns)

groups = [x for x in range(num_f)]

plt.figure(figsize = (12,16))

i = 1
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(data.columns[group], y=0.85, loc='center')
    i += 1
plt.show()

print(data.info())

# check missing values
print('null check')
print(data.isnull().values.any())

print(np.var(data['Close'] - data['Adj Close']))

'''
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean= imp_mean.fit(data.iloc[:,1:7])
data.iloc[:,1:7]=imp_mean.transform(data.iloc[:,1:7])
'''

# split


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(data.iloc[:,[1,3,4,6]],data.iloc[:,2:3],test_size=0.33, random_state=1)

'''
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train=sc.fit_transform(X_train)

x_test=sc.fit_transform(X_test)

y_train=sc.fit_transform(Y_train)

y_test =sc.fit_transform(Y_test)

#svr
from sklearn.svm import SVR
svr_reg =SVR(kernel ='rbf')
svr_reg.fit(x_train,y_train)
pre=svr_reg.predict(x_test)
'''
# liner

from sklearn.linear_model import LinearRegression

reg_li =LinearRegression()
reg_li.fit(x_train,y_train)


y_pred_li=reg_li.predict(x_test)


# random 

from sklearn.ensemble import RandomForestRegressor

reg_rf =RandomForestRegressor(n_estimators=10 , random_state=0)

reg_rf.fit(x_train,y_train)

y_pred_rf=reg_rf.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((2488,1)).astype(int), values= data, axis=1)

X_l=data.iloc[:,[1,3,4,6]].values

model = sm.OLS(data.iloc[:,2:3].values,X_l).fit()
print(model.summary())


import statsmodels.api as sm

X = np.append(arr = np.ones((2488,1)).astype(int), values= data, axis=1)

X_l=data.iloc[:,[1,4,6]].values

model = sm.OLS(data.iloc[:,2:3].values,X_l).fit()
print(model.summary())



# score

from sklearn.metrics import max_error
print("--------")
print('max error rf')
print(max_error(y_test, y_pred_rf))
print('max error li')
print(max_error(y_test, y_pred_li))
#print(max_error(y_test, pre))

from sklearn.metrics import mean_absolute_error
print("--------")
print('mean absolte error rf')
print(mean_absolute_error(y_test, y_pred_rf))
print('mean absolte error li')
print(mean_absolute_error(y_test, y_pred_li))
#print(mean_absolute_error(y_test, pre))


from sklearn.metrics import r2_score
print("--------")
print('r2 score for rf')
print(r2_score(y_test, y_pred_rf))
print('r2 score for rf li')
print(r2_score(y_test, y_pred_li))
#print(r2_score(y_test,pre))





