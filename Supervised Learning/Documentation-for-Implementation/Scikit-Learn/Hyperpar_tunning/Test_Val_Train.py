import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv')

X = df.drop('sales', axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split

#help(train_test_split)
 
X_train, X_other, y_train, y_other = train_test_split( X, y, test_size=0.3, random_state=101)


# test_size = 0.5 (50% of 30% other ---> test = 15% of all data)
X_eval,X_test, y_eval, y_test = train_test_split( X_other, y_other, test_size=0.5, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_eval = scaler.transform(X_eval)

from sklearn.linear_model import Ridge

model1 = Ridge(alpha=100)# poor model

model1.fit(X_train, y_train)

y_eval_pred = model1.predict(X_eval)

from sklearn.metrics import mean_squared_error

loss1 = mean_squared_error(y_eval,y_eval_pred)
print(f'For this model the loss is: {loss1}')

model2 = Ridge(alpha=1 )

model2.fit(X_train,y_train)

y_eval_pred2 = model2.predict(X_eval)

loss2 = mean_squared_error(y_eval,y_eval_pred2)
print(f'For this model the loss is: {loss2}')


final_eval_pred = model2.predict(X_test)

final_loss = mean_squared_error(y_test,final_eval_pred)
print(f'For my final model the loss is: {final_loss}')

from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

ridge = Ridge()

grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print(f'Best alpha: {grid_search.best_params_}')

best_ridge = grid_search.best_estimator_
y_eval_pred_best = best_ridge.predict(X_eval)
best_loss = mean_squared_error(y_eval, y_eval_pred_best)
print(f'Best model evaluation loss: {best_loss}')





