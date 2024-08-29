import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv')

X = df.drop('sales', axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split

#help(train_test_split)
 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNet

#help(ElasticNet)

base_elastic_net_model = ElasticNet()

param_grid = {'alpha':[0.1, 1, 5, 10, 50, 100],'l1_ratio':[0.1 , 0.5, 0.7, 0.95, 0.99, 1]}

from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator=base_elastic_net_model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=5 ,verbose=0)

grid_model.fit(X_train,y_train)

print(grid_model.best_estimator_)
print(grid_model.best_params_)

pd.DataFrame(grid_model.cv_results_)

y_pred = grid_model.predict(X_test)

from sklearn.metrics import mean_squared_error

loss = mean_squared_error(y_test,y_pred)
print(f'For this model the loss is: {loss}')