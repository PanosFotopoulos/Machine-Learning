import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\AMES_Final_DF.csv")

#df.info()
#df.head()


# Lets separate our data X and y. Im trying to predict SalePrice

X = df.drop('SalePrice',axis = 1)
y = df['SalePrice']

# Import train-test-split from sklearn

from sklearn.model_selection import train_test_split

#help(train_test_split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
transformed_X_train = scaler.transform(X_train)
transformed_X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNet

base_elastic_net_model = ElasticNet(max_iter=100000)

param_grid = {'alpha':[0.1, 1, 5, 10, 100],'l1_ratio':[0.1 , 0.7, 0.99, 1]}

from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator=base_elastic_net_model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=5 ,verbose=0)

grid_model.fit(transformed_X_train,y_train)

y_pred = grid_model.predict(transformed_X_test)


print(f'This is my grid model best parameters: {grid_model.best_params_}')

from sklearn.metrics import mean_squared_error,mean_absolute_error

loss = np.sqrt(mean_squared_error(y_test,y_pred))
print(f'For this model the loss is: {loss}')

MAE = mean_absolute_error(y_test,y_pred)
print(f'MAE :{MAE}')