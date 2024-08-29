import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv') #apparently r is needed in V.S. code


X = df.drop('sales', axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=100)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

residuals =  y_test - prediction # if model was perfect all would be zeros
#print (residuals)

from sklearn.metrics import mean_squared_error,mean_absolute_error

MSE = mean_squared_error(y_test,prediction) #Loss
MAE = mean_absolute_error(y_test,prediction)
RMSE = np.sqrt(MSE)
RMAE = np.sqrt(MAE)

'''
Run them for clarification
print(f'This is my MSE: {MSE}')
print(f'This is my MAE: {MAE}')
print(f'This is my RMSE: {RMSE}')
print(f'This is my RMAE: {RMAE}')
'''

model.coef_
loss = mean_squared_error(y_test,prediction)
print(f'My model loss is at {loss}')
print(f'My model coefficients: {model.coef_}')
coeff_dif = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
#print(f'The gap between coefficients: {coeff_dif}')
