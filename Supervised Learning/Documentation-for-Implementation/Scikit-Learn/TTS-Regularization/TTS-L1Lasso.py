import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv')

df.head()

X = df.drop('sales', axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split

#help(train_test_split)
 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#let the scaler learn "meet" ONLY the training data .fit
scaler.fit(X_train)

#Apply The scaler to the data

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#datas are ready to import a model

from sklearn.linear_model import LassoCV

lasso_cv_model = LassoCV(eps=0.1, n_alphas=100, cv=5)

lasso_cv_model.fit(X_train,y_train)

prediction = lasso_cv_model.predict(X_test)

from sklearn.metrics import mean_squared_error

loss = mean_squared_error(y_test,prediction)
print(f'For this model the loss is: {loss}')
