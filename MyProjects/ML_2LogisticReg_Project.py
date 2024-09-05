import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\heart.csv")
df.describe().T

from sklearn.model_selection import train_test_split

X = df.drop('target',axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV()
model.fit(scaled_X_train,y_train)

print ('Best perfoming C: ', model.C_)
print ('My model parameters: ', model.get_params())
print ('My Model Coefficents: ',model.coef_)

patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]

actual_pred = model.predict(patient)
print(actual_pred)
# so patinet belongs to class 0
