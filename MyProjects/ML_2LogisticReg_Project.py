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
