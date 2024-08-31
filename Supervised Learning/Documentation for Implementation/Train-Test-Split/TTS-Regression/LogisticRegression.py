import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\hearing_test.csv') 

X = df.drop('test_result',axis=1)
y = df['test_result']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(scaled_X_train,y_train)

log_model.coef_

y_pred = log_model.predict(scaled_X_test,)
#print (y_pred)

y_pred_prob = log_model.predict_proba(scaled_X_test,)
#print (y_pred_prob)
