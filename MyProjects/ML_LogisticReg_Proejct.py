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

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(scaled_X_train,y_train)

#from sklearn.model_selection import cross_validate

#cv_results = cross_validate(model1, scaled_X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'],return_train_score=False)
#print(cv_results)

from sklearn.model_selection import cross_val_score

cross_val_scoring = cross_val_score(model1, scaled_X_train, y_train, cv=5, scoring='balanced_accuracy')
print (cross_val_scoring)

#After the model is ready and we can actually accomplish a prediction

y_pred = model1.predict(scaled_X_test)

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test, y_pred)
print (acc_score)



