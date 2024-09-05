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

model = LogisticRegression()
model.fit(scaled_X_train,y_train)

#from sklearn.model_selection import cross_validate

#cv_results = cross_validate(model1, scaled_X_train, y_train, cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'],return_train_score=False)
#print(cv_results)

#from sklearn.model_selection import cross_val_score

#cross_val_scoring = cross_val_score(model1, scaled_X_train, y_train, cv=5, scoring='balanced_accuracy')
#print (cross_val_scoring)

#After the model is ready and we can actually accomplish a prediction

from sklearn.model_selection import cross_validate

cross_validate_scores = cross_validate(model, scaled_X_train, y_train, cv =5, scoring=['accuracy', 'precision', 'recall', 'f1'])

y_pred = model.predict(scaled_X_test)

#from sklearn.metrics import accuracy_score

#acc_score = accuracy_score(y_test, y_pred)
#print (acc_score)

from sklearn.model_selection import GridSearchCV

param_grid = {
                'C':[0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'newton-cg', 'lbfgs', 'saga'] 
              }

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='r2')

grid_search.fit(scaled_X_train,y_train)

print('Best Parameters: ', grid_search.best_params_)
print('Best Score: ', grid_search.best_score_)
print('Best Estimator: ', grid_search.best_estimator_)
print('Best Index: ', grid_search.best_index_)


#Best Parameters:  {'C': 1, 'solver': 'liblinear'}
#Best Score:  0.3174137931034485
#Best Estimator:  LogisticRegression(C=1, solver='liblinear')
#Best Index:  8

# so now we can actually make our best model

best_model = grid_search.best_estimator_
y_pred = best_model.predict(scaled_X_test)

from sklearn.metrics import classification_report, accuracy_score

print("Best Parameters:", grid_search.best_params_)
print("Best Score on Training Data:", grid_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))