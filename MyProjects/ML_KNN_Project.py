import pandas as pd
import numpy as np 
import seaborn as sns

df = pd.read_csv(r"C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\sonar.all-data.csv")

df.head()

df['Targe'] = df[  'Label'].map({'R':0, 'M':1})

X = df.drop(['Targe','Label'],axis= 1)
y = df['Targe']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()
knn = KNeighborsClassifier()

operations = [('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)

k_values = list(range(1,30))

param_grid = {'knn__n_neighbors':k_values}

full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(X_train,y_train)
full_cv_classifier.best_estimator_.get_params()
best_k = full_cv_classifier.best_estimator_.get_params()['knn__n_neighbors']
print(f'This is my best k: {best_k}')

#  'knn__n_neighbors': 1


#taking all the k values
from sklearn.metrics import accuracy_score
test_error_rates = []

for k in range(1,30):
    knn_new_model = KNeighborsClassifier(n_neighbors=k)
    knn_new_model.fit(X_train,y_train)
    
    y_pred_test = knn_new_model.predict(X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    
    test_error_rates.append(test_error)

y_pred_test = full_cv_classifier.predict(X_test)

print(f'For {test_error_rates}')

from sklearn.metrics import accuracy_score
test_error = 1 - accuracy_score(y_test,y_pred_test)
print(f'This is my error: {test_error}')

import matplotlib.pyplot as plt

# Extract the results from GridSearchCV
results = full_cv_classifier.cv_results_

# Plot the mean test score (accuracy) for each value of k
plt.figure(figsize=(10, 6))
plt.plot(k_values, results['mean_test_score'], marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('KNN Hyperparameter Tuning (k values)')
plt.grid(True)
plt.show()