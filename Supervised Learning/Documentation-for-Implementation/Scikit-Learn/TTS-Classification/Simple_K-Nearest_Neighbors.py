import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib as plt


df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\gene_expression.csv') 

df.head()

#run it to see the distribution
sns.scatterplot(df, x='Gene One', y = 'Gene Two', hue = 'Cancer Present')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Cancer Present', axis=1)
y = df['Cancer Present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)# Very important to dont fit the scaler to the data. This prevends data leakage

from sklearn.neighbors import KNeighborsClassifier


knn_model = KNeighborsClassifier(n_neighbors= 1)
knn_model.fit(scaled_X_train,y_train)

y_pred = knn_model.predict(scaled_X_test)
#print(f'This is my 1st prediction: {y_pred}')

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test,y_pred)
#print (f'This is the confusion matrix: {cm}')
cr = classification_report(y_test,y_pred)
#print(f'The classification report for this mode is {cr}')



from sklearn.metrics import accuracy_score
test_error_rates = []

for k in range(1,30):
    knn_new_model = KNeighborsClassifier(n_neighbors=k)
    knn_new_model.fit(scaled_X_train,y_train)
    
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    
    test_error_rates.append(test_error)

'''
plt.plot(range(1,30),test_error_rates)
plt.ylabel('error')
plt.xlabel('K')
'''
#pipeline ---> Gridsearch CV

scaler = StandardScaler()
knn = KNeighborsClassifier()

operations = [('scaler',scaler),('knn',knn)]

from sklearn.pipeline import Pipeline
pipe = Pipeline(operations)

from sklearn.model_selection import GridSearchCV    

k_values = list(range(1,20))

param_grid = {'knn__n_neighbors':k_values}

full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

full_cv_classifier.fit(X_train,y_train)

print(full_cv_classifier.best_estimator_.get_params())

final_pred_ = full_cv_classifier.predict(X_test)

#print(classification_report(y_test,final_pred_))

#now lets say we have a new patient and want to check if he has cancer or not
# his data on gen 1 and gen 2 are:
new_patient = [[11.8,6.3]]


import matplotlib.pyplot as plt

# Extract the results from GridSearchCV
results = full_cv_classifier.cv_results_

best_k = full_cv_classifier.best_estimator_.get_params()['knn__n_neighbors']
print(f'This is my best k: {best_k}')

# Plot the mean test score (accuracy) for each value of k
plt.figure(figsize=(10, 6))
plt.plot(k_values, results['mean_test_score'], marker='o', linestyle='-', color='b')

# Adding labels and title
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('KNN Hyperparameter Tuning (k values)')
plt.grid(True)
plt.show()