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
print (f'This is the confusion matrix: {cm}')
cr = classification_report(y_test,y_pred)
print(f'The classification report for this mode is {cr}')