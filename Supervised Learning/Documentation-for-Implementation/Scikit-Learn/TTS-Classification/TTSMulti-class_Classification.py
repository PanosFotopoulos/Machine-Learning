import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\iris.csv') 

#Some statistical research

df.head() # see the top 5 rows
df.info() # 150 rows
df.describe() # see the value range
df ['species'].value_counts() # the label column


#Develop a model that if u take the sepal_length sepal_width petal_length petal_width to predict what flower is

sns.countplot(x='species',data=df)# perfectly balance
sns.scatterplot(x='petal_length', y='petal_width', data=df, hue='species')
sns.pairplot(df,hue='species')
#satosa is very separaited meanwhile versicolor and virginica can be hard to separate

X = df.drop('species',axis=1)
y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

log_model = LogisticRegression(solver='saga', multi_class='ovr',max_iter=5000)

penalty = ['l1','l2','elasticnet']
l1_ratio = np.linspace(0,1,20)
C = np.logspace(0,10,20) #λ


param_grid = {'penalty':penalty, 'l1_ratio':l1_ratio, 'C':C}

grid_model = GridSearchCV(log_model,param_grid=param_grid)

grid_model.fit(scaled_X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

grid_model.best_params_ # penalty equals to l1

y_pred = grid_model.predict(scaled_X_test)

print(f'My predictions:{y_pred}')

#lets evaluate now the model

model_accuracy = accuracy_score(y_test,y_pred)
print(f'My accuracy:{model_accuracy}')

model_confusion_matrix = confusion_matrix(y_test,y_pred)
print(f'My confusion matrix:{confusion_matrix}')

print(classification_report(y_test,y_pred))


