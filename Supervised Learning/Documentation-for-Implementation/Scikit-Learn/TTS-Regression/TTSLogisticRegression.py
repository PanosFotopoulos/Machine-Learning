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

coef = log_model.coef_
print(f'my coef are: {coef}')
'''
My coef are: [[-0.94953524  3.45991194]]* and the dataframe with labels Age/physical_score. You can notice that the coef for age is negative
and thats true because as as the age is getting higher/older that mean that more people are failing the test. 
And the coef for physcial is postive that means that they oods increase to pass the test
'''
y_pred = log_model.predict(scaled_X_test,)
#print (y_pred)

y_pred_prob = log_model.predict_proba(scaled_X_test,)
#print (y_pred_prob)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
'''
True Possitive   False Negative
False Possitive  True Negative
'''
my_report = classification_report(y_test,y_pred)

print(f'My classfication reports are: {my_report}')

from sklearn.metrics import precision_score,recall_score

my_precision_score = precision_score(y_test,y_pred)
print(f'My classfication prcision score is: {my_precision_score}')

my_recall_score = recall_score(y_test,y_pred)
print(f'My classfication recall score is: {my_recall_score}')


In_what_class_do_this_belong = log_model.predict_proba(scaled_X_test)[0]
print(f'The probability "%" to be class 0 or 1 is {In_what_class_do_this_belong}')

what_class_actualy_belongs_at = y_test[0]
print(f'Actually belongs at class: {what_class_actualy_belongs_at}')