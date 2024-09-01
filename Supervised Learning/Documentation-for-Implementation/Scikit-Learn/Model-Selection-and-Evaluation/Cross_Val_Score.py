import pandas as pd

df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv') 
df.head()

X = df.drop('sales', axis = 1)
y = df['sales']

from sklearn.model_selection import train_test_split

#help(train_test_split)
 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import Ridge

bad_model = Ridge(alpha=100)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(bad_model,X_train,y_train,scoring = 'neg_mean_squared_error',cv=5)

#print (scores.mean())
print (f'The score of the model is: {abs(scores.mean())}') # 8.215396464543607 is not very good

model2 = Ridge(alpha=1)
scores2 = cross_val_score(model2,X_train,y_train,scoring = 'neg_mean_squared_error',cv=5)

#print (scores2.mean())
print (f'The score of the model is: {abs(scores2.mean())}')

model2.fit(X_train,y_train)

y_final_test_pred = model2.predict(X_test)

from sklearn.metrics import mean_squared_error

loss = mean_squared_error(y_test,y_final_test_pred)
print(f'For this model the loss is: {loss}')

