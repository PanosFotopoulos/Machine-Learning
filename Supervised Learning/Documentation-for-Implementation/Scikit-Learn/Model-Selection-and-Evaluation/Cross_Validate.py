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

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge


bad_model = Ridge(alpha=100)

bad_scores = cross_validate(bad_model,X_train,y_train,scoring=['neg_mean_squared_error','neg_mean_absolute_error'],cv=10)
bad_scores = pd.DataFrame(bad_scores)
#scores.mean()

#Improve the model

model = Ridge(alpha=1)
scores = cross_validate(model,X_train,y_train,scoring=['neg_mean_squared_error','neg_mean_absolute_error'],cv=10)
scores = pd.DataFrame(scores)
#scores.mean()

model.fit(X_train,y_train)

y_final_prediction = model.predict(X_test)
from sklearn.metrics import mean_squared_error
loss = mean_squared_error(y_test,y_final_prediction)
print(f'For this model the loss is: {loss}')
