import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\AMES_Final_DF.csv")

df.info()
df.head()


# Lets separate our data X and y. Im trying to predict SalePrice

X = df.drop('SalePrice',axis = 1)
y = df['SalePrice']

# Import train-test-split from sklearn

from sklearn.model_selection import train_test_split

#help(train_test_split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


from sklearn.linear_model import ElasticNetCV

elastic_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.7, 0.95, 0.99, 1], #0.1 equals to 10% lasso and 90% ridge etc..
                             eps=0.001, n_alphas= 100, max_iter=1000000) 

elastic_model.fit(X_train,y_train)

print (f' This is the l1 ratio {elastic_model.l1_ratio}, and this is the best performing l1 ratio: {elastic_model.l1_ratio_}' ) # That means Lasso perform better than Ridge

elastic_predictions = elastic_model.predict(X_test)

print (f'With this prediction: {elastic_predictions}')


