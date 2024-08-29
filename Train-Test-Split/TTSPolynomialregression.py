import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Panos\Desktop\PythonforMLandDTScience\DATA\Advertising.csv') 

X=df.drop('sales',axis=1)
y=df['sales']

from sklearn.preprocessing import PolynomialFeatures

#2nd order Polynomial
poly_converter = PolynomialFeatures(degree=2, include_bias=False)
poly_converter.fit(X)

poly_features = poly_converter.transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,y_train)

test_prediction=model.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

MAE = mean_absolute_error(y_test,test_prediction)
MSE = mean_squared_error(y_test,test_prediction)

'''
Run them for clarification
print(f'This is my MSE: {MSE}')
print(f'This is my MAE: {MAE}') 
'''

#create a different degree polynomial

train_rmse_error_list = []
test_rmse_error_list = []

for best_degree_search in range (1,10):
    
    searching_for_optimal_poly_converter = PolynomialFeatures(degree=best_degree_search , include_bias= False)
    #fit and transform
    searching_for_optimal_poly_features = searching_for_optimal_poly_converter.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(searching_for_optimal_poly_features, y, test_size=0.3, random_state=101)
    
    searching_for_optimal_model = LinearRegression()
    searching_for_optimal_model.fit(X_train,y_train)
    
    optimal_train_preidiction = searching_for_optimal_model.predict(X_train)
    optimal_test_preidiction = searching_for_optimal_model.predict(X_test)
    
    train_rmse_error = np.sqrt(mean_absolute_error(y_test,optimal_test_preidiction))
    test_rmse_error = np.sqrt(mean_squared_error(y_test,optimal_test_preidiction))
    
    #evaluate by rmse
    train_rmse_error_list.append(train_rmse_error)
    test_rmse_error_list.append(test_rmse_error)

for index,value in enumerate(test_rmse_error_list,start=1):
    print(f'{index} degree polynomial got error equal to {value}')
       


'''
Test the plots on jupyter
plt.plot(range(1,10),train_rmse_error)
plt.plot(range(1,10),test_rmse_error)

Only by printing (with out even plotting) the errors from the test , you can observe that on the 6th degree the error is getting explode.
So lets print again till the 5th degree

plt.plot(range(1,6),train_rmse_error[:5])
plt.plot(range(1,6),test_rmse_error[:5])
'''    


#Most valid and safe is 3rd order degree, so lets set our final model


optimal_poly_converter = PolynomialFeatures(degree=3,include_bias=False)
optimal_model = LinearRegression()
optimal_X_converter = optimal_poly_converter.fit_transform(X)
optimal_model.fit(optimal_X_converter,y)

print(f'My final model coefficient are {optimal_model.coef_}')