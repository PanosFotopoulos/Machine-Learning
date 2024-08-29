import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from TTSLinearRegression import LinearRegression
from TTSLinearRegression import X, y, RMAE, RMSE

final_model = LinearRegression()
print(X,y)

final_model.fit(X,y)

final_model_coef = final_model.coef_

print (f'My final model coefficient are {final_model_coef}')

#Test the final model on random data

campaign = [[149,22,12]]
camplain_prediction = final_model.predict(campaign)
print(f'We expect to get {camplain_prediction} units of sales for the campaign, with RMAE: {RMAE} and RMSE: {RMSE}!')

