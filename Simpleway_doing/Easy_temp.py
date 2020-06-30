import pandas as pd
import numpy as np
import matplotlib.pylab as plt


sales= pd.read_csv('GlobalTemperatures.csv', index_col = 0, parse_dates= [0])
sales = sales.iloc[:, 0:1]

# Dealing with the Missing Value
temp = sales.fillna(sales['LandAverageTemperature'].mean())
from datetime import datetime

# Ploting the Date-Time Graph
plt.xlabel("UTC Date")
plt.ylabel("Demand of Electricity")
plt.plot(sales)
plt.show()

  
# Import the library 
from pmdarima import auto_arima 
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to  GlobalTemperatures dataset 
fitted_step = auto_arima(temp['LandAverageTemperature'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   
                          suppress_warnings = True,   
                          stepwise = True)           
  
# To print the summary 
fitted_step.summary() 


# Split data into train / test sets 
train = sales.iloc[:len(sales)- 638] 
test = sales.iloc[len(sales)- 638:] # set one year(12 months) for testing 
  
# Fit a SARIMAX(0, 1, 1)x(1, 1, 1, 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['LandAverageTemperature'],  
                order = (1, 0, 1),  
                seasonal_order =(1, 1, 1, 12)) 
  
result = model.fit() 
print(result.summary())

start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True, color = 'green') 
test['LandAverageTemperature'].plot(legend = True, color = 'yellow')  
plt.show()

# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
print(rmse(test['LandAverageTemperature'], predictions))
  
# Calculate mean squared error 
print(mean_squared_error(test['LandAverageTemperature'], predictions))

# Forcasting for next 10 years

# Train the model on the full dataset 
model = SARIMAX(temp['LandAverageTemperature'],  
                        order = (0, 1, 1),  
                        seasonal_order =(2, 1, 1, 12)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(temp),  
                          end = (len(temp)-1) + 10 * 12,  
                          typ = 'levels').rename('Forecast') 
  
# Plot the forecast values 
sales['LandAverageTemperature'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
plt.show()