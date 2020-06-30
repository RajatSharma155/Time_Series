# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 09:26:38 2020

@author: Rajat sharma
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


temp = pd.read_csv('GlobalTemperatures.csv', index_col = 0, parse_dates= [0])
temp = temp.iloc[:, 0:1]

# Dealing with the Missing Value
temp = temp.fillna(temp['LandAverageTemperature'].mean())

# Ploting the Date-Time Graph
plt.xlabel("UTC Date")
plt.ylabel("LandAverage Temperature")
plt.plot(temp)
plt.show()

# Checking the stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window = 12).mean()
    rolstd = timeseries.rolling(window= 12).std()
    print(rolmean, rolstd)
    

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationarity(temp)

# Log Conversion For trend Elemination
value = temp

# Trend Elimination
moving_avg = value.rolling(window = 12).mean()
plt.plot(value)
plt.plot(moving_avg, color='red')
plt.title("Moving Average")
plt.show()

value_moving_avg_diff = value - moving_avg

value_moving_avg_diff.dropna(inplace=True)
test_stationarity(value_moving_avg_diff)


# Sesonality elimination

# Differencing
value_diff = value - value.shift()
plt.plot(value_diff)
plt.show()

value_diff.dropna(inplace=True)
test_stationarity(value_diff)

# Decomposing
from statsmodels.tsa.seasonal import seasonal_decompose
value = value.fillna(value['LandAverageTemperature'].mean())
decomposition = seasonal_decompose(value)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(value, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Now dealing with the residuals
decomposed_data = residual
decomposed_data = decomposed_data.to_frame()
decomposed_data.rename(columns = {'resid':'LandAverageTemperature'}, inplace = True)
decomposed_data.dropna(inplace = True)
test_stationarity(decomposed_data)


# Now plotting the ACF and PACF Graph for p and q
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(value_diff, nlags=20)
lag_pacf = pacf(value_diff, nlags=20, method='ols')

# Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle="--", color= 'gray')
plt.axhline(y=-1.96/np.sqrt(len(value_diff)), linestyle="--", color= 'gray')
plt.axhline(y=1.96/np.sqrt(len(value_diff)), linestyle="--", color= 'gray')
plt.title('AutoCorrelation Function')

# Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle="--", color= 'gray')
plt.axhline(y=-1.96/np.sqrt(len(value_diff)), linestyle="--", color= 'gray')
plt.axhline(y=1.96/np.sqrt(len(value_diff)), linestyle="--", color= 'gray')
plt.title('Partial-AutoCorrelation FUnction')
plt.tight_layout()
plt.show()

# Making the Arima model
# AR
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(value, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(value_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %4f'% sum((results_AR.fittedvalues-value_diff['LandAverageTemperature'])**2))
plt.show()

# MA
model = ARIMA(value, order=(0, 1, 3))  
results_MA = model.fit(disp=-1)  
plt.plot(value_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-value_diff['LandAverageTemperature'])**2))
plt.show()

# Combined
model = ARIMA(value, order=(2, 1, 3))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(value_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-value_diff['LandAverageTemperature'])**2))
plt.show()

# Taking Back the value
predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(value.iloc[:, 0], index=value.index)
predictions_ARIMA = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

plt.plot(temp)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA[0]-temp['LandAverageTemperature'])**2)/len(temp)))
plt.show()

# Forecast for the next 100 years 
results_ARIMA.plot_predict(1, 3312)
X = results_ARIMA.forecast(steps = 1200)[0]
converted_results = np.exp(X)
plt.title("last")
plt.show()


# Forecast for the next 100 years 
forecast = results_ARIMA.predict(start = len(temp),  
                          end = (len(temp)-1) + 100 * 12,  
                          typ = 'levels').rename('Forecast') 
  
# Plot the forecast values 
temp['LandAverageTemperature'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True)
plt.show()






