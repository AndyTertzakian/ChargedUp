# Databricks notebook source
# MAGIC %md
# MAGIC # Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC # Import & Setup

# COMMAND ----------

# import libraries
import pandas as pd
import math
# import altair as alt
import datetime as dt
import numpy as np
import json
import time
import urllib
import requests
import seaborn as sns
from pandas.io.json import json_normalize
from datetime import timedelta
import plotly.graph_objects as go
import pytz
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
import folium
import holidays
from calendar import monthrange

from statsmodels.tsa.seasonal import seasonal_decompose

# COMMAND ----------

## Modeling Libraries

# Model Evaluation
from sklearn.metrics import mean_squared_error as mse

# ARIMA/VAR Models
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet


# holt winters -- good for seasonality, probably not best for us
# single exponential smoothing
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
#from statsmodels.tsa.holtwinters import ExponentialSmoothing

# COMMAND ----------

## Connect to AWS
secret_scope = "w210-scope"
access_key = dbutils.secrets.get(scope=secret_scope, key="aws-access-key")
secret_key = dbutils.secrets.get(scope=secret_scope, key="aws-secret-key")
encoded_secret_key = secret_key.replace("/", "%2F")
aws_bucket_name = "w210v2"
mount_name = "w210v2"

try:
    dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
except Exception as e:
    print("Already mounted :)")

# COMMAND ----------

display(dbutils.fs.ls(f"/mnt/{mount_name}/data/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Datasets

# COMMAND ----------

slrp = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_ts")
boulder = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_ts")
palo_alto = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_ts")

# COMMAND ----------

# Convert to Pandas Datafames
slrp = slrp.toPandas()
boulder = boulder.toPandas()
palo_alto = palo_alto.toPandas()

# COMMAND ----------

print(slrp.shape)
print(boulder.shape)
print(palo_alto.shape)
print(slrp.columns)
print(boulder.columns)
print(palo_alto.columns)

# COMMAND ----------

slrp['Ports Occupied'].value_counts()

# COMMAND ----------

boulder['Ports Occupied'].value_counts()

# COMMAND ----------

palo_alto['Ports_Occupied'].value_counts()

# COMMAND ----------

# Calculate Time Deltas to confirm the dataframes are approximately the correct sizes
slrp_td = slrp['DateTime'].max() - slrp['DateTime'].min()
print(slrp_td)
print((slrp_td).days*24*6)
print(slrp.shape)

print()

boulder_td = boulder['Date Time'].max() - boulder['Date Time'].min()
print(boulder_td)
print((boulder_td).days*24*6)
print(boulder.shape)

print()

palo_alto_td = palo_alto['DateTime'].max() - palo_alto['DateTime'].min()
print(palo_alto_td)
print((palo_alto_td).days*24*6)
print(palo_alto.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC Ideas
# MAGIC - Hour of day
# MAGIC - Day of week
# MAGIC - Cosine/Sine similarities for cyclical features (hour of day, day of week)
# MAGIC - Weekend/holiday indicator
# MAGIC - One-hot encode any categorical features (ex. `note` in `slrp`)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SlrpEV

# COMMAND ----------

###### SLRP ############
# Make a copy
slrp_transformed = slrp.copy()

# One hot encode the note column
slrp_transformed = pd.concat([slrp_transformed, 
                              pd.get_dummies(slrp_transformed['note'], prefix = 'note')], axis = 1)\
                    .drop(columns = ['note', 'note2'])

# slrp_transformed['dayofweek'] = slrp_transformed['DateTime'].dt.dayofweek

# Apply cosine and sine transformations to cyclical features
slrp_transformed['month_cosine'] = np.cos(2 * math.pi * slrp_transformed['Month'] / slrp_transformed['Month'].max())
slrp_transformed['month_sine'] = np.sin(2 * math.pi * slrp_transformed['Month'] / slrp_transformed['Month'].max())
slrp_transformed['hour_cosine'] = np.cos(2 * math.pi * slrp_transformed['DateTime'].dt.hour / 
                                         slrp_transformed['DateTime'].dt.hour.max())
slrp_transformed['hour_sine'] = np.sin(2 * math.pi * slrp_transformed['DateTime'].dt.hour / 
                                       slrp_transformed['DateTime'].dt.hour.max())
slrp_transformed['dayofweek_cosine'] = np.cos(2 * math.pi * slrp_transformed['DayofWeek'] / 
                                              slrp_transformed['DayofWeek'].max())
slrp_transformed['dayofweek_sine'] = np.sin(2 * math.pi * slrp_transformed['DayofWeek'] / 
                                            slrp_transformed['DayofWeek'].max())
slrp_transformed['IsWeekend'] = slrp_transformed['IsWeekend'].astype(int)

# Drop unnecessary columns
slrp_transformed = slrp_transformed.drop(columns = ['DayofWeek', 'Month', 'Year', 'Date', 'Year-Month', 'Ports Occupied', 'Plugs'])

# Drop other unnecessary columns that we might need later though
slrp_transformed = slrp_transformed.drop(columns = ['station', 'power_W', 'Ports Charging'])

# Sort by DateTime
slrp_transformed = slrp_transformed.sort_values(by = 'DateTime')

slrp_transformed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

### Palo Alto ####
pa_transformed = palo_alto[['Station_Location', 'datetime_pac', 'Ports_Available']].copy()
pa_transformed

# COMMAND ----------

### Boulder ###

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seasonality 

# COMMAND ----------

### look at seasonality
df =slrp
df.set_index('DateTime')

results = seasonal_decompose(df['Ports Occupied'], period = 6*24*7) #period = 1 week, 6*24*7
#results = seasonal_decompose(df['Ports Occupied'], period = 6*24) #period = period = 1 day 6*24

fig = results.plot(observed=True, seasonal=True, trend=True, resid=True)
fig.set_size_inches((20,10))
plt.show()

# consider zooming in to view

# COMMAND ----------

### look at seasonality
df = pa_transformed[pa_transformed['Station_Location'] == 'BRYANT']
df.set_index('datetime_pac')


#def seasonality_plot_filters():
#    '''Creates a plot with a drop down menu to view seasonal decompose'''

def seas_plot(df, columnname, periodvalue):
    results=seasonal_decompose(df[str(columnname)], period = periodvalue)
    fig = results.plot(observed=True, seasonal=True, trend=True, resid=True)
    fig.set_size_inches((20,10))
    
    return fig


seas_plot(df, 'Ports_Available', 6*24*7)

# considering zooming in to view

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Parameters
# MAGIC 
# MAGIC Predictor:
# MAGIC - Number of stations available
# MAGIC 
# MAGIC Model Types:
# MAGIC - ARIMA Models
# MAGIC - LSTM Models
# MAGIC - CNN Models
# MAGIC - FB Prophet

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare For Models

# COMMAND ----------

# Make datetime the index
#df.set_index('DateTime')

# Make first column the output col

# Remaining cols as other features

# COMMAND ----------

# DBTITLE 0,Filter df
def filter_df(df, date_col, start_date, train_months, val_weeks, test_weeks=0, window_size = 6):
    """
    Filter the dataframe to only the train, validation, and test dates
    
    date_col = column of dataframe with datetime info
    start_date should have format of YYYY-MM-DD
    """
    
    split = start_date.split('-')
    start_year = int(split[0])
    start_month = int(split[1])
    start_day = int(split[2])
    start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
    
    total_days = 0
    next_year = start_year
    next_month = start_month
    
    while train_months > 0:
        days_in_month = monthrange(next_year, next_month)[1]
        next_month = next_month + 1 if next_month < 12 else 1
        next_year = next_year + 1 if next_month < 12 else next_year
        total_days = total_days + days_in_month
        train_months = train_months - 1
        
    train_end_date = start_date + dt.timedelta(days = total_days)
    
    val_end_date = train_end_date + dt.timedelta(days = int(val_weeks*7))
    test_end_date = val_end_date + dt.timedelta(days = int(test_weeks*7))
    
    print(start_date, train_end_date, val_end_date, test_end_date)
    
    df = df[(df[date_col] >= start_date) & (df[date_col] < test_end_date)]
    
    df['TrainDevTest'] = np.where(df[date_col] < train_end_date, 'train', np.where(df[date_col] < val_end_date, 'dev', 'test'))
    
    train_end_index = df[df['DateTime'] == df[df[date_col] == train_end_date]['DateTime'].min()].index[0] - (window_size - 1)
    dev_end_index = df[df['DateTime'] == df[df[date_col] == val_end_date]['DateTime'].min()].index[0] - (window_size - 1)
    print(train_end_index)
    print(dev_end_index)
    
    return train_end_index, dev_end_index, df.reset_index(drop = True)

# COMMAND ----------

def train_dev_test_split(df, y_var):
    """
    Input is a dataframe that has already passed through the filtering function
    df should have a column called TrainDevTest indicating which group the datetime entry belongs in
    """
    
    

# COMMAND ----------

train_end_index, dev_end_index, trial = filter_df(slrp_transformed, 'DateTime', '2022-01-01', 1, 1, 1)
# print(trial.head())
# print(trial.tail())
print(train_end_index, dev_end_index)

# COMMAND ----------

def select_prior_months(df, date_col, train_months, start_date = '2021-01-01'):
    """
    Filter the dataframe to only a certain number of months, as defined by num_months
    
    At the end, set the index equal to the date_col column
    
    date_col is the column of the dataframe with the datetime info
    
    start_date should have format of YYYY-MM-DD
    """
    
    split = start_date.split('-')
    start_year = int(split[0])
    start_month = int(split[1])
    start_day = int(split[2])
    start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
    
    total_months = train_months / 0.7
    
    end_date = start_date + dt.timedelta(days = int(total_months * 30))
    
#     end_day = start_day + int(((train_months / 0.7) - int(train_months/0.7))*30)
#     end_month = start_month + total_months if start_month < (12-total_months+1) else total_months - (12 - start_month)
#     end_year = start_year if start_month < (12-train_months+1) else start_year + 1 
#     end_date = dt.datetime(year = end_year, month = end_month, day = end_day)
    
    print(start_date)
    print(end_date)
    
    temp = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
    
    return temp.set_index(date_col)

# COMMAND ----------

def df_to_X_y(df, window_size=6):
    """ 
    Tranform pandas dataframe given a window_size for LSTM model
    Window_size, is the the number of inputs, to predict window_size+1 (or i + window_size)
    Returns 2 numpy arrays. Inputs (features), and actual labels
    """
    
    df_as_np = df.to_numpy()
    
    #X is matrix of inputs given window size
    #y is actual outputs
    X = []
    y = []
    
    # iterate to create matrix
    for i in range(len(df_as_np) - window_size): 
        #take the i and the next values, makes a list of list for multiple inputs
        row = [r for r in df_as_np[i:i+window_size]] 
        X.append(row)
        # taking the element after window size
        label = df_as_np[i + window_size][0] 
        y.append(label)
    
    return np.array(X), np.array(y)
#     return pd.DataFrame(X)

# COMMAND ----------

# def df_to_X_y2(df, n_input=6, n_out = 1):
#     """ 
#     Tranform pandas dataframe into arrays of inputs and outputs
#     n_inputs, is the the number of inputs, to use to predict n_input+1 (or i + n_input) aka window size
#     n_outs, is the number of steps out you want to predict. Default is 1
#     Returns 2 numpy arrays. Inputs (features), and actual labels
#     """
    
#     df_as_np = df.to_numpy()
    
#     #X is matrix of inputs given window size
#     #y is actual outputs
#     X = []
#     y = []
    
#     # iterate to create matrix
#     for i in range(len(df_as_np) - n_out +1): 
#         #take the i and the next values, makes a list of list for multiple inputs
#         row = [r for r in df_as_np[i:i+window_size]] 
#         X.append(row)
#         
#         # taking the element after window size
#         label = [df_as_np[i + window_size][0:n_out-1]] 
#         y.append(label)
    
#     return np.array(X), np.array(y)

# COMMAND ----------



# COMMAND ----------

df_to_X_y(trial)

# COMMAND ----------

def split_train_test(X, y):
    '''
    Split inputs and labels into tain, validation and test sets for LSTMs
    Returns x and y arrays for train, validation, and test.
    '''
    
    # get size of Array
    num_timestamps = X.shape[0]
    
    # define split proportions - can consider making this input to functions
    train_proportion = 0.7 # consider input
    dev_proportion = 0.15 # consider input
    test_proportion = (1 - train_proportion - dev_proportion) # can leave this as is
    
    # define split points
    train_start = 0 # do we need this? can we always start at 0?
    train_end = int(num_timestamps * train_proportion)
    dev_end = int(num_timestamps * (train_proportion + dev_proportion))
    
    # splitting
    X_train, y_train = X[train_start:train_end], y[train_start:train_end]
    X_val, y_val = X[train_end:dev_end], y[train_end:dev_end]
    X_test, y_test = X[dev_end:], y[dev_end:]
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[0]

# COMMAND ----------

slrp_transformed.columns

# COMMAND ----------

### for multivariate lstm

# Prepare X and y training/testing data
slrp_filtered = select_prior_months(slrp_transformed, 'DateTime', 3, '2022-01-01')
X, y = df_to_X_y(slrp_filtered)
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC # ARIMA Models

# COMMAND ----------

def arima_filter(df, start_date, end_date, date_col):
    ''' 
    Filter df based on the date_col. Pull dates greater than or equal to start_date
    and less than end_date
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)]

# COMMAND ----------

def arima_eda(df, target_var, lags_pacf, lags_acf):
    ''' 
    plot the pacf - used to determine order of AR
    plot acf - used to determine order of MA
    prints our if time series is stationary or not using adfuller p-value threshold of 0.05
    target_var: df column of interest, that you want to predict
    lags_pacf: lags to check for
    lags_acf: lags to check for
    '''
    
    # how correlated are different time periods
    plot_pacf(df[target_var], lags = lags_pacf)
    plot_acf(df[target_var], lags = lags_acf)
    
    
    # p-value to determine if non-stationary.
    p_val = adfuller(df[target_var])[1]
    print('p value of adfuller test:\t', p_val)
    
    if p_val <= 0.05:
        print('time series is stationary')
    else:
        print('nonstationary time series')
    
    

# COMMAND ----------

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 4, 1)
date_col = 'DateTime'
slrp_arima = arima_filter(slrp_transformed, start, end, date_col)
arima_eda(slrp_arima, 'Ports Available', 25, slrp_arima.shape[0]-1)

# Check seasonality here per second graph, daily or weekly
#Autocorrelation takes into account a lot of indirect effects

# COMMAND ----------

arima_eda(slrp_transformed, 'Ports Available', 25, 78000)

# COMMAND ----------

auto_arima(slrp_arima['Ports Available'], d = 0, trace = True, suppress_warnings = True)

# Why is the degree 2 from the PACF graph not showing up? Do we trust these results?

# COMMAND ----------

# parameters based on data/EDA/PACF/ACF graphs
model = ARIMA(slrp_arima['Ports Available'], order = (144, 0, 2))
model = model.fit()
# model.summary()

# COMMAND ----------

pred = model.predict(start = slrp_arima.shape[0], end = slrp_arima.shape[0] + 24*6*14 - 1, typ='levels')
actual = arima_filter(slrp_transformed, end, end + dt.timedelta(days = 14), 'DateTime')
plt.subplots(figsize = (15, 7))
plt.plot(actual['DateTime'], pred)
plt.plot(actual['DateTime'], actual['Ports Available'])
plt.plot(actual['DateTime'], np.around(pred))
print('MSE:\t', mse(actual['Ports Available'], pred))
print('MSE (rounded):', mse(actual['Ports Available'], np.around(pred)))

# COMMAND ----------

actual['Ports Available'][0:6]

# COMMAND ----------

np.around(pred[0:50])

# COMMAND ----------

pacf = plot_pacf(slrp_transformed['Ports Available'], lags = 25)
acf = plot_acf(slrp_transformed['Ports Available'], lags = 5000)

# COMMAND ----------

dftest = adfuller(slrp_transformed['Ports Available'])
print(dftest)

print('ADF: ', dftest[0])
print('P-Value: ', dftest[1])
print('Num Lags: ', dftest[2])
print('Num Obs used for ADF Regression adn Critical Values Calculation: ', dftest[3])
print('Critical Values: ', dftest[4])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Models

# COMMAND ----------

# Prepare X and y training/testing data
slrp_simp = pd.DataFrame(select_prior_months(slrp_transformed, 'DateTime', 3, '2022-01-01')['Ports Available'])
X, y = df_to_X_y(slrp_simp)
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)


# COMMAND ----------

#### LSTM Model no features ####

# Build LSTM Model
modelnf = Sequential()

# soecify shape, number of information we have
modelnf.add(InputLayer((6, 1))) 

#this is most simple lstm
modelnf.add(LSTM(64))
modelnf.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelnf.add(Dense(1, 'linear'))
modelnf.summary()

# COMMAND ----------

cp2 = ModelCheckpoint('modelnf/', save_best_only=True)
    #savebestonly will delete checkpoints that are not used
modelnf.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


# COMMAND ----------

modelnf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp2])

# COMMAND ----------

# load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model('modelnf/')
model2

plot_predictions1(model2, X_test, y_test, slrp_filtered, 'UC Berkeley', train_end, 'DateTime', 0, len(y_test))

# COMMAND ----------



# COMMAND ----------

# Build LSTM Model
model = Sequential()

# soecify shape, number of information we have
model.add(InputLayer((6, 11))) 

#this is most simple lstm
model.add(LSTM(64))
model.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
model.add(Dense(1, 'linear'))
model.summary()

# COMMAND ----------

cp1 = ModelCheckpoint('model1/', save_best_only=True)
    #savebestonly will delete checkpoints that are not used
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# learnign rate, used to adjust by loss. if large moves faster

# COMMAND ----------

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp1])
#10 runs through data sets
# call back cpl, to determining if saving model, if bester than previous epoch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

def plot_predictions1(model, X, y, train_df, station, train_end, date_col, start=0, end=1000):
    '''
    function to plot actual, predictions and rounded predictions
    model: is trained lstm model
    X: features matrix
    y: label array
    train_df: data frame with all records
    station: filter for specific station
    train_end: index of end of training set
    date_col: column name as str
    start: index on where to start to show results
    end: index on where to end results want to show
    '''
    
    ## get time stamps to add to results plot
    train_df = train_df.reset_index()
    last_train_timestamp = train_df.loc[train_end + len(y), date_col]
    first_test_timestamp = last_train_timestamp + dt.timedelta(minutes = 10)
    all_test_timestamps = pd.date_range(start = first_test_timestamp, periods = end, freq = '10T')
    

    ### get predictions, vector of predictions with inputs and flatten
    predictions = model.predict(X).flatten()
    predictions_rounded = np.round(predictions)
    
    ### train results -- should we return this too?
    df = pd.DataFrame(data = {'Predictions': predictions, 
                              'Predictions (rounded)': predictions_rounded,
                              'Actuals': y,
                              'DateTime': all_test_timestamps})
    
    #### plot
    plt.subplots(figsize = (15,7))
    
    # plot a portion of the time series
    plt.plot(df['DateTime'][start:end], df['Predictions'][start:end], label = 'Predicted')
    plt.plot(df['DateTime'][start:end], df['Predictions (rounded)'][start:end], label= 'Predicted (rounded)')
    plt.plot(df['DateTime'][start:end], df['Actuals'][start:end], label = 'Actual')
    
    plt.xlabel('DateTime', fontsize = 16);
    plt.ylabel('Number of Available Stations', fontsize = 16);
    plt.legend(fontsize = 14);
    plt.title('Charging Station Availability for ' + station + 
              '\n MSE for raw predictions:\t' + str(mse(y, predictions)) +
              '\n MSE for rounded predictions:\t' + str(mse(y, predictions_rounded)), 
              fontsize = 18);

    #  return df, mse(y, predictions)
    print('MSE for raw predictions:\t', mse(y, predictions))
    print('MSE for rounded predictions:\t', mse(y, predictions_rounded))

# COMMAND ----------

# load model in the event the last epoch is not the best model, if it is will give same results if skip
model1 = load_model('model1/')
model1

plot_predictions1(model1, X_test, y_test, slrp_filtered, 'UC Berkeley', train_end, 'DateTime', 0, len(y_test))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Prophet

# COMMAND ----------

slrp_prophet = slrp_filtered[['Ports Available']].reset_index().rename(columns = {'DateTime': 'ds',
                                                                                  'Ports Available': 'y'})

# COMMAND ----------

m = Prophet()
m.fit(slrp_prophet[slrp_prophet['ds'] < dt.datetime(2022, 4, 1)])
# 34 days between April 1, 2022 and May 5, 2022 (end of SlrpEV dataset)
future = m.make_future_dataframe(periods = 34*24*6, freq = '10min')
forecast = m.predict(future)

# COMMAND ----------

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][(forecast['ds'] < dt.datetime(2022, 5, 5)) & 
                                                     (forecast['ds'] >= dt.datetime(2022, 4, 1))].head()

# COMMAND ----------

preds = forecast[(forecast['ds'] < dt.datetime(2022, 4, 17)) & (forecast['ds'] >= dt.datetime(2022, 4, 1))]
actuals = slrp_prophet[(slrp_prophet['ds'] < dt.datetime(2022, 4, 17)) & 
                        (slrp_prophet['ds'] >= dt.datetime(2022, 4, 1))]
# preds['rounded'] = [0] * len(preds['yhat'])
# preds.loc[preds['rounded'] > 0, 'rounded'] = 8
preds['rounded'] = np.around(preds['yhat']).clip(upper = 8)
plt.subplots(figsize = (15,7))
plt.plot(preds['ds'], preds['yhat'], label = 'Predicted')
plt.plot(preds['ds'], preds['rounded'], label = 'Predicted (rounded)')
plt.plot(actuals['ds'], actuals['y'], label = 'Actual')
plt.xlabel('DateTime', fontsize = 16);
plt.ylabel('Number of Available Stations', fontsize = 16);
plt.legend(fontsize = 14);
plt.title('Charging Station Availability for UC Berkeley', fontsize = 18);
#     return df, mse(y, predictions)
print('MSE for raw predictions:\t', mse(actuals['y'], preds['yhat']))
print('MSE for rounded predictions:\t', mse(actuals['y'], preds['rounded']))

# COMMAND ----------

# MAGIC %md
# MAGIC #Palo Alto

# COMMAND ----------

# MAGIC %md
# MAGIC ## ARIMA

# COMMAND ----------

stations = pa_transformed['Station_Location'].unique()

# COMMAND ----------

pa_transformed.columns

# COMMAND ----------

df = pa_transformed[pa_transformed['Station_Location'] == stations[0]][['datetime_pac', 'Ports_Available']]
print(df['datetime_pac'].max().date())
                    
#start = dt.datetime(2020, 12, 1) interesting to see in small short frame have less autocorrelation
start = dt.datetime(2020, 1, 1)
end = dt.datetime(2020,6,1)
date_col = 'datetime_pac'
pa_train = arima_filter(df, start, end, date_col)
pa_test = arima_filter(df, end, dt.datetime(2021,1,2), date_col)

arima_eda(pa_train, 'Ports_Available', 25, pa_train.shape[0]-1)

# Check seasonality here per second graph, daily or weekly
#Autocorrelation takes into account a lot of indirect effects

# COMMAND ----------

arima_eda(df, 'Ports_Available', 25, 78000)

# COMMAND ----------

## if we run auto_arima on entire set, and not just train... is this a dataleak?
auto_arima(pa_train['Ports_Available'], trace = True, suppress_warnings = True)

# COMMAND ----------

pa_train.set_index('datetime_pac', inplace = True)

# COMMAND ----------

pa_test.set_index('datetime_pac', inplace = True)

# COMMAND ----------

pa_test.head()

# COMMAND ----------


# split data before this
# parameters based on data/EDA/PACF/ACF graphs
model = ARIMA(pa_train['Ports_Available'], order = (1, 0, 5))
model = model.fit()
#model.summary()

# COMMAND ----------

pred = model.predict(start = len(pa_train), end = len(pa_train) + len(pa_test)-1, typ='levels')


plt.subplots(figsize = (15, 7))
plt.plot(pred)
plt.plot(pa_test['Ports_Available'])
plt.plot(np.around(pred))
plt.title('ARIMA: Predicting Availability, ' + str(stations[0]))

print('Mean of DataSet: ', pa_test['Ports_Available'].mean())
print('MSE:\t', mse(pa_test, pred))
print('RMSE:\t', math.sqrt(mse(pa_test, pred)))
print('MSE (rounded):', mse(pa_test, np.around(pred)))
print('RMSE (rounded):', math.sqrt(mse(pa_test, np.around(pred))))


# COMMAND ----------

dftest = adfuller(df['Ports_Available'])
print(dftest)

print('ADF: ', dftest[0])
print('P-Value: ', dftest[1])
print('Num Lags: ', dftest[2])
print('Num Obs used for ADF Regression adn Critical Values Calculation: ', dftest[3])
print('Critical Values: ', dftest[4])

# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM

# COMMAND ----------

## need to rework this

def plot_predictions2(model, X, y, train_df, station, train_end, date_col, start=0, end=1000, figtitle_ = ''):
    '''
    function to plot actual, predictions and rounded predictions
    model: is trained lstm model
    X: features matrix
    y: label array
    train_df: data frame with all records
    station: filter for specific station
    train_end: index of end of training set
    date_col: column name as str
    start: index on where to start to show results
    end: index on where to end results want to show
    '''
    
    ## get time stamps to add to results plot
    train_df = train_df.reset_index()
    last_train_timestamp = train_df.loc[train_end + len(y), date_col]
    first_test_timestamp = last_train_timestamp + dt.timedelta(minutes = 10)
    all_test_timestamps = pd.date_range(start = first_test_timestamp, periods = end, freq = '10T')
    

    ### get predictions, vector of predictions with inputs and flatten
    predictions = model.predict(X).flatten()
    predictions_rounded = np.round(predictions)
    
    ### train results -- should we return this too?
    df = pd.DataFrame(data = {'Predictions': predictions, 
                              'Predictions (rounded)': predictions_rounded,
                              'Actuals': y,
                              'DateTime': all_test_timestamps})
    
    #### plot
    f, ax = plt.subplots(figsize = (15,7))
    
    # plot a portion of the time series
    ax.plot(df['DateTime'][start:end], df['Predictions'][start:end], label = 'Predicted')
    ax.plot(df['DateTime'][start:end], df['Predictions (rounded)'][start:end], label= 'Predicted (rounded)')
    ax.plot(df['DateTime'][start:end], df['Actuals'][start:end], label = 'Actual')
    
    ax.set_xlabel('DateTime', fontsize = 16);
    ax.set_ylabel('Number of Available Stations', fontsize = 16);
    ax.legend(fontsize = 14);
    ax.set_title(str(figtitle_) + '\n MSE for raw predictions:\t'+str( mse(y, predictions)) + '\nMSE for rounded predictions:\t' + str(mse(y, predictions_rounded)) , 
                 fontsize = 18);

    #  return df, mse(y, predictions)
    print('MSE for raw predictions:\t', mse(y, predictions))
    print('MSE for rounded predictions:\t', mse(y, predictions_rounded))
    return ax

# COMMAND ----------

stations = pa_transformed['Station_Location'].unique()

rows = 9
cols = 1

f, ax = plt.subplots(rows, cols, figsize = (15,7))

for i, station in enumerate(stations):
    pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
    X, y = df_to_X_y(pa)
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    

    print()
    print(station)
    
    
    #### LSTM Model no features ####
    # Build LSTM Model
    modelpa = Sequential()
    
    # specify shape, number of information we have
    modelpa.add(InputLayer((6, 1))) 
    
    #this is most simple lstm
    modelpa.add(LSTM(64))
    modelpa.add(Dense(8, 'relu'))
    # predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
    modelpa.add(Dense(1, 'linear'))
    modelpa.summary()
    
    modelname = str('model' + station + "/")
    cp3 = ModelCheckpoint(modelname, save_best_only=True)
    #savebestonly will delete checkpoints that are not used
    
    modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
    #load model in the event the last epoch is not the best model, if it is will give same results if skip
    model2 = load_model(modelname)
    model2
    
    ax[i] = plot_predictions2(model2, X_test, y_test, pa_transformed, station, train_end, 'datetime_pac', 0, len(y_test), 'Charging availability '+ str(station))
    

# COMMAND ----------

station = stations[0]


pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[1]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[2]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[3]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[4]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[5]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[7]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

station = stations[8]

pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]).set_index('datetime_pac')
    
X, y = df_to_X_y(pa)
    
X_train, y_train, X_val, y_val, X_test, y_test, train_end = split_train_test(X, y)
    
    
#### LSTM Model no features ####
# Build LSTM Model
modelpa = Sequential()
    
# specify shape, number of information we have
modelpa.add(InputLayer((6, 1))) 
    
#this is most simple lstm
modelpa.add(LSTM(64))
modelpa.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelpa.add(Dense(1, 'linear'))
modelpa.summary()
    
modelname = str('model' + station + "/")
cp3 = ModelCheckpoint(modelname, save_best_only=True)
#savebestonly will delete checkpoints that are not used
    
modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
#load model in the event the last epoch is not the best model, if it is will give same results if skip
model2 = load_model(modelname)
model2


plot_predictions1(model2, X_test, y_test, pa_transformed[pa_transformed['Station_Location'] == station], station, train_end, 'datetime_pac', 0, len(y_test))

# COMMAND ----------

pa_transformed.columns

# COMMAND ----------

station = stations[6]
station

# #cambrige which is closest to this station in terms of distance
model2 = load_model(str('model' + 'CAMBRIDGE' + "/"))
model2

df = pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']]

# convert station into format for X_text and y_test
X, y = df_to_X_y(df.set_index('datetime_pac') , window_size=6)

# get time stamps to add to results plot

all_test_timestamps = df['datetime_pac'].drop(df.tail(6).index)
    

### get predictions, vector of predictions with inputs and flatten
predictions = model2.predict(X).flatten()
predictions_rounded = np.round(predictions)
    
### train results -- should we return this too?
df = pd.DataFrame(data = {'Predictions': predictions, 
                          'Predictions (rounded)': predictions_rounded,
                          'Actuals': y,
                          'DateTime': all_test_timestamps})
    
#### plot
plt.subplots(figsize = (15,7))
    
# plot a portion of the time series
plt.plot(df['DateTime'], df['Predictions'], label = 'Predicted')
plt.plot(df['DateTime'], df['Predictions (rounded)'], label= 'Predicted (rounded)')
plt.plot(df['DateTime'], df['Actuals'], label = 'Actual')
    
plt.xlabel('DateTime', fontsize = 16);
plt.ylabel('Number of Available Stations', fontsize = 16);
plt.legend(fontsize = 14);
plt.title('Charging Station Availability for ' + station + 
           '\n MSE for raw predictions:\t' + str(mse(y, predictions)) +
           '\n MSE for rounded predictions:\t' + str(mse(y, predictions_rounded)), 
           fontsize = 18);

#  return df, mse(y, predictions)
print('MSE for raw predictions:\t', mse(y, predictions))
print('MSE for rounded predictions:\t', mse(y, predictions_rounded))