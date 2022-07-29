# Databricks notebook source
# MAGIC %md
# MAGIC # Modeling
# MAGIC 
# MAGIC ARIMA, Prophet, and LSTM

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
import folium
import holidays
from calendar import monthrange
import altair as alt

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

from keras.models import Model, Sequential

# Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

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
boulder = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_ts_clean")
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

### look at seasonality
df =slrp
df.set_index('DateTime')

results = seasonal_decompose(df['Ports Occupied'], period = 6*24*7) #period = 1 week, 6*24*7
#results = seasonal_decompose(df['Ports Occupied'], period = 6*24) #period = period = 1 day 6*24

fig = results.plot(observed=True, seasonal=True, trend=True, resid=True)
fig.set_size_inches((20,10))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Palo Alto

# COMMAND ----------

### Palo Alto ####
pa_transformed = palo_alto[['Station_Location', 'datetime_pac', 'Ports_Available']].copy()
pa_transformed

# COMMAND ----------

def seas_plot(df, columnname, periodvalue):
    results=seasonal_decompose(df[str(columnname)], period = periodvalue)
    fig = results.plot(observed=True, seasonal=True, trend=True, resid=True)
    fig.set_size_inches((20,10))
    
    return fig

# COMMAND ----------

### look at seasonality
df = pa_transformed[pa_transformed['Station_Location'] == 'BRYANT']
df.set_index('datetime_pac')


#def seasonality_plot_filters():
#    '''Creates a plot with a drop down menu to view seasonal decompose'''


seas_plot(df, 'Ports_Available', 6*24*7)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

#boulder[(boulder['Station']=='5333 Valmont Rd') & (boulder['Ports Available'] != 4)].sort_values(by=['Date Time']).head()

date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder
result = test.groupby('Station').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

# COMMAND ----------

# Sort by Date Time
boulder = boulder.sort_values(by = ['Date Time', 'Station'])
boulder.head()

# COMMAND ----------

boulder['proper_name'].unique()

# COMMAND ----------

### Boulder ###

boulder

df = boulder[boulder['proper_name'] == '900 Walnut St']
df.set_index('Date Time')

seas_plot(df, 'Ports Available', 6*24*7)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Parameters
# MAGIC 
# MAGIC Predictor:
# MAGIC - Number of stations available
# MAGIC 
# MAGIC Model Types:
# MAGIC - ARIMA Models
# MAGIC - FB Prophet
# MAGIC - LSTM Models
# MAGIC - CNN Models

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare For Models

# COMMAND ----------

def select_prior_months(df, date_col, train_months, start_date = '2021-01-01'):
    """
    Filter the dataframe to only a certain number of months, as defined by num_months
    
    At the end, set the index equal to the date_col column
    
    date_col is the column of the dataframe with the datetime info
    
    start_date should have format of YYYY-MM-DD
    """
    
    # converting start date to date format
    split = start_date.split('-')
    start_year = int(split[0])
    start_month = int(split[1])
    start_day = int(split[2])
    start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
    
    #total months to pull if train_months represents 70%
    total_months = train_months / 0.7
    
    end_date = start_date + dt.timedelta(days = int(total_months * 30))
    
#     end_day = start_day + int(((train_months / 0.7) - int(train_months/0.7))*30)
#     end_month = start_month + total_months if start_month < (12-total_months+1) else total_months - (12 - start_month)
#     end_year = start_year if start_month < (12-train_months+1) else start_year + 1 
#     end_date = dt.datetime(year = end_year, month = end_month, day = end_day)
    
    print(start_date)
    print(end_date)
    
    
    # filter df to dates from date_col equal to or after start and before end date
    temp = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
    
    return temp

# COMMAND ----------

def split_train_test(X, y, train_prop = 0.7, dev_prop = 0.15):
    '''
    Split inputs and labels into train, validation and test sets for LSTMs
    Returns x and y arrays for train, validation, and test.
    '''
    
    # get size of Array
    num_timestamps = X.shape[0]
    
    
    # define split proportions - can consider making this input to functions
    train_proportion = float(train_prop) # consider input
    dev_proportion = float(dev_prop) # consider input
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
    
    ##can we include dates in this??? by pulling the index??
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[0]

# COMMAND ----------

def split2_TrainTest(df,  train_prop = 0.7):
    ''' take a given df and return 2 df. One for training, and one for testing'''
    
    # get proportions
    train_prop = float(train_prop)
    test_prop = float(1 - train_prop)
    
    
    ### split dataframe ####
    num_timestamps = df.shape[0]
    
    # define split points
    train_start = 0
    train_end = int(num_timestamps * train_prop)
    
    # splitting
    traindf = df[train_start:train_end]
    testdf = df[train_end:]
    
    print(traindf.shape, testdf.shape)
    
    return traindf, testdf 


# COMMAND ----------

## plot model dataframe
def plot_predsActuals(df, predict_col, roundp_col, output_col, date_col, station, subtitle_ = '', fig_size = (15, 7) ):
    """
    Given a df. Identify the prediction column, the rounded prediction column, the actual column, the station name, 
    a subtitle, and fig size, and create a plot with all info.
    """
    
    # plot actuals and predictions
    plt.subplots(figsize = fig_size)
    plt.plot(df[date_col], df[predict_col], label = 'Predicted')
    plt.plot(df[date_col], df[roundp_col], label = 'Predicted (rounded)')
    plt.plot(df[date_col], df[output_col], label = 'Actuals')
    
    # add nice labels
    plt.xlabel('DateTime', fontsize = 16);
    plt.ylabel('Number of Available Stations', fontsize = 16);
    plt.legend(fontsize = 14);
    plt.title('Charging Station Availability for ' + str(station) + str('\n') + str(subtitle_), fontsize = 18);
    

# COMMAND ----------

# MAGIC %md
# MAGIC # ARIMA Models

# COMMAND ----------

def arima_filter(df, start_date, end_date, date_col):
    ''' 
    filter data frame to be between the start date and end date, not including the end date. 
    date_col is the date column used to filter the dataframe
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] # should we reset the index

# COMMAND ----------

def arima_eda(df, target_var, lags_pacf, lags_acf, m_ = 'ywm'):
    ''' 
    plots pacf and acf to understand how many lags
    '''
    plot_pacf(df[target_var], lags = lags_pacf, method = m_ )
    plot_acf(df[target_var], lags = lags_acf)
    p_val = adfuller(df[target_var])[1]
    print('p value of adfuller test:\t', p_val)
    if p_val <= 0.05:
        print('time series is stationary')
    else:
        print('nonstationary time series')

# COMMAND ----------

## ARIMA Model
def run_arima(traindf, testdf, actualcol, date_col, station):
    '''
    run arima model given a training df, a testing df. 
    as a string provide the name of the actualcol aka output column, the date column, 
    and the station name
    '''
    print(station)
    #### new
    traindf.set_index(date_col, drop=False, inplace=True)
    testdf.set_index(date_col, drop = False, inplace = True)
    
    ### get model parameters
    values_p = auto_arima(traindf[actualcol], d = 0, trace = True, suppress_warnings = True)
    print(values_p)
    p_order = values_p.get_params().get("order")
    print('order complete for ', station)
    
    
    ## fit model
    # parameters based on autoarima
    model = ARIMA(traindf[actualcol], order = p_order)
    print('model created')
    model = model.fit()
    print('model fit')
    # model.summary()
    
    ### get predictions
    pred = model.predict(start = traindf.shape[0], end = traindf.shape[0] + testdf.shape[0] - 1, typ='levels')

    ### getting actual data from previous data
    testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'}, inplace = True)
    
    ## createdf to output
    testdf['predictions'] = pred.values
    testdf['predictions (rounded)'] = np.around(pred).values
    
    ## Evaluation Metrics ###
    MSE_raw = mse(testdf['Actuals'], testdf['predictions'])
    MSE_rounded = mse(testdf['Actuals'], testdf['predictions (rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict({'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded}) 
                 })
    
    print(Evals)
    
    return model, testdf, Evals

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing Functions

# COMMAND ----------

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
slrp_arima = arima_filter(slrp_transformed, start, end, date_col)

arima_eda(slrp_arima, 'Ports Available', 25, slrp_arima.shape[0]-1)

# COMMAND ----------

train_prop = 0.7
df = slrp_arima
    
# get proportions
train_prop = float(train_prop)
print('train_prop: ' ,train_prop)
test_prop = float(1 - train_prop)
print('test_prop: ', test_prop)

### split dataframe ####
num_timestamps = df.shape[0]
print('dfshape ', df.shape)
print('number of time stamps: ', num_timestamps)

# define split points
train_start = 0
train_end = int(num_timestamps * train_prop)
print('trainend: ', train_end)

# splitting
traindf = df[train_start:train_end]
testdf = df[train_end:]

print('Train shape: ', traindf.shape, 'Test shape: ', testdf.shape)
traindf.head()

# COMMAND ----------

testdf.head()

# COMMAND ----------

actualcol= 'Ports Available'

params_ = auto_arima(traindf[actualcol], d = 0, trace = True, suppress_warnings = True)
print(params_)
p_order = params_.get_params().get("order")

#https://stats.stackexchange.com/questions/178577/how-to-read-p-d-and-q-of-auto-arima

# ## fit model
# # parameters based on data/EDA/PACF/ACF graphs
model = ARIMA(traindf[actualcol], order = p_order)
model = model.fit()
model.summary()

# COMMAND ----------

# ### get predictions

station = 'Slrp'

pred = model.predict(start = traindf.shape[0], end = traindf.shape[0] + testdf.shape[0] - 1, typ='levels')
print('pred shape: ', pred.shape)
pred
# ### geting actual data from previous data
testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'}, inplace = True)
testdf.head()

# # ## createdf to output
testdf['predictions'] = pred.values
#testdf = testdf.assign(predictions=pd.Series(pred).values)
testdf['predictions (rounded)'] = np.around(pred).values

# # ## Evaluation Metrics ###
MSE_raw = mse(testdf['Actuals'], testdf['predictions'])
MSE_rounded = mse(testdf['Actuals'], testdf['predictions (rounded)'])
RMSE_raw = math.sqrt(MSE_raw)
RMSE_rounded = math.sqrt(MSE_rounded)

Evals = dict({station: 
             dict({'MSE_Raw': MSE_raw,
                  'MSE_round': MSE_rounded,
                  'RMSE_Raw': RMSE_raw,
                  'RMSE': RMSE_rounded}) 
             })

print(Evals)
testdf.head()

# model, testdf, Evals


# COMMAND ----------

Evals#['Slrp']['MSE_Raw']

# COMMAND ----------

testdf.shape

# COMMAND ----------

df = testdf
predict_col = 'predictions'
roundp_col = 'predictions (rounded)'
output_col = 'Actuals'
date_col = 'DateTime'
station = 'Slrp'
subtitle_ = 'MSE Predictions: ' + str(Evals['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(Evals['Slrp']['MSE_round'])
fig_size = (15, 7)


# plot actuals and predictions
plt.subplots(figsize = fig_size)
plt.plot(df[date_col], df[predict_col], label = 'Predicted')
plt.plot(df[date_col], df[roundp_col], label = 'Predicted (rounded)')
plt.plot(df[date_col], df[output_col], label = 'Actuals')

plt.xlabel('DateTime', fontsize = 16);
plt.ylabel('Number of Available Stations', fontsize = 16);
plt.legend(fontsize = 14);
plt.title('Charging Station Availability for ' + str(station) + str('\n') + str(subtitle_), fontsize = 18);


# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley

# COMMAND ----------

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'

## filter data set
slrp_arima = arima_filter(slrp_transformed, start, end, date_col)

## run EDA
arima_eda(slrp_arima, 'Ports Available', 25, slrp_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(slrp_arima, 0.7)

#run model
berk_model, b_testdf, bevals = run_arima(traindf, testdf, actualcol, date_col, station)


info = 'MSE Predictions: ' + str(bevals['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(bevals['Slrp']['MSE_round'])
size_ = (15, 7)

# plot
plot_predsActuals(b_testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# COMMAND ----------

b_testdf.head()

# COMMAND ----------

### write dataframe to s3
# create spark data frame
results_4 = spark.createDataFrame(b_testdf) 

cols = results_4.columns
for col in cols:
    results_4 = results_4.withColumnRenamed(col, col.replace(" ", "_"))
display(results_4)

## Write to AWS S3
(results_4
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/BerkeleySlrp_ARIMA"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

date_col = 'datetime_pac'
actualcol= 'Ports_Available'

test = pa_transformed
result = test.groupby('Station_Location').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

# COMMAND ----------

# stations = pa_transformed['Station_Location'].unique()
# results_df = pd.DataFrame()
# metrics = dict()
# loc_ = 'PaloAlto'
# date_col = 'datetime_pac'
# actualcol= 'Ports_Available'


# for station in stations:
    
#     # iterate for each station
#     stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]
    
#     # filter to last 3 months of dataset
#     end = stationdf[date_col].max()
#     start_date = end - dt.timedelta(days = int(3*30))
#     start  = start_date.replace(hour=0, minute=0, second = 0)
#     ## filter data set
#     pa_arima = arima_filter(stationdf, start, end, date_col)

#     ## run EDA
#     arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

#     ## split into train and test
#     traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#     #run model
#     model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

#     ### plot
#     #subtitle
#     info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
#     size_ = (15, 7)
#     plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)
    
#     # capture metrics
#     metrics.update(evals)
    
#     # add additional dataframe columns for visualizations
#     testdf['Location'] = loc_
#     testdf['SiteName'] = station
    
#     # append each station df to results df
#     results_df = results_df.append(testdf)

# print(metrics)
# results_df.head()

# COMMAND ----------

results_df = pd.DataFrame()
metrics = dict()
loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

# COMMAND ----------

#station = 'BRYANT' #stationary, The computed initial AR coefficients are not stationary
station = 'CAMBRIDGE'
# station ='HAMILTON'
# station ='HIGH'
# station ='MPL'
# station ='RINCONADA_LIB'
# station ='SHERMAN'
# station ='TED_THOMPSON'
# station ='WEBSTER'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='HAMILTON'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='HIGH'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='MPL'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='SHERMAN'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
print(actualcol)
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='TED_THOMPSON'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
print(actualcol)
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station ='WEBSTER'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
print(actualcol)
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

stationdf = pa_transformed[pa_transformed['Station_Location'] == 'RINCONADA_LIB'][[date_col, actualcol]]
stationdf.reset_index(drop = True, inplace = True)

end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())


df = pa_arima 
target_var =  actualcol 
lags_pacf = 25 
lags_acf = pa_arima.shape[0]-1



plot_pacf(df[target_var], lags = lags_pacf, method = 'ols') ### why does this work ols, ols-inefficient, ols-adjusted, idadjusted and idbiased
plot_acf(df[target_var], lags = lags_acf)
p_val = adfuller(df[target_var])[1]
print('p value of adfuller test:\t', p_val)
if p_val <= 0.05:
    print('time series is stationary')
else:
    print('nonstationary time series')

# COMMAND ----------

station ='RINCONADA_LIB'

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]
stationdf

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
#pa_arima.reset_index(drop = True, inplace = True)

print(pa_arima.head())
print(actualcol)
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1, 'ols') 
#method = 'ols') ### why does this work ols, ols-inefficient, ols-adjusted, idadjusted and idbiased

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

station = 'BRYANT' #stationary, The computed initial AR coefficients are not stationary

# iterate for each station
stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
pa_arima = arima_filter(stationdf, start, end, date_col)
print(pa_arima.head())
## run EDA
arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(pa_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)
results_df.head()

# COMMAND ----------

### write results to s3 bucket

# create spark data frame
results_pa = spark.createDataFrame(results_df) 

cols = results_pa.columns
for col in cols:
    results_pa = results_pa.withColumnRenamed(col, col.replace(" ", "_"))
display(results_pa)

## Write to AWS S3
(results_pa
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/PaloAlto_ARIMA_Tail3months"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

boulder['proper_name'].unique()

# COMMAND ----------

date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder
result = test.groupby('proper_name').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

# COMMAND ----------

# stations = boulder['proper_name'].unique()
# results_df = pd.DataFrame()
# metrics = dict()
# loc_ = 'Boulder'
# date_col = 'Date Time'
# actualcol= 'Ports Available'


# for station in stations:
    
#     # iterate for each station
#     stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]
    
#     # filter to last 3 months of dataset
#     end = stationdf[date_col].max()
#     start_date = end - dt.timedelta(days = int(3*30))
#     start  = start_date.replace(hour=0, minute=0, second = 0)
#     ## filter data set
#     b_arima = arima_filter(stationdf, start, end, date_col)

#     ## run EDA
#     arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

#     ## split into train and test
#     traindf, testdf = split2_TrainTest(b_arima, 0.7)

#     #run model
#     model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

#     ### plot
#     #subtitle
#     info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
#     size_ = (15, 7)
#     plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)
    
#     # capture metrics
#     metrics.update(evals)
    
#     # add additional dataframe columns for visualizations
#     testdf['Location'] = loc_
#     testdf['SiteName'] = station
    
#     # append each station df to results df
#     results_df = results_df.append(testdf)

# print(metrics)
# results_df.head()

# COMMAND ----------

station ='1739 Broadway' #nonstationary time series,  ValueError: The computed initial AR coefficients are not stationary



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='1745 14th street'
# station ='1770 13th St' 
# station ='2052 Junction Pl'
# station ='2667 Broadway' 
# station ='3172 Broadway' 
# station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='1770 13th St' #stationary, ValueError: The computed initial AR coefficients are not stationary
# station ='2052 Junction Pl'
# station ='2667 Broadway' 
# station ='3172 Broadway' 
# station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='2052 Junction Pl'
# station ='2667 Broadway' 
# station ='3172 Broadway' 
# station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='2667 Broadway' 
# station ='3172 Broadway' 
# station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='3172 Broadway' 
# station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='3335 Airport Rd'
# station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='5333 Valmont Rd' 
# station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='5565 51st St'
# station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='5660 Sioux Dr'
# station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='600 Baseline Rd'
# station ='900 Walnut St



# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

station ='900 Walnut St'

# iterate for each station
stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# filter to last 3 months of dataset
end = stationdf[date_col].max()
start_date = end - dt.timedelta(days = int(3*30))
start  = start_date.replace(hour=0, minute=0, second = 0)
## filter data set
b_arima = arima_filter(stationdf, start, end, date_col)

## run EDA
arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(b_arima, 0.7)

#run model
model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

### plot
#subtitle
info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
size_ = (15, 7)
plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# capture metrics
metrics.update(evals)

# add additional dataframe columns for visualizations
testdf['Location'] = loc_
testdf['SiteName'] = station

# append each station df to results df
results_df = results_df.append(testdf)

print(metrics)

# COMMAND ----------

# write results to s3 bucket
## write dataframe to s3

# create spark data frame
results_ab = spark.createDataFrame(results_df) 

cols = results_ab.columns
for col in cols:
    results_ab = results_ab.withColumnRenamed(col, col.replace(" ", "_"))
display(results_ab)

## Write to AWS S3
(results_ab
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/Boulder_ARIMA_Tail3months"))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Prophet

# COMMAND ----------

### run prophet function
## given traindf, testdf, date_col, Outputcol, 

## adjust df to needed col names
# rename date col to ds
# rename actuals to y

## create model
# fit  model with train data

## make predictions
# create future dataframe, use test data shape for periods, and freq hard code to 10m
# make predictions


## output model, dataframe with dt, actuals, predictions, and predictions rounded, metrics dict

def run_prophet(traindf, testdf, date_col, output_col):
    if date_col != 'ds':
        traindf = traindf.rename(columns={date_col: 'ds'})
    if output_col != 'y':
        traindf = traindf.rename(columns={output_col: "y"})
    
    print(traindf.columns)
    # create model
    m = Prophet()
    m.fit(traindf)
    
    # make predictions
    future = m.make_future_dataframe(periods = testdf.shape[0], freq = '10min')
    forecast = m.predict(future)
    
#     forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][(forecast['ds'] <= testdf[date_col].max()) & 
#                                                      (forecast['ds'] >= testdf[date_col].min())].head()
    
    
    preds = forecast[(forecast['ds'] <= testdf[date_col].max()) & (forecast['ds'] >= testdf[date_col].min())]
    # rounding predictions
    ## need to change how we are rounding if there is more than 1 station being predicted for
    ## this assumes that there is at least one point in time when all stations are available (probably a fair assumption)
    preds['rounded'] = np.around(preds['yhat']).clip(upper = traindf['y'].max())
    preds = preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'rounded']]
    
    #create dataframe to output
    testdf = testdf.rename(columns = {output_col: "Actuals", date_col: 'DateTime'})
    testdf = testdf[['DateTime', 'Actuals']].merge(preds, left_on = 'DateTime', right_on = 'ds')
    
    pred_col = 'yhat'
    
    ## Evaluation Metrics ###
    MSE_raw = mse(testdf['Actuals'], testdf[pred_col])
    MSE_rounded = mse(testdf['Actuals'], testdf['rounded'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)
    
#     Evals = dict({station: 
#                  dict({'MSE_raw': MSE_raw,
#                       'MSE_round': MSE_rounded,
#                       'RMSE_raw': RMSE_raw,
#                       'RMSE_round': RMSE_rounded}) 
#                  })

    Evals = dict({'MSE_raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_raw': RMSE_raw,
                      'RMSE_round': RMSE_rounded})
    
    print(Evals)
    
    return m, testdf, Evals
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

## TESTING THE PROPHET FUNCTION
slrp_prophet_train = arima_filter(slrp_transformed, '01-01-2022', '04-01-2022', 'DateTime')
slrp_prophet_test = arima_filter(slrp_transformed, '04-01-2022', '04-15-2022', 'DateTime')
prophet_model, prophet_testdf, prophet_Evals = run_prophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available')

# COMMAND ----------

prophet_testdf.head()

# COMMAND ----------

# write results to s3 bucket
## write dataframe to s3

# create spark data frame
results_pslrp = spark.createDataFrame(prophet_testdf) 

cols = results_pslrp.columns
for col in cols:
    results_pslrp = results_pslrp.withColumnRenamed(col, col.replace(" ", "_"))
display(results_pslrp)

## Write to AWS S3
(results_pslrp
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/Slrp_prophet"))

# COMMAND ----------

plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

boulder['Station'].unique()

# COMMAND ----------

def run_prophet_boulder(df, dt_col, station_col, output_col, loc, train_start, train_end, test_start, test_end):
    prophet_train = arima_filter(df, train_start, train_end, dt_col)
    prophet_test = arima_filter(df, test_start, test_end, dt_col)
    
    results_df = pd.DataFrame()
    
    for station in prophet_train[station_col].unique():
        station_train = prophet_train[prophet_train[station_col] == station]
        station_test = prophet_test[prophet_test[station_col] == station]
        
        prophet_model, prophet_testdf, prophet_Evals = run_prophet(station_train, station_test, dt_col, output_col)
        
        plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', station + ' (' + loc + ')')
        print('Completed 1 for loop')
        
        # add additional dataframe columns for visualizations
        prophet_testdf['Location'] = loc
        prophet_testdf['SiteName'] = station
        # append each station df to results df
        results_df = results_df.append(prophet_testdf)
    
        
    return results_df

# COMMAND ----------

boulderprophet = run_prophet_boulder(boulder, 'Date Time', 'Station', 'Ports Available', 'Boulder', 
                    '01-01-2022', '04-01-2022', '04-01-2022', '04-15-2022' )

# COMMAND ----------

# write results to s3 bucket

# create spark data frame
results_pbould = spark.createDataFrame(boulderprophet) 

cols = results_pbould.columns
for col in cols:
    results_pbould = results_pbould.withColumnRenamed(col, col.replace(" ", "_"))
display(results_pbould)

## Write to AWS S3
(results_pbould
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/boulder_prophet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

paloaltoprophet = run_prophet_boulder(pa_transformed, 'datetime_pac', 'Station_Location', 'Ports_Available', 'Palo Alto', 
                    '01-01-2019', '04-01-2019', '04-01-2019', '04-15-2019')

# COMMAND ----------

# write results to s3 bucket

# create spark data frame
results_ppaloalto = spark.createDataFrame(paloaltoprophet) 

cols = results_ppaloalto.columns
for col in cols:
    results_ppaloalto = results_ppaloalto.withColumnRenamed(col, col.replace(" ", "_"))
display(results_ppaloalto)

## Write to AWS S3
(results_ppaloalto
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/paloAlto_prophet"))

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prior Work Code

# COMMAND ----------

# import math
# import time
# import pandas as pd
# import numpy  as np 
# from matplotlib import pyplot
# from tensorflow import keras
# from numpy import array 
# from keras.models import Sequential 
# from keras.layers import LSTM,GRU,ConvLSTM2D
# from keras.layers import RepeatVector
# from keras.layers import Dense,Dropout,Flatten,TimeDistributed
# from keras.layers import BatchNormalization 
# from keras.layers import Bidirectional
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D 
# from keras.models import Model 
# from keras.layers import Input

# COMMAND ----------

# # split a multivariate sequence into samples
# def split_sequences(sequences, n_steps_in, n_steps_out):
# 	X, y = list(), list()
# 	for i in range(len(sequences)):
# 		# find the end of this pattern
# 		end_ix = i + n_steps_in
# 		out_end_ix = end_ix + n_steps_out-1 ### why do they minus 1
# 		# check if we are beyond the dataset
# 		if out_end_ix > len(sequences):
# 			break
# 		# gather input and output parts of the pattern
# 		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
# 		X.append(seq_x)
# 		y.append(seq_y)
# 	return array(X), array(y)

# COMMAND ----------

# def read_data(string, model_id, t_win, n_steps_in, n_steps_out, n_features):
    
#     Z = pd.read_csv(string)
#     Z=Z.to_numpy()
     
#     Z.shape 
#     if model_id=='LSTM' or model_id=='GRU' or model_id=='BiLSTM' or model_id=='StackedLSTM' or model_id=='Conv1D' or  model_id=='Decoder':
#         X, y = split_sequences(Z, n_steps_in, n_steps_out ) 
#
#     elif model_id=='CNN_LSTM' or model_id=='CNNDecoder': 
#         X, y = split_sequences(Z, t_win, n_steps_out )
#         X = X.reshape((X.shape[0], n_steps_in, n_steps_out, n_features))
#     elif model_id=='ConvLSTM':
#         X, y = split_sequences(Z, t_win, n_steps_out )
#         X = X.reshape((X.shape[0], n_steps_in, 1, n_steps_out, n_features))

#     n_train=int(0.7*len(X)) 
    
#     X_train=X[0: n_train,];         y_train = y[0:n_train,]
#     X_test =X[n_train: len(X),];    y_test  = y[n_train:len(X),]
#     X_train.shape
#     X_test.shape 
#     X_train[0,]
    
#     return X_train,y_train,X_test,y_test

# COMMAND ----------

# #########
# # LSTM
# ##########
# def fit_model_LSTM(res ,_iter, X_train,y_train,X_test,y_test,t_win,
#                    n_steps_in,n_steps_out,n_features, nf_1, nf_2, ker_size, 
#               po_size,n_n_lstm,dropout,n_epoch,bat_size):
#    
     #YC - Create model   
#     model = Sequential() 
#     model.add(LSTM(n_n_lstm,   input_shape=(n_steps_in, n_features)))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dropout(dropout)) 
#     model.add(Dense(n_steps_out, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     keras.utils.plot_model(model, show_shapes=True)
#     model.fit(X_train, y_train, epochs=n_epoch, batch_size=bat_size,verbose=2)
     
    # YC - predict
#     temp = model.predict(X_test, verbose=2)
#    #YC get shape of prediction outputs, and target number of steps out
#     m,n=temp.shape #YC - m = (datasize?)
#     t_target = n_steps_out 
       
    # create a an array of predictions of 0s, and actuals
#     yhat=np.zeros((m,t_target))
#     y_obs=np.array(y_test[0:m,0:t_target])
#     scores1= np.zeros(m) 
    
    # creating threshold of 0.5
#     for i in np.arange(m):  
#         for j in np.arange(t_target):  
#             if temp[i][j]>=0.5:
#                 yhat[i][j]=1 # predidcting available           
#         val=1 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target
#         scores1[i]=val       
#     _mean1 = np.mean(scores1)     
#     res[_iter,:]=[nf_1, nf_2, ker_size, po_size, n_n_lstm,dropout,n_epoch,
#                   bat_size, _mean1 ]  
#     return res 

# COMMAND ----------

# def run(model_id, n_steps_in,n_steps_out,n_features,n_epoch,n_trivals,n_out,
#             nf_1,nf_2,ker_size,po_size,n_n_lstm,dropout,bat_size): 
    
#     t_win=n_steps_in*n_steps_out        
#     n_station=9
#     string='U:/DL/data_chg_all_feature_'   
#     station=[string+'1.csv',string+'2.csv',string+'3.csv',string+'4.csv',string+'5.csv'
#              ,string+'6.csv',string+'7.csv',string+'8.csv',string+'9.csv']
  
#     res_all=[]
#     for s in range(n_station):         
#         X_train,y_train,X_test,y_test =read_data(station[s],model_id, t_win, 
#                                 n_steps_in,n_steps_out,n_features)
#         res=np.zeros([n_trivals,n_out])
#         X_train.shape
#         y_train.shape 
#         for _iter in range(n_trivals):   
#             if model_id=='LSTM':
#                 res=fit_model_LSTM(res,_iter, X_train,y_train,X_test,y_test ,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size) 
#
#             elif model_id=='BiLSTM':
#                 res=fit_model_BiLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size)     
#             elif model_id=='StackedLSTM':                    
#                 res=fit_model_StackedLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size)  
#             elif model_id=='Conv1D':                    
#                 res=fit_model_Conv1D(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size) 
#             elif model_id=='CNN_LSTM':
#                  res=fit_model_CNN_LSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size)   
#             elif model_id=='GRU':
#                  res=fit_model_GRU(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size) 
#             elif model_id=='ConvLSTM':
#                  res=fit_model_ConvLSTM(res,_iter, X_train,y_train,X_test,y_test,t_win, 
#                            n_steps_in,n_steps_out,n_features, nf_1, nf_2, 
#                            ker_size, po_size,n_n_lstm,dropout,n_epoch,
#                            bat_size) 
               
#         _mean = np.mean(res[:,-1:],axis=0)
#         _std  = np.std(res[:,-1:],axis=0)
#         res_all.append([_mean,_std])
        
#     temp=[]
#     for i in range(n_station):            
#         temp.append(res_all[i][0])
        
#     accuracy_avg1=np.mean(temp, axis=0)
#     accuracy_avg2=np.mean(temp, axis=1)  
#     return accuracy_avg1, accuracy_avg2, res_all

# COMMAND ----------

# def main():
   
#     n_steps_in =3  # t-3,t-2,t-1
#     n_features = 148 # 148 for all features
#     n_steps_out =6# num of predicted steps, if n_steps_out =1 or 3 po_size needs to be 1
#     n_epoch_global=15
#     n_trivals=10
#     n_out=9 
#     nf_1=32
#     nf_2=16
#     ker_size=4
#     po_size=2
#     n_n_lstm=16
#     dropout=0.4
#     bat_size= 30
#     accuracy_avg_1=[]
#     accuracy_avg_2=[]
#     flag_sensitivity=0
#     model_id='LSTM'
#     #model_id='GRU'
#     # model_id='BiLSTM'
#     #model_id='StackedLSTM'
#     #model_id='Conv1D'   
#     #model_id='CNN_LSTM' 
#    # model_id='ConvLSTM'
#     if  model_id=='ConvLSTM' or  model_id=='Conv1D'  :
#         ker_size=1       

#     if flag_sensitivity==1: #sensitivity analysis
#         parameter = [11,12,13,14,15,16,17,18,19,20] #,13,14,15,16,17,18,19,20
#         for i in range(len(parameter)):
#             avg1,avg2,res_all =  run(model_id,n_steps_in,n_steps_out,n_features,
#                            parameter[i],n_trivals,n_out,nf_1,nf_2,ker_size,
#                            po_size,n_n_lstm,dropout,bat_size)  
#             accuracy_avg_1.append(avg1)
#             accuracy_avg_2.append(avg2)            
#     else:   avg1,avg2,res_all =  run(model_id,n_steps_in,n_steps_out,n_features,
#                             n_epoch_global,n_trivals,n_out,nf_1,nf_2,ker_size,
#                             po_size,n_n_lstm,dropout,bat_size)   
#     accuracy_avg_1.append(avg1)
#     accuracy_avg_2.append(avg2)
    
#     ## output results    
#     print('model: ', model_id)
#     print('sensitivity_flag = ', flag_sensitivity) 
#     if flag_sensitivity==1:
#         print('parameter : ', parameter) 
#     print('n_step out: ', n_steps_out)
#     print('n_epoch,n_trivals, nf_1,nf_2,po_size,n_n_lstm,dropout,bat_size', 
#           n_epoch_global,n_trivals, nf_1,nf_2,po_size,n_n_lstm,dropout,bat_size)       
#     print('accuracy_avg_1: ',accuracy_avg_1)
#     print('accuracy_avg_2: ',accuracy_avg_2)
#     return res_all
    
# res_all = main()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC #NEW Functions - LSTM (mulivariate or univariate, one-step or multi-step)

# COMMAND ----------

def select_months(df, date_col, train_months, start_date = '2021-01-01'):
    """
    Filter the dataframe to only a certain number of months, after start date
    
    date_col is the column of the dataframe with the datetime info
    
    start_date should have format of YYYY-MM-DD
    
    returns the filtered dataframe
    """
    
    # converting start date to date time format
    split = start_date.split('-')
    start_year = int(split[0])
    start_month = int(split[1])
    start_day = int(split[2])
    start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
    
    #total months to pull if train_months represents 70%
    total_months = train_months / 0.7
    
    end_date = start_date + dt.timedelta(days = int(total_months * 30))
    
    
    print(start_date)
    print(end_date)
    
    
    # filter df to dates from date_col equal to or after start and before end date
    temp = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
    
    return temp

# COMMAND ----------

def dfSplit_Xy(df, date_col = 'DateTime', n_input=6, n_out = 1):
    """ 
    Tranform pandas dataframe into arrays of inputs and outputs. 
    The output (value predicting) must be located at the end of the df
    n_inputs, is the the number of inputs, to use to predict n_input+1 (or i + n_input) aka window size
    n_outs, is the number of steps out you want to predict. Default is 1
    Returns 3 numpy arrays. Inputs (features), and actual labels, and an array of dates for actual labels
    """
    
    ind_dates = df[date_col].to_numpy() #### dates
    df_as_np = df.set_index(date_col).to_numpy()
    
    #X is matrix of inputs
    #y is actual outputs, labels
    X = []
    y = []
    y_dates = [] #### for plotting later
    
    
    for i in range(len(df_as_np)): 
        #print(i)
        start_out = i + n_input
        start_end = start_out + n_out

        # to make sure we always have n_out values in label array
        if start_end > len(df_as_np):
            break

        #take the i and the next values, makes a list of list for multiple inputs
        row = df_as_np[i:start_out, :]
        X.append(row)

        # Creating label outputs extended n_out steps. -1, last column is label
        label = df_as_np[start_out:start_end, -1]
        y.append(label)
        
        # array of dates
        label_dates = ind_dates[start_out:start_end]####
        y_dates.append(label_dates) #### 
        
    
    return np.array(X), np.array(y), np.array(y_dates)

# COMMAND ----------

def split_train_test(X, y, y_dates, train_prop = 0.7, dev_prop = 0.15):
    '''
    Split inputs and labels into train, validation and test sets for LSTMs
    Returns x and y arrays for train, validation, and test.
    '''
    
    # get size of Array
    num_timestamps = X.shape[0]
    
    
    # define split proportions - can consider making this input to functions
    train_proportion = float(train_prop) # consider input
    dev_proportion = float(dev_prop) # consider input
    test_proportion = (1 - train_proportion - dev_proportion) # can leave this as is
    
    
    # define split points
    train_start = 0 # do we need this? can we always start at 0?
    train_end = int(num_timestamps * train_proportion)
    dev_end = int(num_timestamps * (train_proportion + dev_proportion))
    
    
    # splitting
    X_train, y_train = X[train_start:train_end], y[train_start:train_end]#, y_dates[train_start:train_end]
    X_val, y_val = X[train_end:dev_end], y[train_end:dev_end]#, y_dates[train_end:dev_end]
    # include dates for plotting later
    X_test, y_test, yd_test = X[dev_end:], y[dev_end:], y_dates[dev_end:]
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, X_train.shape[0], yd_test.shape)
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[0], yd_test

# COMMAND ----------

def run_lstm(n_inputs, n_features, n_outputs, X_train, y_train, X_val, y_val, n_epochs = 10):
    '''Run lstm model, and get fitted model'''
    
    # Build LSTM Model
    model = Sequential()
    model.add(InputLayer((n_inputs, n_features))) 
    model.add(LSTM(64, input_shape = (n_inputs, n_features)))
    model.add(Dense(8, 'relu'))
    model.add(Dense(n_outputs, 'linear'))
    model.summary()
    
    ### checkpoints to save best model
    cp = ModelCheckpoint('model/', save_best_only=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    
    ## Fit model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, callbacks=[cp])
    
    # load model in the event the last epoch is not the best model, if it is will give same results if skip
    model = load_model('model/')
    
    return model

# COMMAND ----------

def plot_predictions(model, X, y, train_df, station, train_end, date_col,y_dates, start=0, end=1000):
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
    
#     #train_df = temp
#     #start=0
#     #end=np.prod(y_test.shape)
#     #print(y_test.shape)

#     last_train_timestamp = train_df.loc[train_end + len(y), date_col]
#     first_test_timestamp = last_train_timestamp + dt.timedelta(minutes = 10)
#     all_test_timestamps = pd.date_range(start = first_test_timestamp, periods = end, freq = '10T')
    ### need to fix dates
    all_test_timestamps = y_dates.flatten()

    # print(last_train_timestamp)
    # print(first_test_timestamp)
    # print(all_test_timestamps)


    ### get predictions, vector of predictions with inputs and flatten
    predictions = model.predict(X_test).flatten()
    predictions_rounded = np.round(predictions)

#     print(predictions.shape)
    # should return dataframe
    df = pd.DataFrame(data = {'Predictions': predictions, 
                              'Predictions (rounded)': predictions_rounded,
                              'Actuals': y_test.flatten(),
                              'DateTime': all_test_timestamps})
    
    
    ### evaluation metrics
    MSE_raw = mse(y_test.flatten(), predictions)
    MSE_rounded = mse(y_test.flatten(), predictions_rounded)
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    
    #### this entire section can be separate, and do not need start and end, can filter outputed df beforehand if needed smaller dataset
    
    #### plot 
    plt.subplots(figsize = (15,7))
    # plot a portion of the time series
    plt.plot(df['DateTime'][start:end], df['Predictions'][start:end], label = 'Predicted')
    plt.plot(df['DateTime'][start:end], df['Predictions (rounded)'][start:end], label= 'Predicted (rounded)')
    plt.plot(df['DateTime'][start:end], df['Actuals'][start:end], label = 'Actual')

    plt.xlabel('DateTime', fontsize = 16);
    plt.ylabel('Number of Available Stations', fontsize = 16);
    plt.legend(fontsize = 14);
    plt.title('Charging Station Availability for ' + station, fontsize = 18);

    return df, Evals


# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing to Create Functions

# COMMAND ----------

slrp_transformed[[c for c in slrp_transformed if c not in ['Ports Available']] + ['Ports Available']].head()

# COMMAND ----------

slrp_transformed[['DateTime','Ports Available']].head()

# COMMAND ----------

## filter inputs
start_date = '2022-01-01'
train_months = 3
#df = slrp_transformed[[c for c in slrp_transformed if c not in ['Ports Available']] + ['Ports Available']]
df = slrp_transformed[['DateTime','Ports Available']]
date_col = 'DateTime'

###

### Filter
split = start_date.split('-')
start_year = int(split[0])
start_month = int(split[1])
start_day = int(split[2])
start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
print('Start Date: ', start_date)

total_months = train_months / 0.7
print(total_months)


end_date = start_date + dt.timedelta(days = int(total_months * 30))
print(end_date)

temp = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
temp.head()


# COMMAND ----------

df = temp
test = df.set_index(date_col)
print(test.index.to_series()[0:8])
print(test.index.to_series().min())

# COMMAND ----------

# sequence split

# inputs
df = temp

ind_dates = df[date_col].to_numpy() ####

df_as_np = df.set_index(date_col).to_numpy()
n_input = 6
n_out = 2

X = []
y = []
y_dates = [] ####

## function elements
for i in range(len(df_as_np)): 
    print(i)
    start_out = i + n_input
    start_end = start_out + n_out
    
    # to make sure we always have n_out values in label array
    if start_end > len(df_as_np):
        break
    
    #take the i and the next values, makes a list of list for multiple inputs
    row = df_as_np[i:start_out, :]
    print(row)
    X.append(row)
    
    # Creating label outputs extended n_out steps
    label = df_as_np[start_out:start_end, -1]
    print(label)
    y.append(label)
    
    label_dates = ind_dates[start_out:start_end]####
    print(label_dates)###
    y_dates.append(label_dates) ####
    
    
X = np.array(X)
y = np.array(y)
y_dates = np.array(y_dates) #####

# COMMAND ----------

# this runs good as is
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

# COMMAND ----------

#### LSTM Model no features ####
n_inputs = n_input
n_features = X_train.shape[2]
n_outputs  = n_out


# Build LSTM Model
modelnf = Sequential()

# soecify shape, number of information we have
modelnf.add(InputLayer((n_inputs, n_features))) 

#this is most simple lstm
modelnf.add(LSTM(64, input_shape = (n_inputs, n_features)))
modelnf.add(Dense(8, 'relu'))
# predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
modelnf.add(Dense(n_outputs, 'linear'))
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

# COMMAND ----------

print(y_test.shape)
np.prod(y_test.shape)

# COMMAND ----------

#X_train, y_train, X_val, y_val, X_test, y_test, train_end

# train_df = temp
start=0
end=np.prod(y_test.shape)
# print(y_test.shape)

# last_train_timestamp = train_df.loc[train_end + len(y), date_col]
# first_test_timestamp = last_train_timestamp + dt.timedelta(minutes = 10)
# all_test_timestamps = pd.date_range(start = first_test_timestamp, periods = end, freq = '10T')

# print(last_train_timestamp)
# print(first_test_timestamp)
# print(all_test_timestamps)

all_test_timestamps = y_dates.flatten()

### get predictions, vector of predictions with inputs and flatten
predictions = model2.predict(X_test).flatten()
predictions_rounded = np.round(predictions)

print(predictions.shape)

df = pd.DataFrame(data = {'Predictions': predictions, 
                          'Predictions (rounded)': predictions_rounded,
                          'Actuals': y_test.flatten(),
                          'DateTime': all_test_timestamps})

#### need to fix dates...




# COMMAND ----------

y_test.shape[0]

# COMMAND ----------

station = 'UC Berkeley'

plt.subplots(figsize = (15,7))
# plot a portion of the time series
plt.plot(df['DateTime'][start:end], df['Predictions'][start:end], label = 'Predicted')
plt.plot(df['DateTime'][start:end], df['Predictions (rounded)'][start:end], label= 'Predicted (rounded)')
plt.plot(df['DateTime'][start:end], df['Actuals'][start:end], label = 'Actual')
    
plt.xlabel('DateTime', fontsize = 16);
plt.ylabel('Number of Available Stations', fontsize = 16);
plt.legend(fontsize = 14);
plt.title('Charging Station Availability for ' + station, fontsize = 18);

MSE_raw = mse(y_test.flatten(), predictions)
MSE_rounded = mse(y_test.flatten(), predictions_rounded)
RMSE_raw = math.sqrt(MSE_raw)
RMSE_rounded = math.sqrt(MSE_rounded)



Evals = dict({station: 
             dict(
                 {'MSE_Raw': MSE_raw,
                  'MSE_round': MSE_rounded,
                  'RMSE_Raw': RMSE_raw,
                  'RMSE': RMSE_rounded
                 }) 
            })

#  return df, Evals
print(Eval)


### shoudl we also get accuracy...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test for Palo Alto

# COMMAND ----------

pa_transformed.head()

# COMMAND ----------

stepsout =  6
historyin = 6
n_features = 1

stations = pa_transformed['Station_Location'].unique()[0:2]
results_df = pd.DataFrame()
metrics = dict()



for station in stations:
    pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][['datetime_pac', 'Ports_Available']])
    if station != 'MPL':
        pa = pa.tail(129600)
    else:
        pa = pa
    
    X, y, y_dates = dfSplit_Xy(pa, 'datetime_pac', historyin, stepsout)
    
    X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)
    

    print()
    print(station)
    
    
    #### LSTM Model no features ####
    # Build LSTM Model
    modelpa = Sequential()
    
    # specify shape, number of information we have
    modelpa.add(InputLayer((historyin, n_features)))  ## this should be an input 
    
    #this is most simple lstm
    modelpa.add(LSTM(64, input_shape = (historyin, n_features))) # this should be input
    modelpa.add(Dense(8, 'relu'))
    # predicitng a linear value, is this right? the example is neg and pos, ours only 0 and pos
    modelpa.add(Dense(stepsout, 'linear'))
    modelpa.summary()
    
    
    
    modelname = str('model' + station + "/")
    cp3 = ModelCheckpoint(modelname, save_best_only=True)
    #savebestonly will delete checkpoints that are not used
    
    modelpa.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    modelpa.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp3])
    
    #load model in the event the last epoch is not the best model, if it is will give same results if skip
    model2 = load_model(modelname)
    model2
    
    df, evals = plot_predictions(model2, X_test, y_test, pa_transformed, station, train_end, 'datetime_pac', y_dates, 0, y_test.shape[0])
    metrics.update(evals)
    df['Station'] = station
    results_df = results_df.append(df)

print(metrics)
results_df.head()

# Next should write dataframe as csv to s3 bucket

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate one-step

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Multistep
# MAGIC 6 steps

# COMMAND ----------

## filter inputs
start_date = '2022-01-01'
train_months = 3
#df = slrp_transformed[[c for c in slrp_transformed if c not in ['Ports Available']] + ['Ports Available']]
#df = slrp_transformed[['DateTime','Ports Available']]
date_col = 'DateTime'
results_df = pd.DataFrame()
bk_metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'Berkeley'
station = 'slrpEV'

# filter to 3months
df = select_months(slrp_transformed, date_col, train_months, start_date)[['DateTime','Ports Available']]

# split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)
# split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

modelB = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

df, evals = plot_predictions(modelB, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
bk_metrics.update(evals)

# add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

# append each station df to results df
results_df = results_df.append(df)

print(bk_metrics)
results_df.head()

# COMMAND ----------

print(bk_metrics)

# COMMAND ----------

## write dataframe to s3

# create spark data frame
results_2 = spark.createDataFrame(results_df) 

cols = results_2.columns
for col in cols:
    results_2 = results_2.withColumnRenamed(col, col.replace(" ", "_"))
display(results_2)

## Write to AWS S3
(results_2
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/Berkeley_LSTM_MultiStepNoFeatures_3months"))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Multivariate Multistep

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

# MAGIC %md
# MAGIC ###Univariate Multistep 
# MAGIC 6 steps, 10 minutes to 1 hour

# COMMAND ----------

stations = pa_transformed['Station_Location'].unique()
results_df = pd.DataFrame()
metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'PaloAlto'



for station in stations:
    pa = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][
        ['datetime_pac', 'Ports_Available']]).sort_values(by=['datetime_pac'])
    if station != 'SHERMAN':  ## think this is an error, should be sherman??
        pa = pa.tail(129600) # pull last 3 months
    else:
        pa = pa # if sherman pull entire dataset, ~1month
    
    # split into array format
    X, y, y_dates = dfSplit_Xy(pa, 'datetime_pac', n_inputs, n_outputs)
    # split into train, val, and test
    X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)
    
    print()
    print(station)
    
    modelpa = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)
    
    df, evals = plot_predictions(modelpa, X_test, y_test, pa_transformed, station, train_end, 'datetime_pac', y_dates, 0, y_test.shape[0])
    
    # capture metrics
    metrics.update(evals)
    
    # add additional dataframe columns for visualizations
    df['Location'] = loc_
    df['SiteName'] = station
    df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])
    
    # append each station df to results df
    results_df = results_df.append(df)

print(metrics)
results_df.head()

# Next should write dataframe as csv to s3 bucket

# COMMAND ----------

metrics

# COMMAND ----------

## write dataframe to s3

# create spark data frame
results_1 = spark.createDataFrame(results_df) 

cols = results_1.columns
for col in cols:
    results_1 = results_1.withColumnRenamed(col, col.replace(" ", "_"))
display(results_1)

## Write to AWS S3
(results_1
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/PaloAlto_LSTM_MultiStepNoFeatures_Tail129600"))


#https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python 
#https://towardsdatascience.com/reading-and-writing-files-from-to-amazon-s3-with-pandas-ccaf90bfe86c

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder CO

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Multistep
# MAGIC 6 steps

# COMMAND ----------

boulder.head()

# COMMAND ----------

boulder_tf = boulder.copy()
boulder_tf.head()


# COMMAND ----------

stations = boulder_tf['proper_name'].unique()

date_col = 'Date Time'
results_df = pd.DataFrame()
b_metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'Boulder'



for station in stations:
    b = pd.DataFrame( boulder_tf[boulder_tf['proper_name'] == station][
        ['Date Time', 'Ports Available']]).sort_values(by=['Date Time'])
    
    b = b.tail(129600) #### this was missing a 0! last 3 months of the data , 10*6*24
    
    # split into array format
    X, y, y_dates = dfSplit_Xy(b, date_col, n_inputs, n_outputs)
    # split into train, val, and test
    X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)
    
    print()
    print(station)
    
    modelpa = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)
    
    df, evals = plot_predictions(modelpa, X_test, y_test, b, station, train_end, date_col, y_dates, 0, y_test.shape[0])
    
    # capture metrics
    b_metrics.update(evals)
    
    # add additional dataframe columns for visualizations
    df['Location'] = loc_
    df['SiteName'] = station
    df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])
    
    # append each station df to results df
    results_df = results_df.append(df)

print(b_metrics)
results_df.head()

# Next should write dataframe as csv to s3 bucket

# COMMAND ----------

boulder.head()

# COMMAND ----------

b_metrics

# COMMAND ----------

## write dataframe to s3

# create spark data frame
results_3 = spark.createDataFrame(results_df) 

cols = results_3.columns
for col in cols:
    results_3 = results_3.withColumnRenamed(col, col.replace(" ", "_"))
display(results_3)

## Write to AWS S3
(results_3
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/Boulder_LSTM_MultiStepNoFeatures_Tail129600"))



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Results Visual

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Results

# COMMAND ----------

## read in lstm results to calculate one step MSE/RMSE 
boulder_res = spark.read.parquet(f"/mnt/{mount_name}/data/Boulder_LSTM_MultiStepNoFeatures_Tail129600/")
boulder_res = boulder_res.toPandas()

berkeley_res = spark.read.parquet(f"/mnt/{mount_name}/data/Berkeley_LSTM_MultiStepNoFeatures_3months/")
berkeley_res = berkeley_res.toPandas()

paloalto_res = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_LSTM_MultiStepNoFeatures_Tail129600/")
paloalto_res = paloalto_res.toPandas()

# COMMAND ----------

## read in prophetresults to calculate one step MSE/RMSE 
boulderph_res = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_prophet/")
boulderph_res = boulderph_res.toPandas()

paloaltoph_res = spark.read.parquet(f"/mnt/{mount_name}/data/paloAlto_prophet/")
paloaltoph_res = paloaltoph_res.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arima

# COMMAND ----------

arima_slrp = {'Berkeley': 
              {'Slrp': 
              {'MSE_Raw': 1.3430667625018382, 'MSE_round': 1.4353182751540041, 
                  'RMSE_Raw': 1.1589075728900204, 'RMSE': 1.1980476931883823}}}


## arima Palo ALto
arima_pa = {'Palo Alto': 
            {'CAMBRIDGE': 
            {
                'MSE_Raw': 1.032623265576834, 'MSE_round': 1.083885772565018, 
                'RMSE_Raw': 1.0161807248599208, 'RMSE': 1.0410983491318282}, 
            'HAMILTON': 
            {
                'MSE_Raw': 0.14388850599598707, 'MSE_round': 0.0, 
                'RMSE_Raw': 0.3793263845239177,'RMSE': 0.0}, 
            'HIGH': 
            {
                'MSE_Raw': 0.9776234770620584, 'MSE_round': 1.138194798572157, 
                'RMSE_Raw': 0.9887484397267378, 'RMSE': 1.0668621272555123
            }, 
            'MPL': 
            {
                'MSE_Raw': 0.628790081147563, 'MSE_round': 0.8296787353391127, 
                'RMSE_Raw': 0.7929628497903057, 'RMSE': 0.9108670239607496
            }, 
            'SHERMAN': 
            {
                'MSE_Raw': 0.27357851593931004, 'MSE_round': 0.3232016210739615, 
                'RMSE_Raw': 0.5230473362319228, 'RMSE': 0.568508241869862}, 
            'TED_THOMPSON': 
            {
                'MSE_Raw': 0.42436444509175103, 'MSE_round': 0.8268740438551759, 
                'RMSE_Raw': 0.6514326097853492, 'RMSE': 0.9093261482302023}, 
            'WEBSTER': 
            {
                'MSE_Raw': 0.8527277228908278, 'MSE_round': 0.9739928607853137, 
                'RMSE_Raw': 0.9234325762560187, 'RMSE': 0.9869107663742015}, 
            'RINCONADA_LIB': 
            {
                'MSE_Raw': 2.1476738144642046e-28, 'MSE_round': 0.0, 
                'RMSE_Raw': 1.4654943925052066e-14, 'RMSE': 0.0
            }}}


arima_bo = {'Boulder': {'1100 Spruce St': 
            {'MSE_Raw': 0.36543427956471486, 'MSE_round': 0.4824763366589921, 
              'RMSE_Raw': 0.6045116041605114, 'RMSE': 0.6946051660180711}, 
            '1100 Walnut': 
            {
                'MSE_Raw': 0.25336291866524463, 'MSE_round': 0.3192632386799693, 
                'RMSE_Raw': 0.5033516848737517, 'RMSE': 0.5650338385264809}, 
            '1360 Gillaspie Dr': 
            {
                'MSE_Raw': 0.22226549876837567, 'MSE_round': 0.27884369403939624, 
                'RMSE_Raw': 0.47145042026535056, 'RMSE': 0.5280565254207131}, 
            '1400 Walnut St': 
            {
                'MSE_Raw': 0.07357209938479074, 'MSE_round': 0.07981580966999233, 
                'RMSE_Raw': 0.2712417729347579, 'RMSE': 0.28251691926324046}, 
            '1500 Pearl St': 
            {
                'MSE_Raw': 0.3562032612017224, 'MSE_round': 0.5207880260969682, 
                'RMSE_Raw': 0.5968276645747267, 'RMSE': 0.7216564460302203}, 
            '1505 30th St': 
            {
                'MSE_Raw': 1.3264388940351561, 'MSE_round': 1.308007162957278, 
                'RMSE_Raw': 1.1517112893582124, 'RMSE': 1.1436814079791968}, 
            '1745 14th street': 
            {
                'MSE_Raw': 0.3085052288335017, 
                'MSE_round': 0.48324379636735737, 
                'RMSE_Raw': 0.5554324700929013, 
                'RMSE': 0.695157389637309}, 
            '2052 Junction Pl': 
            {
                'MSE_Raw': 0.11566671407451307, 'MSE_round': 0.1286774111025838, 
                'RMSE_Raw': 0.34009809478224523, 'RMSE': 0.3587163379365147}, 
            '2667 Broadway': 
            {
                'MSE_Raw': 0.033889935874492474, 'MSE_round': 0.0332565873624968, 
                'RMSE_Raw': 0.18409219395317247, 'RMSE': 0.1823638872213926}, 
            '3172 Broadway': 
            {
                'MSE_Raw': 0.555043263980237, 'MSE_round': 0.8176004093118444, 
                'RMSE_Raw': 0.7450122576040189, 'RMSE': 0.9042125907726813}, 
            '3335 Airport Rd': 
            {
                'MSE_Raw': 0.17429547140499302, 'MSE_round': 0.1951905858275774, 
                'RMSE_Raw': 0.41748709130342343, 'RMSE': 0.4418037865699856}, 
            '5333 Valmont Rd': 
            {
                'MSE_Raw': 0.49902037561451096, 'MSE_round': 0.6336658992069583, 
                'RMSE_Raw': 0.7064137425153272, 'RMSE': 0.7960313431058844}, 
            '5565 51st St': 
            {
                'MSE_Raw': 0.043095752905294944, 'MSE_round': 0.043745203376822715, 
                'RMSE_Raw': 0.20759516590059351, 'RMSE': 0.20915354019672416}, 
             '5660 Sioux Dr': {
                 'MSE_Raw': 0.14700462506719372, 'MSE_round': 0.16781785622921463, 
                 'RMSE_Raw': 0.3834118217624409, 'RMSE': 0.4096557777320059}, 
            '600 Baseline Rd': 
            {
                'MSE_Raw': 0.34236507316333825, 'MSE_round': 0.4223586595037094, 
                'RMSE_Raw': 0.5851197084044754, 'RMSE': 0.6498912674468779}, 
            '900 Walnut St': 
            {
                'MSE_Raw': 0.2831676110599813, 'MSE_round': 0.34586850856996676, 
                'RMSE_Raw': 0.5321349556832189, 'RMSE': 0.5881058651042061
            }}}


arima = dict()
arima.update(arima_slrp)
arima.update(arima_pa)
arima.update(arima_bo)


arimadf = pd.DataFrame.from_dict({(i,j): arima[i][j] 
                           for i in arima.keys() 
                           for j in arima[i].keys()},
                       orient='index')

arimadf.reset_index(inplace=True)

arimadf = arimadf.rename(columns={"level_0":"location", "level_1":"station", "RMSE":"RMSE_round"})
arimadf['Model'] ='ARIMA' 
arimadf

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM - multi

# COMMAND ----------

boulder_metrics = dict()
stations = boulder_res['SiteName'].unique()

for station in stations: 
    stationdf = boulder_res[boulder_res['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions_(rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    boulder_metrics.update(Evals)

print(boulder_metrics)

# COMMAND ----------

lstm_slrp = {'Berkeley':{'slrpEV': {'MSE_Raw': 0.19469725887572972, 'MSE_round': 0.20408163265306123, 'RMSE_Raw': 0.4412451233449835, 'RMSE': 0.45175395145262565}}}

lstm_palo_alto = {'Palo Alto':{'BRYANT': {'MSE_Raw': 0.22889939791343378, 'MSE_round': 0.21536601677041, 'RMSE_Raw': 0.47843431933070374, 'RMSE': 0.46407544297281017}, 'CAMBRIDGE': {'MSE_Raw': 0.47660792485938, 'MSE_round': 0.4920520602911672, 'RMSE_Raw': 0.6903679633785015, 'RMSE': 0.7014642259525194}, 'HAMILTON': {'MSE_Raw': 0.1535057001595553, 'MSE_round': 0.1544832553114872, 'RMSE_Raw': 0.39179803491027787, 'RMSE': 0.3930435794049907}, 'HIGH': {'MSE_Raw': 0.2585853508108167, 'MSE_round': 0.25411115112231425, 'RMSE_Raw': 0.5085128816567155, 'RMSE': 0.5040943871164548}, 'MPL': {'MSE_Raw': 0.1905598631235267, 'MSE_round': 0.1954265759867078, 'RMSE_Raw': 0.43653162900702475, 'RMSE': 0.4420707816478124}, 'RINCONADA_LIB': {'MSE_Raw': 0.003525780938370341, 'MSE_round': 0.0, 'RMSE_Raw': 0.05937828675846366, 'RMSE': 0.0}, 'SHERMAN': {'MSE_Raw': 475.7569677677619, 'MSE_round': 473.0128726287263, 'RMSE_Raw': 21.81185383610852, 'RMSE': 21.748859110967782}, 'TED_THOMPSON': {'MSE_Raw': 0.21568211465214024, 'MSE_round': 0.22024452561002794, 'RMSE_Raw': 0.46441588544336015, 'RMSE': 0.4693021687676586}, 'WEBSTER': {'MSE_Raw': 0.27830956379901284, 'MSE_round': 0.28410240581648577, 'RMSE_Raw': 0.5275505319862855, 'RMSE': 0.5330125756644826}}}

lstm_boulder = {'Boulder':{'1100 Walnut': {'MSE_Raw': 0.17494893591480404, 'MSE_round': 0.21787613655858637, 'RMSE_Raw': 0.41826897555855613, 'RMSE': 0.4667720391782121}, '1739 Broadway': {'MSE_Raw': 0.040585291020749696, 'MSE_round': 0.025819179962257677, 'RMSE_Raw': 0.20145791377046893, 'RMSE': 0.1606834775646136}, '3172 Broadway': {'MSE_Raw': 0.23929800137188761, 'MSE_round': 0.29336078229541945, 'RMSE_Raw': 0.48918094951856783, 'RMSE': 0.5416279002187936}, '5660 Sioux Dr': {'MSE_Raw': 0.110643421351928, 'MSE_round': 0.13621547435237605, 'RMSE_Raw': 0.33263105891051126, 'RMSE': 0.36907380610438345}, '5565 51st St': {'MSE_Raw': 0.00974294032030815, 'MSE_round': 0.012695144964831017, 'RMSE_Raw': 0.09870633373957392, 'RMSE': 0.11267273390146801}, '1505 30th St': {'MSE_Raw': 0.37030223056725914, 'MSE_round': 0.4275175844913364, 'RMSE_Raw': 0.6085246343142232, 'RMSE': 0.6538482885894376}, '3335 Airport Rd': {'MSE_Raw': 0.10199118015470376, 'MSE_round': 0.1182878709898782, 'RMSE_Raw': 0.31936058015150176, 'RMSE': 0.34393003792905064}, '600 Baseline Rd': {'MSE_Raw': 0.18497864597355698, 'MSE_round': 0.22525304511923142, 'RMSE_Raw': 0.4300914390842452, 'RMSE': 0.47460830704827683}, '2667 Broadway': {'MSE_Raw': 0.007905993510320375, 'MSE_round': 0.007720020586721565, 'RMSE_Raw': 0.08891565391043567, 'RMSE': 0.08786364769756355}, '5333 Valmont Rd': {'MSE_Raw': 5.169619033276567, 'MSE_round': 4.784182535597873, 'RMSE_Raw': 2.27367962415037, 'RMSE': 2.18727742538478}, '1745 14th street': {'MSE_Raw': 0.01586428088190246, 'MSE_round': 0.01604048721907703, 'RMSE_Raw': 0.12595348697794143, 'RMSE': 0.12665104507692398}, '1500 Pearl St': {'MSE_Raw': 0.06956762677206524, 'MSE_round': 0.07282552753474009, 'RMSE_Raw': 0.26375675682731853, 'RMSE': 0.26986205278760494}, '1770 13th St': {'MSE_Raw': 0.05647577274809522, 'MSE_round': 0.0680219591696689, 'RMSE_Raw': 0.237646318608337, 'RMSE': 0.2608101975952415}, '1360 Gillaspie Dr': {'MSE_Raw': 0.11738864561331476, 'MSE_round': 0.14187682278263852, 'RMSE_Raw': 0.34262026445222815, 'RMSE': 0.3766653989718707}, '900 Walnut St': {'MSE_Raw': 0.7058714121733698, 'MSE_round': 0.8537484988848859, 'RMSE_Raw': 0.8401615393323892, 'RMSE': 0.9239851183243624}, '1100 Spruce St': {'MSE_Raw': 0.2136945489251706, 'MSE_round': 0.23400240178418252, 'RMSE_Raw': 0.46227107731846107, 'RMSE': 0.48373794743040627}, '1400 Walnut St': {'MSE_Raw': 0.034565883876274184, 'MSE_round': 0.043146337279121635, 'RMSE_Raw': 0.18591902505196767, 'RMSE': 0.20771696435082435}, '2052 Junction Pl': {'MSE_Raw': 0.07606921945595811, 'MSE_round': 0.10550694801852804, 'RMSE_Raw': 0.27580648914765965, 'RMSE': 0.3248183307920414}}}

lstm_multi = dict()
lstm_multi.update(lstm_slrp)
lstm_multi.update(lstm_palo_alto)
lstm_multi.update(lstm_boulder)


lstmmultidf = pd.DataFrame.from_dict({(i,j): lstm_multi[i][j] 
                           for i in lstm_multi.keys() 
                           for j in lstm_multi[i].keys()},
                       orient='index')

lstmmultidf.reset_index(inplace=True)

lstmmultidf = lstmmultidf.rename(columns={"level_0":"location", "level_1":"station", "RMSE":"RMSE_round"})
lstmmultidf['Model'] ='LSTM Multi' 
lstmmultidf

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM - One

# COMMAND ----------

#### Boulder one step metrics ####
boulder_metrics1 = dict()
boulder1 = boulder_res[boulder_res['StepsOut'] == 1]
stations = boulder1['SiteName'].unique()


for station in stations: 
    stationdf = boulder1[boulder1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions_(rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    boulder_metrics1.update(Evals)

lstm_boulder1 = {'Boulder' : boulder_metrics1}


#### Palo Alto one step metrics ####
lstm_palo_alto1 = dict()
paloalto1 = paloalto_res[paloalto_res['StepsOut'] == 1]
stations = paloalto1['SiteName'].unique()


for station in stations: 
    stationdf = paloalto1[paloalto1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions_(rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    lstm_palo_alto1.update(Evals)

lstm_palo_alto1 = {'Palo Alto' : lstm_palo_alto1}


#### Berkeley One Step ####
lstm_berkeley1 = dict()
berkeley1 = berkeley_res[berkeley_res['StepsOut'] == 1]
stations = berkeley_res['SiteName'].unique()


for station in stations: 
    stationdf = berkeley1[berkeley1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions_(rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    lstm_berkeley1.update(Evals)

lstm_berkeley1 = {'Berkeley' : lstm_berkeley1}



#### Combine and Convert to df ####
lstm_one = dict()
lstm_one.update(lstm_berkeley1)
lstm_one.update(lstm_palo_alto1)
lstm_one.update(lstm_boulder1)


lstmonedf = pd.DataFrame.from_dict({(i,j): lstm_one[i][j] 
                           for i in lstm_one.keys() 
                           for j in lstm_one[i].keys()},
                       orient='index')

lstmonedf.reset_index(inplace=True)

lstmonedf = lstmonedf.rename(columns={"level_0":"location", "level_1":"station", "RMSE":"RMSE_round"})
lstmonedf['Model'] ='LSTM One' 
lstmonedf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prophet

# COMMAND ----------

prophet_slrp = {'Berkeley': {'Slrp': {'MSE_Raw': 0.8884726629601606, 'MSE_round': 0.8556547619047619, 'RMSE_Raw': 0.9425882786032089, 'RMSE': 0.9250160873761936}}}
                
                
#### Boulder one step metrics ####
boulder_metrics2 = dict()
stations = boulderph_res['SiteName'].unique()


for station in stations: 
    stationdf = boulderph_res[boulderph_res['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['yhat'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['rounded'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    boulder_metrics2.update(Evals)

prophet_boulder = {'Boulder' : boulder_metrics2}



#### Palo Alto one step metrics ####
prophet_paloalto = dict()
stations = paloaltoph_res['SiteName'].unique()


for station in stations: 
    stationdf = paloaltoph_res[paloaltoph_res['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['yhat'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['rounded'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict(
                     {'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded
                     }) 
                })
    
    prophet_paloalto.update(Evals)

prophet_paloalto = {'Palo Alto' : prophet_paloalto}



#### Combine and Convert to df ####
prophet_res = dict()
prophet_res.update(prophet_slrp)
prophet_res.update(prophet_paloalto)
prophet_res.update(prophet_boulder)
prophet_res

prophet_df = pd.DataFrame.from_dict({(i,j): prophet_res[i][j] 
                           for i in prophet_res.keys() 
                           for j in prophet_res[i].keys()},
                       orient='index')

prophet_df.reset_index(inplace=True)

prophet_df = prophet_df.rename(columns={"level_0":"location", "level_1":"station", "RMSE":"RMSE_round"})
prophet_df['Model'] ='Prophet' 
prophet_df

# COMMAND ----------

df = arimadf

alt.Chart(df).mark_circle(filled = True).encode(
    x=alt.X('MSE_Raw:Q', 
            axis=alt.Axis(title='MSE', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('MSE_round:Q', 
            axis=alt.Axis(title = ['MSE Rounded'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('location:O', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['MSE_Raw', 'MSE_round', 'location', 'station:O']
).properties(
    title=['ARIMA', 
           'Raw Predictions'],
    width=600,
    height=400
).configure_title(
    fontSize=20,
    color='black'
).configure_legend(
    strokeColor='gray',
    fillColor='#F9F9F9',
    padding=10,
    titleFontSize=12,
    cornerRadius=10,
    orient='top-left')

# COMMAND ----------

df = arimadf
df2 = prophet_df
df3 = lstmonedf[lstmonedf['station'] != 'SHERMAN']


arima = alt.Chart(df).mark_bar(filled = True).encode(
    x=alt.X('station:O', 
            axis=alt.Axis(title='Station', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('location:O', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['MSE_Raw', 'MSE_round', 'RMSE_Raw', 'RMSE_round', 'location', 'station:O']
).properties(
    title=['ARIMA', 
           'Rounded Predictions'],
    width=600,
    height=400)


prophet = alt.Chart(df2).mark_bar(filled = True).encode(
    x=alt.X('station:O', 
            axis=alt.Axis(title='Station', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('location:O', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['MSE_Raw', 'MSE_round', 'RMSE_Raw', 'RMSE_round', 'location', 'station:O']
).properties(
    title=['Prophet', 
           'Rounded Predictions'],
    width=600,
    height=400)



lstm = alt.Chart(df3).mark_bar(filled = True).encode(
    x=alt.X('station:O', 
            axis=alt.Axis(title='Station', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('location:O', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['MSE_Raw', 'MSE_round', 'RMSE_Raw', 'RMSE_round', 'location', 'station:O']
).properties(
    title=['LSTM', 
           'Rounded Predictions'],
    width=600,
    height=400)



alt.hconcat(
    arima,
    prophet,
    lstm,
    data=df
).resolve_scale(
    y='shared'
)

# COMMAND ----------

print(lstmmultidf[lstmmultidf['station'] == 'SHERMAN'])
df = lstmmultidf[lstmmultidf['station'] != 'SHERMAN']

alt.Chart(df).mark_bar(filled = True).encode(
    x=alt.X('station:O', 
            axis=alt.Axis(title='Stations', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round', 
            axis=alt.Axis(title = ['RMSE Rounded'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('location:O', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['MSE_Raw', 'MSE_round', 'location', 'station:O']
).properties(
    title=['LSTM Multistep: 6 steps out', 
           'Rounded Predictions'],
    width=800,
    height=400
).configure_title(
    fontSize=20,
    color='black'
).configure_legend(
    strokeColor='gray',
    fillColor='#F9F9F9',
    padding=10,
    titleFontSize=12,
    cornerRadius=10,
    orient='top-left')