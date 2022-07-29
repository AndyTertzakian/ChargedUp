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



import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)



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
# MAGIC ##Palo Alto

# COMMAND ----------

### Palo Alto ####
pa_transformed = palo_alto[['Station_Location', 'datetime_pac', 'Ports_Available']].sort_values(by = ['datetime_pac', 'Station_Location']).copy()
pa_transformed

# COMMAND ----------

date_col = 'datetime_pac'
actualcol= 'Ports_Available'

test = pa_transformed
result = test.groupby('Station_Location').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

# Sort by Date Time
boulder = boulder.sort_values(by = ['Date Time', 'Station'])
boulder.head()

# COMMAND ----------

#boulder[(boulder['Station']=='5333 Valmont Rd') & (boulder['Ports Available'] != 4)].sort_values(by=['Date Time']).head()

date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder
result = test.groupby('Station').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

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
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] 

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

# ### write dataframe to s3
# # create spark data frame
# results_4 = spark.createDataFrame(b_testdf) 

# cols = results_4.columns
# for col in cols:
#     results_4 = results_4.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_4)

# ## Write to AWS S3
# (results_4
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/BerkeleySlrp_ARIMA"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

results_df = pd.DataFrame()
metrics = dict()
loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

# COMMAND ----------

station = 'CAMBRIDGE'

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head()

# COMMAND ----------

print(palo_alto[(palo_alto['Station_Location'] == 'HAMILTON')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Ports_Available'].max())
print(palo_alto[(palo_alto['Station_Location'] == 'HAMILTON')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Ports_Available'].min())
print(palo_alto[(palo_alto['Station_Location'] == 'HAMILTON')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Total_Ports'].max())
print(palo_alto[(palo_alto['Station_Location'] == 'HAMILTON')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Total_Ports'].min())

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

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
results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

# COMMAND ----------

print(palo_alto[(palo_alto['Station_Location'] == 'RINCONADA_LIB')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Ports_Available'].max())
print(palo_alto[(palo_alto['Station_Location'] == 'RINCONADA_LIB')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Ports_Available'].min())
print(palo_alto[(palo_alto['Station_Location'] == 'RINCONADA_LIB')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Total_Ports'].max())
print(palo_alto[(palo_alto['Station_Location'] == 'RINCONADA_LIB')&(palo_alto['datetime_pac'] > dt.datetime(2020, 12,5))]['Total_Ports'].min())

# COMMAND ----------

# station = 'BRYANT' #stationary, The computed initial AR coefficients are not stationary

# # iterate for each station
# stationdf = pa_transformed[pa_transformed['Station_Location'] == station][[date_col, actualcol]]

# # filter to last 3 months of dataset
# end = stationdf[date_col].max()
# start_date = end - dt.timedelta(days = int(3*30))
# start  = start_date.replace(hour=0, minute=0, second = 0)
# ## filter data set
# pa_arima = arima_filter(stationdf, start, end, date_col)
# print(pa_arima.head())
# ## run EDA
# arima_eda(pa_arima, actualcol, 25, pa_arima.shape[0]-1)

# ## split into train and test
# traindf, testdf = split2_TrainTest(pa_arima, 0.7)

# #run model
# model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

# ### plot
# #subtitle
# info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
# size_ = (15, 7)
# plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# # capture metrics
# metrics.update(evals)

# # add additional dataframe columns for visualizations
# testdf['Location'] = loc_
# testdf['SiteName'] = station

# # append each station df to results df
# results_df = results_df.append(testdf)

# print(metrics)
# results_df.sort_values(by = ['DateTime', 'SiteName']).head(10)

# COMMAND ----------

# ### write results to s3 bucket

# # create spark data frame
# results_pa = spark.createDataFrame(results_df) 

# cols = results_pa.columns
# for col in cols:
#     results_pa = results_pa.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_pa)

# ## Write to AWS S3
# (results_pa
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/PaloAlto_ARIMA_Tail3months"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder
result = test.groupby('proper_name').agg({date_col: ['min', 'max']})
  
print("Dates for each stations")
print(result)

# COMMAND ----------

results_df = pd.DataFrame()
metrics = dict()
loc_ = 'Boulder'
date_col = 'Date Time'
actualcol= 'Ports Available'

# COMMAND ----------

# station ='1739 Broadway' #nonstationary time series,  ValueError: The computed initial AR coefficients are not stationary


# # iterate for each station
# stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# # filter to last 3 months of dataset
# end = stationdf[date_col].max()
# start_date = end - dt.timedelta(days = int(3*30))
# start  = start_date.replace(hour=0, minute=0, second = 0)
# ## filter data set
# b_arima = arima_filter(stationdf, start, end, date_col)

# ## run EDA
# arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

# ## split into train and test
# traindf, testdf = split2_TrainTest(b_arima, 0.7)

# #run model
# model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

# ### plot
# #subtitle
# info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
# size_ = (15, 7)
# plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# # capture metrics
# metrics.update(evals)

# # add additional dataframe columns for visualizations
# testdf['Location'] = loc_
# testdf['SiteName'] = station

# # append each station df to results df
# results_df = results_df.append(testdf)

# print(metrics)

# COMMAND ----------

station ='1745 14th street'


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

# station ='1770 13th St' #stationary, ValueError: The computed initial AR coefficients are not stationary


# # iterate for each station
# stationdf = boulder[boulder['proper_name'] == station][[date_col, actualcol]]

# # filter to last 3 months of dataset
# end = stationdf[date_col].max()
# start_date = end - dt.timedelta(days = int(3*30))
# start  = start_date.replace(hour=0, minute=0, second = 0)
# ## filter data set
# b_arima = arima_filter(stationdf, start, end, date_col)

# ## run EDA
# arima_eda(b_arima, actualcol, 25, b_arima.shape[0]-1)

# ## split into train and test
# traindf, testdf = split2_TrainTest(b_arima, 0.7)

# #run model
# model, testdf, evals = run_arima(traindf, testdf, actualcol, date_col, station)

# ### plot
# #subtitle
# info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])
# size_ = (15, 7)
# plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# # capture metrics
# metrics.update(evals)

# # add additional dataframe columns for visualizations
# testdf['Location'] = loc_
# testdf['SiteName'] = station

# # append each station df to results df
# results_df = results_df.append(testdf)

# print(metrics)

# COMMAND ----------

station ='2052 Junction Pl'


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

# # write results to s3 bucket
# ## write dataframe to s3

# # create spark data frame
# results_ab = spark.createDataFrame(results_df) 

# cols = results_ab.columns
# for col in cols:
#     results_ab = results_ab.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_ab)

# ## Write to AWS S3
# (results_ab
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/Boulder_ARIMA_Tail3months"))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Prophet

# COMMAND ----------

## output model, dataframe with dt, actuals, predictions, and predictions rounded, metrics dict

def run_prophet(traindf, testdf, date_col, output_col, station):
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
    
    Evals = dict({station: 
                 dict({'MSE_raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_raw': RMSE_raw,
                      'RMSE_round': RMSE_rounded}) 
                 })

#     Evals = dict({'MSE_raw': MSE_raw,
#                       'MSE_round': MSE_rounded,
#                       'RMSE_raw': RMSE_raw,
#                       'RMSE_round': RMSE_rounded})
    
    print(Evals)
    
    return m, testdf, Evals
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

## TESTING THE PROPHET FUNCTION

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'

## filter data set
slrp_p = arima_filter(slrp_transformed, start, end, date_col)

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(slrp_p, 0.7)

#slrp_prophet_train = arima_filter(slrp_transformed, '01-01-2022', '04-01-2022', 'DateTime')
#slrp_prophet_test = arima_filter(slrp_transformed, '04-01-2022', '04-15-2022', 'DateTime')
prophet_model, prophet_testdf, prophet_Evals = run_prophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station)

# COMMAND ----------

prophet_testdf.head()

# COMMAND ----------

# # write results to s3 bucket
# ## write dataframe to s3

# # create spark data frame
# results_pslrp = spark.createDataFrame(prophet_testdf) 

# cols = results_pslrp.columns
# for col in cols:
#     results_pslrp = results_pslrp.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_pslrp)

# ## Write to AWS S3
# (results_pslrp
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/Slrp_prophet"))

# COMMAND ----------

plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

boulder['Station'].unique()

# COMMAND ----------

def run_prophet_boulder(df, dt_col, station_col, output_col, loc):
    
    #collect df vales and metrics
    results_df = pd.DataFrame()
    metrics = dict()
    
    for station in df[station_col].unique():
        
        #get station df
        stationdf = df[df[station_col] == station][[dt_col, output_col]]
        
        # get filter dates
        end = stationdf[dt_col].max()
        start_date = end - dt.timedelta(days = int(3*30))
        start  = start_date.replace(hour=0, minute=0, second = 0)
        
        ## filter data set
        b_prophet = arima_filter(stationdf, start, end, dt_col)
        station_train, station_test = split2_TrainTest(b_prophet, 0.7)
        
        
        prophet_model, prophet_testdf, prophet_Evals = run_prophet(station_train, station_test, dt_col, output_col, station)
        
        plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', station + ' (' + loc + ')')
        print('Completed 1 for loop')
        
        # add additional dataframe columns for visualizations
        prophet_testdf['Location'] = loc
        prophet_testdf['SiteName'] = station
        
        # append each station df to results df
        results_df = results_df.append(prophet_testdf)
        
        # capture metrics
        metrics.update(prophet_Evals)
    
    print(metrics)
    
    return results_df, metrics

# COMMAND ----------

boulderprophet, boulder_prophet_res = run_prophet_boulder(boulder, 'Date Time', 'Station', 'Ports Available', 'Boulder')
print(boulder_prophet_res)

# COMMAND ----------

# # write results to s3 bucket

# # create spark data frame
# results_pbould = spark.createDataFrame(boulderprophet) 

# cols = results_pbould.columns
# for col in cols:
#     results_pbould = results_pbould.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_pbould)

# ## Write to AWS S3
# (results_pbould
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/boulder_prophet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

paloaltoprophet, paloalto_prophet_res  = run_prophet_boulder(pa_transformed, 'datetime_pac', 'Station_Location', 'Ports_Available', 'Palo Alto')
print(paloalto_prophet_res)

# COMMAND ----------

# # write results to s3 bucket

# # create spark data frame
# results_ppaloalto = spark.createDataFrame(paloaltoprophet) 

# cols = results_ppaloalto.columns
# for col in cols:
#     results_ppaloalto = results_ppaloalto.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_ppaloalto)

# ## Write to AWS S3
# (results_ppaloalto
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/paloAlto_prophet"))

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Models

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
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Multistep
# MAGIC 6 steps

# COMMAND ----------

## filter inputs

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'


lstmberk_results_df = pd.DataFrame()
bk_metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'Berkeley'
station = 'slrp'



## filter data set
df = arima_filter(slrp_transformed, start, end, date_col)[['DateTime','Ports Available']]

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
lstmberk_results_df = lstmberk_results_df.append(df)

print(bk_metrics)
lstmberk_results_df.head()

# COMMAND ----------

# ## write dataframe to s3

# # create spark data frame
# results_2 = spark.createDataFrame(lstmberk_results_df) 

# cols = results_2.columns
# for col in cols:
#     results_2 = results_2.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_2)

# ## Write to AWS S3
# (results_2
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/Berkeley_LSTM_MultiStepNoFeatures_3months"))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Multivariate Multistep

# COMMAND ----------

## filter inputs
start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'


lstmberk_results_df2 = pd.DataFrame()
bk_metrics2 = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'Berkeley'
station = 'slrp'



## filter data set
df = arima_filter(slrp_transformed[[c for c in slrp_transformed if c not in ['Ports Available']] + ['Ports Available']], start, end, date_col)

# split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)
# split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

modelB = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

df, evals = plot_predictions(modelB, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
bk_metrics2.update(evals)

# add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

# append each station df to results df
lstmberk_results_df2 = lstmberk_results_df2.append(df)

print(bk_metrics2)
lstmberk_results_df2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

# MAGIC %md
# MAGIC ###Univariate Multistep 
# MAGIC 6 steps, 10 minutes to 1 hour

# COMMAND ----------

stations = pa_transformed['Station_Location'].unique()
lstm_paloalto_results_df = pd.DataFrame()
metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'PaloAlto'
date_col = 'datetime_pac'


for station in stations:
    stationdf = pd.DataFrame( pa_transformed[pa_transformed['Station_Location'] == station][
        ['datetime_pac', 'Ports_Available']]).sort_values(by=['datetime_pac'])
    
    # filter to last 3 months of dataset
    end = stationdf[date_col].max()
    start_date = end - dt.timedelta(days = int(3*30))
    start  = start_date.replace(hour=0, minute=0, second = 0)
    ## filter data set
    pa = arima_filter(stationdf, start, end, date_col)
    
    
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
    lstm_paloalto_results_df = lstm_paloalto_results_df.append(df)

print(metrics)
lstm_paloalto_results_df.head()


# COMMAND ----------

# ## write dataframe to s3

# # create spark data frame
# results_1 = spark.createDataFrame(lstm_paloalto_results_df) 

# cols = results_1.columns
# for col in cols:
#     results_1 = results_1.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_1)

# ## Write to AWS S3
# (results_1
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/PaloAlto_LSTM_MultiStepNoFeatures_Tail129600"))


# #https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python 
# #https://towardsdatascience.com/reading-and-writing-files-from-to-amazon-s3-with-pandas-ccaf90bfe86c

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder CO

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Multistep
# MAGIC 6 steps

# COMMAND ----------

boulder_tf = boulder.copy()
boulder_tf.head()


# COMMAND ----------

stations = boulder_tf['proper_name'].unique()

date_col = 'Date Time'
lstmboulder_results_df = pd.DataFrame()
b_metrics = dict()
n_inputs = 6
n_outputs = 6
loc_ = 'Boulder'



for station in stations:
    b = pd.DataFrame( boulder_tf[boulder_tf['proper_name'] == station][
        ['Date Time', 'Ports Available']]).sort_values(by=['Date Time'])

    
    # filter to last 3 months of dataset
    end = b[date_col].max()
    start_date = end - dt.timedelta(days = int(3*30))
    start  = start_date.replace(hour=0, minute=0, second = 0)
    ## filter data set
    b = arima_filter(b, start, end, date_col)
    
    
    
    #b = b.tail(129600) #### this was missing a 0! last 3 months of the data , 10*6*24
    
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
    lstmboulder_results_df = lstmboulder_results_df.append(df)

print(b_metrics)
lstmboulder_results_df.head()


# COMMAND ----------

# ## write dataframe to s3

# # create spark data frame
# results_3 = spark.createDataFrame(lstmboulder_results_df) 

# cols = results_3.columns
# for col in cols:
#     results_3 = results_3.withColumnRenamed(col, col.replace(" ", "_"))
# display(results_3)

# ## Write to AWS S3
# (results_3
#      .repartition(1)
#     .write
#     .format("parquet")
#     .mode("overwrite")
#     .save(f"/mnt/{mount_name}/data/Boulder_LSTM_MultiStepNoFeatures_Tail129600"))



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Results Visual

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Results

# COMMAND ----------

# ## read in lstm results to calculate one step MSE/RMSE 
# boulder_res = spark.read.parquet(f"/mnt/{mount_name}/data/Boulder_LSTM_MultiStepNoFeatures_Tail129600/")
# boulder_res = boulder_res.toPandas()

# berkeley_res = spark.read.parquet(f"/mnt/{mount_name}/data/Berkeley_LSTM_MultiStepNoFeatures_3months/")
# berkeley_res = berkeley_res.toPandas()

# paloalto_res = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_LSTM_MultiStepNoFeatures_Tail129600/")
# paloalto_res = paloalto_res.toPandas()

# COMMAND ----------

# ## read in prophetresults to calculate one step MSE/RMSE 
# boulderph_res = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_prophet/")
# boulderph_res = boulderph_res.toPandas()

# paloaltoph_res = spark.read.parquet(f"/mnt/{mount_name}/data/paloAlto_prophet/")
# paloaltoph_res = paloaltoph_res.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arima

# COMMAND ----------

arima_slrp = {'Berkeley': {'Slrp': {'MSE_Raw': 1.3430667625018382, 'MSE_round': 1.4353182751540041, 'RMSE_Raw': 1.1589075728900204, 'RMSE': 1.1980476931883823}}}


## arima Palo ALto
arima_pa = {'Palo Alto': 
            {'CAMBRIDGE': {'MSE_Raw': 1.032623265576834, 'MSE_round': 1.083885772565018, 'RMSE_Raw': 1.0161807248599208, 'RMSE': 1.0410983491318282}, 
             'HAMILTON': {'MSE_Raw': 0.14388850599598707, 'MSE_round': 0.0, 'RMSE_Raw': 0.3793263845239177, 'RMSE': 0.0}, 
             'HIGH': {'MSE_Raw': 0.9776234770620584, 'MSE_round': 1.138194798572157, 'RMSE_Raw': 0.9887484397267378, 'RMSE': 1.0668621272555123}, 
             'MPL': {'MSE_Raw': 0.628790081147563, 'MSE_round': 0.8296787353391127, 'RMSE_Raw': 0.7929628497903057, 'RMSE': 0.9108670239607496}, 
             'SHERMAN': {'MSE_Raw': 0.27357851593931004, 'MSE_round': 0.3232016210739615, 'RMSE_Raw': 0.5230473362319228, 'RMSE': 0.568508241869862}, 
             'TED_THOMPSON': {'MSE_Raw': 0.42436444509175103, 'MSE_round': 0.8268740438551759, 'RMSE_Raw': 0.6514326097853492, 'RMSE': 0.9093261482302023}, 
             'WEBSTER': {'MSE_Raw': 0.8527277228908278, 'MSE_round': 0.9739928607853137, 'RMSE_Raw': 0.9234325762560187, 'RMSE': 0.9869107663742015}, 
             'RINCONADA_LIB': {'MSE_Raw': 2.1476738144642046e-28, 'MSE_round': 0.0, 'RMSE_Raw': 1.4654943925052066e-14, 'RMSE': 0.0}
            }}


arima_bo = {'Boulder': 
            {'1745 14th street': {'MSE_Raw': 0.3085052288335017, 'MSE_round': 0.48324379636735737, 'RMSE_Raw': 0.5554324700929013, 'RMSE': 0.695157389637309}, 
             '2052 Junction Pl': {'MSE_Raw': 0.11566671407451307, 'MSE_round': 0.1286774111025838, 'RMSE_Raw': 0.34009809478224523, 'RMSE': 0.3587163379365147}, 
             '2667 Broadway': {'MSE_Raw': 0.033889935874492474, 'MSE_round': 0.0332565873624968, 'RMSE_Raw': 0.18409219395317247, 'RMSE': 0.1823638872213926}, 
             '3172 Broadway': {'MSE_Raw': 0.555043263980237, 'MSE_round': 0.8176004093118444, 'RMSE_Raw': 0.7450122576040189, 'RMSE': 0.9042125907726813}, 
             '3335 Airport Rd': {'MSE_Raw': 0.17429547140499302, 'MSE_round': 0.1951905858275774, 'RMSE_Raw': 0.41748709130342343, 'RMSE': 0.4418037865699856}, 
             '5333 Valmont Rd': {'MSE_Raw': 0.49902037561451096, 'MSE_round': 0.6336658992069583, 'RMSE_Raw': 0.7064137425153272, 'RMSE': 0.7960313431058844}, 
             '5565 51st St': {'MSE_Raw': 0.043095752905294944, 'MSE_round': 0.043745203376822715, 'RMSE_Raw': 0.20759516590059351, 'RMSE': 0.20915354019672416}, 
             '5660 Sioux Dr': {'MSE_Raw': 0.14700462506719372, 'MSE_round': 0.16781785622921463, 'RMSE_Raw': 0.3834118217624409, 'RMSE': 0.4096557777320059}, 
             '600 Baseline Rd': {'MSE_Raw': 0.34236507316333825, 'MSE_round': 0.4223586595037094, 'RMSE_Raw': 0.5851197084044754, 'RMSE': 0.6498912674468779}, 
             '900 Walnut St': {'MSE_Raw': 0.2831676110599813, 'MSE_round': 0.34586850856996676, 'RMSE_Raw': 0.5321349556832189, 'RMSE': 0.5881058651042061}}}


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
# MAGIC ### Prophet

# COMMAND ----------

prophet_slrp = {'Berkeley': {'Slrp': {'MSE_raw': 1.4402963892556122, 'MSE_round': 1.1745379876796715, 'RMSE_raw': 1.200123489169182, 'RMSE_round': 1.0837610380889653}}}
                
                

        
# boulder_metrics2 = dict()
# stations = boulderph_res['SiteName'].unique()


# for station in stations: 
#     stationdf = boulderph_res[boulderph_res['SiteName'] == station]
    
#     MSE_raw = mse(stationdf['Actuals'], stationdf['yhat'])
#     MSE_rounded = mse(stationdf['Actuals'], stationdf['rounded'])
#     RMSE_raw = math.sqrt(MSE_raw)
#     RMSE_rounded = math.sqrt(MSE_rounded)

#     Evals = dict({station: 
#                  dict(
#                      {'MSE_Raw': MSE_raw,
#                       'MSE_round': MSE_rounded,
#                       'RMSE_Raw': RMSE_raw,
#                       'RMSE': RMSE_rounded
#                      }) 
#                 })
    
#     boulder_metrics2.update(Evals)

boulder_metrics2 = boulder_prophet_res
prophet_boulder = {'Boulder' : boulder_metrics2}




# prophet_paloalto = dict()
# stations = paloaltoph_res['SiteName'].unique()


# for station in stations: 
#     stationdf = paloaltoph_res[paloaltoph_res['SiteName'] == station]
    
#     MSE_raw = mse(stationdf['Actuals'], stationdf['yhat'])
#     MSE_rounded = mse(stationdf['Actuals'], stationdf['rounded'])
#     RMSE_raw = math.sqrt(MSE_raw)
#     RMSE_rounded = math.sqrt(MSE_rounded)

#     Evals = dict({station: 
#                  dict(
#                      {'MSE_Raw': MSE_raw,
#                       'MSE_round': MSE_rounded,
#                       'RMSE_Raw': RMSE_raw,
#                       'RMSE': RMSE_rounded
#                      }) 
#                 })
    
#     prophet_paloalto.update(Evals)

prophet_paloalto = paloalto_prophet_res
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

# MAGIC %md
# MAGIC ### LSTM - multi

# COMMAND ----------

# lstm_slrp = {'Berkeley':{'slrp': {'MSE_Raw': 0.18017677965201723, 'MSE_round': 0.18870626322998382, 'RMSE_Raw': 0.42447235440251846, 'RMSE': 0.43440334164228506}}} ##no features
lstm_slrp = {'Berkeley':{'slrp': {'MSE_Raw': 0.20934948151372632, 'MSE_round': 0.24069231727057652, 'RMSE_Raw': 0.4575472451165304, 'RMSE': 0.4906040330761423}}} ##features


lstm_palo_alto = {'Palo Alto':
                  {'HIGH': {'MSE_Raw': 0.19531087989490256, 'MSE_round': 0.2039965986394558, 'RMSE_Raw': 0.44193990529811017, 'RMSE': 0.45165982624034184}, 
                   'BRYANT': {'MSE_Raw': 0.7814800004543127, 'MSE_round': 0.6221938775510204, 'RMSE_Raw': 0.8840135748133694, 'RMSE': 0.7887926708273983}, 
                   'HAMILTON': {'MSE_Raw': 0.002097285332060513, 'MSE_round': 0.0, 'RMSE_Raw': 0.045796127915583795, 'RMSE': 0.0}, 
                   'MPL': {'MSE_Raw': 0.21667496928196373, 'MSE_round': 0.22329931972789116, 'RMSE_Raw': 0.4654835864796564, 'RMSE': 0.47254557423373583}, 
                   'RINCONADA_LIB': {'MSE_Raw': 7.3931685140948815, 'MSE_round': 9.0, 'RMSE_Raw': 2.7190381597349607, 'RMSE': 3.0}, 
                   'WEBSTER': {'MSE_Raw': 26.89036790525547, 'MSE_round': 29.860204081632652, 'RMSE_Raw': 5.1855923388997205, 'RMSE': 5.46444911053554}, 
                   'TED_THOMPSON': {'MSE_Raw': 0.13699481353532542, 'MSE_round': 0.14829931972789115, 'RMSE_Raw': 0.3701281042224778, 'RMSE': 0.3850965070315377}, 
                   'CAMBRIDGE': {'MSE_Raw': 0.23483353097958473, 'MSE_round': 0.23622448979591837, 'RMSE_Raw': 0.4845962556392535, 'RMSE': 0.4860293096058286}, 
                   'SHERMAN': {'MSE_Raw': 312.20047185842213, 'MSE_round': 314.0271002710027, 'RMSE_Raw': 17.669195563421162, 'RMSE': 17.720809808555668}}}


# ###
# lstm_boulder = {'Boulder':{'1100 Walnut': {'MSE_Raw': 0.17494893591480404, 'MSE_round': 0.21787613655858637, 'RMSE_Raw': 0.41826897555855613, 'RMSE': 0.4667720391782121}, '1739 Broadway': {'MSE_Raw': 0.040585291020749696, 'MSE_round': 0.025819179962257677, 'RMSE_Raw': 0.20145791377046893, 'RMSE': 0.1606834775646136}, '3172 Broadway': {'MSE_Raw': 0.23929800137188761, 'MSE_round': 0.29336078229541945, 'RMSE_Raw': 0.48918094951856783, 'RMSE': 0.5416279002187936}, '5660 Sioux Dr': {'MSE_Raw': 0.110643421351928, 'MSE_round': 0.13621547435237605, 'RMSE_Raw': 0.33263105891051126, 'RMSE': 0.36907380610438345}, '5565 51st St': {'MSE_Raw': 0.00974294032030815, 'MSE_round': 0.012695144964831017, 'RMSE_Raw': 0.09870633373957392, 'RMSE': 0.11267273390146801}, '1505 30th St': {'MSE_Raw': 0.37030223056725914, 'MSE_round': 0.4275175844913364, 'RMSE_Raw': 0.6085246343142232, 'RMSE': 0.6538482885894376}, '3335 Airport Rd': {'MSE_Raw': 0.10199118015470376, 'MSE_round': 0.1182878709898782, 'RMSE_Raw': 0.31936058015150176, 'RMSE': 0.34393003792905064}, '600 Baseline Rd': {'MSE_Raw': 0.18497864597355698, 'MSE_round': 0.22525304511923142, 'RMSE_Raw': 0.4300914390842452, 'RMSE': 0.47460830704827683}, '2667 Broadway': {'MSE_Raw': 0.007905993510320375, 'MSE_round': 0.007720020586721565, 'RMSE_Raw': 0.08891565391043567, 'RMSE': 0.08786364769756355}, '5333 Valmont Rd': {'MSE_Raw': 5.169619033276567, 'MSE_round': 4.784182535597873, 'RMSE_Raw': 2.27367962415037, 'RMSE': 2.18727742538478}, '1745 14th street': {'MSE_Raw': 0.01586428088190246, 'MSE_round': 0.01604048721907703, 'RMSE_Raw': 0.12595348697794143, 'RMSE': 0.12665104507692398}, '1500 Pearl St': {'MSE_Raw': 0.06956762677206524, 'MSE_round': 0.07282552753474009, 'RMSE_Raw': 0.26375675682731853, 'RMSE': 0.26986205278760494}, '1770 13th St': {'MSE_Raw': 0.05647577274809522, 'MSE_round': 0.0680219591696689, 'RMSE_Raw': 0.237646318608337, 'RMSE': 0.2608101975952415}, '1360 Gillaspie Dr': {'MSE_Raw': 0.11738864561331476, 'MSE_round': 0.14187682278263852, 'RMSE_Raw': 0.34262026445222815, 'RMSE': 0.3766653989718707}, '900 Walnut St': {'MSE_Raw': 0.7058714121733698, 'MSE_round': 0.8537484988848859, 'RMSE_Raw': 0.8401615393323892, 'RMSE': 0.9239851183243624}, '1100 Spruce St': {'MSE_Raw': 0.2136945489251706, 'MSE_round': 0.23400240178418252, 'RMSE_Raw': 0.46227107731846107, 'RMSE': 0.48373794743040627}, '1400 Walnut St': {'MSE_Raw': 0.034565883876274184, 'MSE_round': 0.043146337279121635, 'RMSE_Raw': 0.18591902505196767, 'RMSE': 0.20771696435082435}, '2052 Junction Pl': {'MSE_Raw': 0.07606921945595811, 'MSE_round': 0.10550694801852804, 'RMSE_Raw': 0.27580648914765965, 'RMSE': 0.3248183307920414}}}
lstm_boulder = {'Boulder':b_metrics}


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

lstm_paloalto_results_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LSTM - One

# COMMAND ----------

#### Boulder one step metrics ####
boulder_metrics1 = dict()
boulder1 = lstmboulder_results_df[lstmboulder_results_df['StepsOut'] == 1]
stations = boulder1['SiteName'].unique()


for station in stations: 
    stationdf = boulder1[boulder1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions (rounded)'])
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
paloalto1 = lstm_paloalto_results_df[lstm_paloalto_results_df['StepsOut'] == 1]
stations = paloalto1['SiteName'].unique()


for station in stations: 
    stationdf = paloalto1[paloalto1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions (rounded)'])
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
berkeley1 = lstmberk_results_df2[lstmberk_results_df2['StepsOut'] == 1]
stations = lstmberk_results_df2['SiteName'].unique()


for station in stations: 
    stationdf = berkeley1[berkeley1['SiteName'] == station]
    
    MSE_raw = mse(stationdf['Actuals'], stationdf['Predictions'])
    MSE_rounded = mse(stationdf['Actuals'], stationdf['Predictions (rounded)'])
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
# MAGIC ## Charts

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
    tooltip = ['MSE_Raw:Q', 'MSE_round', 'RMSE_Raw+
               ', 'RMSE_round', 'location', 'station:O']
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
    tooltip = ['MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q', 'location', 'station:O']
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
    tooltip = ['MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q', 'location', 'station:O']
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

# COMMAND ----------

# MAGIC %md
# MAGIC # Neural Prophet

# COMMAND ----------

# from neuralprophet import NeuralProphet
# ## cannot have missing data in NP
# ## expecting only for ds and y


# def run_Nprophet(traindf, testdf, date_col, output_col, station):
#     if date_col != 'ds':
#         traindf = traindf.rename(columns={date_col: 'ds'})
        
#     if output_col != 'y':
#         traindf = traindf.rename(columns={output_col: "y"})
        
    
#     print(traindf.columns)
    
#     traindf = traindf[['ds', 'y']]
    
    
#     # create model
#     m = NeuralProphet()
#     m.fit(traindf, freq = '10min')#, epochs = 10)
    
#     # make predictions
#     future = m.make_future_dataframe(traindf, periods = testdf.shape[0])#, freq = '10min')
#     forecast = m.predict(future)
    
    
    
#     preds = forecast[(forecast['ds'] <= testdf[date_col].max()) & (forecast['ds'] >= testdf[date_col].min())]
    
#     print(preds)
#     # rounding predictions
#     ## need to change how we are rounding if there is more than 1 station being predicted for
#     ## this assumes that there is at least one point in time when all stations are available (probably a fair assumption)
#     preds['rounded'] = np.around(preds['yhat']).clip(upper = traindf['y'].max())
#     preds = preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'rounded']]
    
#     #create dataframe to output
#     testdf = testdf.rename(columns = {output_col: "Actuals", date_col: 'DateTime'})
#     testdf = testdf[['DateTime', 'Actuals']].merge(preds, left_on = 'DateTime', right_on = 'ds')
    
#     pred_col = 'yhat'
    
#     ## Evaluation Metrics ###
#     MSE_raw = mse(testdf['Actuals'], testdf[pred_col])
#     MSE_rounded = mse(testdf['Actuals'], testdf['rounded'])
#     RMSE_raw = math.sqrt(MSE_raw)
#     RMSE_rounded = math.sqrt(MSE_rounded)
    
#     Evals = dict({station: 
#                  dict({'MSE_raw': MSE_raw,
#                       'MSE_round': MSE_rounded,
#                       'RMSE_raw': RMSE_raw,
#                       'RMSE_round': RMSE_rounded}) 
#                  })

# #     Evals = dict({'MSE_raw': MSE_raw,
# #                       'MSE_round': MSE_rounded,
# #                       'RMSE_raw': RMSE_raw,
# #                       'RMSE_round': RMSE_rounded})
    
#     print(Evals)
#     print(m.test(testdf))
    
#     return m, testdf, Evals
    

# COMMAND ----------

## TESTING THE PROPHET FUNCTION

start = dt.datetime(2022, 1, 1)
end = dt.datetime(2022, 5, 5)
date_col = 'DateTime'
actualcol= 'Ports Available'
output_col = 'Ports Available'
station = 'Slrp'

## filter data set
slrp_p = arima_filter(slrp_transformed, start, end, date_col)

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(slrp_p, 0.7)

traindf = slrp_prophet_train
testdf = slrp_prophet_test

#prophet_model, prophet_testdf, prophet_Evals = run_Nprophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station)


if date_col != 'ds':
    traindf = traindf.rename(columns={date_col: 'ds'})
    testdf = testdf.rename(columns={date_col: 'ds'})

if output_col != 'y':
    traindf = traindf.rename(columns={output_col: "y"})
    testdf = testdf.rename(columns={output_col: "y"})
        
print(traindf.columns)
traindf = traindf[['ds', 'y']]
testdf = testdf[['ds', 'y']]
    
# create model
m = NeuralProphet()
m.fit(traindf, freq = '10min')#, epochs = 10)
    
# make predictions
future = m.make_future_dataframe(traindf, periods = testdf.shape[0])#, freq = '10min')
forecast = m.predict(future)

print(forecast)
print(m.test(testdf))
    
preds = forecast[(forecast['ds'] <= testdf['ds'].max()) & (forecast['ds'] >= testdf['ds'].min())]

print(preds)
# # rounding predictions
# ## need to change how we are rounding if there is more than 1 station being predicted for
# ## this assumes that there is at least one point in time when all stations are available (probably a fair assumption)
# preds['rounded'] = np.around(preds['yhat']).clip(upper = traindf['y'].max())
# preds = preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'rounded']]

# #create dataframe to output
# testdf = testdf.rename(columns = {output_col: "Actuals", date_col: 'DateTime'})
# testdf = testdf[['DateTime', 'Actuals']].merge(preds, left_on = 'DateTime', right_on = 'ds')

# pred_col = 'yhat'

# ## Evaluation Metrics ###
# MSE_raw = mse(testdf['Actuals'], testdf[pred_col])
# MSE_rounded = mse(testdf['Actuals'], testdf['rounded'])
# RMSE_raw = math.sqrt(MSE_raw)
# RMSE_rounded = math.sqrt(MSE_rounded)

# Evals = dict({station: 
#              dict({'MSE_raw': MSE_raw,
#                   'MSE_round': MSE_rounded,
#                   'RMSE_raw': RMSE_raw,
#                   'RMSE_round': RMSE_rounded}) 
#              })

# #     Evals = dict({'MSE_raw': MSE_raw,
# #                       'MSE_round': MSE_rounded,
# #                       'RMSE_raw': RMSE_raw,
# #                       'RMSE_round': RMSE_rounded})

# print(Evals)
# print(m.test(testdf))

    

