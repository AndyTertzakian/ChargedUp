# Databricks notebook source
# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

# import libraries
import pandas as pd
# import altair as alt
import datetime as dt
import numpy as np
#import json
import time
#import urllib
#import requests
import seaborn as sns
#from vega_datasets import data
#from pandas.io.json import json_normalize
from datetime import timedelta
#import plotly.graph_objects as go
import pytz
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
# import folium
import holidays
import boto3
import pyspark.sql.functions as F
import pickle

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

# Saving plots
import io

# Prevent Java tracking logs
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC # AWS Setup

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

# MAGIC %md
# MAGIC # Reference Functions

# COMMAND ----------

def df_filter(df, start_date, end_date, date_col):
    ''' 
    filter data frame to be between the start date and end date, not including the end date. 
    date_col is the date column used to filter the dataframe
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] # should we reset the index

# write results to s3 bucket
def write_seed_to_s3(df, save_filename):

    # create spark data frame
    results_ab = spark.createDataFrame(df) 

    ## Write to AWS S3
    (results_ab
        .repartition(1)
        .write
        .format("parquet")
        .mode("overwrite")
        .save(f"/mnt/{mount_name}/data/{save_filename}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM Reference Functions

# COMMAND ----------

def dfSplit_Xy(df, date_col='datetime', n_input=6, n_out=36):
    """ 
    Tranform pandas dataframe into arrays of inputs and outputs. 
    The output (value predicting) must be located at the end of the df
    n_inputs, is the the number of inputs, to use to predict n_input+1 (or i + n_input) aka window size
    n_outs, is the number of steps out you want to predict. Default is 1
    Returns 2 numpy arrays. Inputs (features), and actual labels
    """
    
    ind_dates = df[date_col].to_numpy() ####
    df_as_np = df.set_index(date_col).to_numpy()
    
    #X is matrix of inputs
    #y is actual outputs, labels
    X = []
    y = []
    y_dates = [] ####
    
    
    for i in range(len(df_as_np)): 
        #print(i)
        start_out = i + n_input
        start_end = start_out + n_out

        # to make sure we always have n_out values in label array
        if start_end > len(df_as_np):
            break

        #take the i and the next values, makes a list of list for multiple inputs
        row = df_as_np[i:start_out, :]
        #print(row)
        X.append(row)

        # Creating label outputs extended n_out steps. -1, last column is label
        label = df_as_np[start_out:start_end, -1]
        #print(label)
        y.append(label)
        
        # array of dates
        label_dates = ind_dates[start_out:start_end]####
        #print(label_dates)###
        y_dates.append(label_dates) #### 
        
#         print('X shape == {}.'.format(np.array(X).shape))
#         print('y shape == {}.'.format(np.array(y).shape))
    # can we have a ydates for the dates??? and timesteps
    
    return np.array(X), np.array(y), np.array(y_dates)


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

def run_lstm(n_inputs, n_features, n_outputs, X_train, y_train, X_val, y_val, n_epochs = 10):
    '''Run lstm model, and get fitted model'''
    
    # Build LSTM Model
    model = Sequential()
    model.add(InputLayer((n_inputs, n_features))) 
    model.add(LSTM(64, input_shape = (n_inputs, n_features), activation = 'relu'))
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

def plot_predictions(model, X_test, y_test, train_df, station, train_end, date_col, y_dates, start=0, end=1000):
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
    
    all_test_timestamps = y_dates.flatten()


    ### get predictions, vector of predictions with inputs and flatten
    predictions = model.predict(X_test)
    predictions = predictions.flatten()
    
#     predictions_rounded = np.round(predictions)

#     print(predictions.shape)
    # should return dataframe
    df = pd.DataFrame(data = {'Predictions': predictions,
                              'Actuals': y_test.flatten(),
                              'DateTime': all_test_timestamps})
    
    rounded = pd.DataFrame(data = {'Predictions (rounded)': predictions})
    
    # Round the predictions to (# of stations * ~10 kW) as upper bound, and 0 as lower bound
    rounded = rounded.clip(upper = pd.Series({'Predictions (rounded)': 8 * 10000}), 
                           lower = pd.Series({'Predictions (rounded)': 0}), axis=1)
    
    df['Predictions (rounded)'] = rounded['Predictions (rounded)']
    
    df['DateTime'] = all_test_timestamps
    
    print(df.head())
    
    ### evaluation metrics
    MSE_raw = mse(y_test.flatten(), predictions)
    MSE_rounded = mse(y_test.flatten(), df['Predictions (rounded)'])
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
# MAGIC ## ARIMA Reference Functions
# MAGIC 
# MAGIC `arima_filter`
# MAGIC 
# MAGIC `split2_TrainTest`
# MAGIC 
# MAGIC `plot_predsActuals`

# COMMAND ----------

def arima_filter(df, start_date, end_date, date_col):
    ''' 
    filter data frame to be between the start date and end date, not including the end date. 
    date_col is the date column used to filter the dataframe
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] 

# COMMAND ----------

### simple split into train and test
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
# MAGIC # Create Power Data

# COMMAND ----------

# DBTITLE 1,Load Original Time Series Data
slrp_df = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_ts")
slrp_df = slrp_df.toPandas()
# slrp_df[slrp_df["Date"] == dt.date(2022,2,1)].head(36)

# COMMAND ----------

cols_to_keep = ["DateTime", "station", 
#                 "Ports Available",
                "power_W"
               ]
seed_df = slrp_df[cols_to_keep]


rename_cols_dict = {"DateTime": "datetime",
                    "station": "station",
                    "power_W": "power_W",
#                     "Ports Available": 'ports_available'
                   }
seed_df = seed_df.rename(columns=rename_cols_dict)

seed_df = seed_df.drop_duplicates()

seed_df = df_filter(seed_df, 
                    start_date= dt.datetime(2021, 11, 1), 
                    end_date= seed_df['datetime'].max() + dt.timedelta(minutes = 10), 
                    date_col="datetime")
seed_df = seed_df.sort_values(by="datetime")
seed_df = seed_df.reset_index(drop=True)
seed_df.head()
write_seed_to_s3(seed_df, 'slrp_power')

# COMMAND ----------

write_seed_to_s3(seed_df[seed_df['datetime'] >= dt.datetime(2022, 2, 1, 0, 0, 0)], 'slrp_power_test')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Power Data

# COMMAND ----------

slrp = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_power")
slrp = slrp.toPandas()

slrp_seed = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_power_seed")
slrp_seed = slrp_seed.toPandas()

slrp_test = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_power_test")
slrp_test = slrp_test.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Features to Data

# COMMAND ----------

def add_features(df):
    df_transformed = df.copy()
    
    # Add columns to build off of
    df_transformed['month'] = df_transformed['datetime'].dt.month
    df_transformed['day_of_week'] = df_transformed['datetime'].dt.dayofweek
    df_transformed['weekend'] = np.where(df_transformed['day_of_week'] >= 5, 1, 0)
    
    
    # Apply cosine and sine transformations to cyclical features
    df_transformed['month_cosine'] = np.cos(2 * math.pi * df_transformed['month'] / df_transformed['month'].max())
    df_transformed['month_sine'] = np.sin(2 * math.pi * df_transformed['month'] / df_transformed['month'].max())
    df_transformed['hour_cosine'] = np.cos(2 * math.pi * df_transformed['datetime'].dt.hour / 
                                             df_transformed['datetime'].dt.hour.max())
    df_transformed['hour_sine'] = np.sin(2 * math.pi * df_transformed['datetime'].dt.hour / 
                                           df_transformed['datetime'].dt.hour.max())
    df_transformed['dayofweek_cosine'] = np.cos(2 * math.pi * df_transformed['day_of_week'] / 
                                                  df_transformed['day_of_week'].max())
    df_transformed['dayofweek_sine'] = np.sin(2 * math.pi * df_transformed['day_of_week'] / 
                                                df_transformed['day_of_week'].max())

    # Drop unnecessary columns
    df_transformed = df_transformed.drop(columns = ['month', 'day_of_week'])
    
    #move output to end of col for LSTMs
    df_transformed = df_transformed[[c for c in df_transformed if c not in ['power_W']] + ['power_W']]
    
    return df_transformed

# COMMAND ----------

slrp_features = add_features(slrp)
slrp_seed_features = add_features(slrp_seed)
slrp_test_features = add_features(slrp_test)

# COMMAND ----------

####### Filtering data to 129 days ###########
## Use for training 3 months of data in a 70/30 train test split --> means we should select 129 days for training + test data
start = dt.datetime(2021, 11, 1)
end_date = (start + dt.timedelta(days = int(129))).replace(hour = 0, minute = 0, second = 0)
slrp_feat = arima_filter(slrp_features, start, end_date, 'datetime')

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Overall Average

# COMMAND ----------

def predict_average_overall(df, n_out):
    """
    Use the entire training set to make predictions of ports available
    """
    return [(df['power_W'].mean())] * n_out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Average Over Last 12 Timesteps
# MAGIC 
# MAGIC `n_inputs`=12 in LSTM models

# COMMAND ----------

def predict_average_n_timestamps(df, n_in, n_out):
    """
    Use the last n_in timesteps only to make predictions of ports available for n_out timesteps out
    """
    
    # Get the last n_in entries from the ports available column
    train_set = list(df.tail(n_in)['power_W'])
    
    # Define list for the predictions
    preds = []
    
    # For each prediction you want to make
    for i in range(n_out):
        # Make the prediction based on the mean of the train set
        prediction = np.mean(train_set)
        
        # Update the predictions list
        preds.append(prediction)
        
        # Update the training set by using the prediction from the last timestep and dropping the first timestep
        train_set.append(prediction)
        train_set.pop(0)
    
    return preds

# COMMAND ----------

def predict_avg_by_day_hour(df, df_test):
    """
    Make predictions based on the day of week and the hour of day -- return the average
    """
    df_mod = df.copy()
    df_test_mod = df_test.copy()
    
    # Add day of week and hour columns
    df_mod['day_of_week'] = df['datetime'].dt.dayofweek
    df_mod['hour'] = df['datetime'].dt.hour
    df_test_mod['day_of_week'] = df_test['datetime'].dt.dayofweek
    df_test_mod['hour'] = df_test['datetime'].dt.hour
    
    # Group by these features, calculate the mean, rename the column
    df_grouped = df_mod.groupby(['day_of_week', 'hour']).mean()
    df_grouped = df_grouped.rename(columns = {'power_W': 'prediction'})
    # Don't need to round bc power is a continuous, not discrete variable
    # df_grouped = df_grouped.round({'prediction': 0})
    df_grouped = df_grouped.reset_index()
    
    df_preds = df_test_mod.merge(df_grouped, how = 'left', on = ['day_of_week', 'hour'])
    
    return df_preds['prediction']

# COMMAND ----------

# MAGIC %md
# MAGIC # More Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## ARIMA Model

# COMMAND ----------

def run_arima(traindf, testdf, actualcol, date_col, station, no_features = True):
    '''
    run arima model given a training df, a testing df. 
    as a string provide the name of the actualcol aka output column, the date column, 
    and the station name
    '''
    print(station)
    #### new
    traindf = traindf.set_index(date_col, drop=False, inplace=False)
    testdf = testdf.set_index(date_col, drop = False, inplace = False)
    
    ########## check if we have features #################
    # no features #
    if no_features: 
        ### get model parameters
        print("no features")
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
        #model.summary()
        
        ### get predictions
        pred = model.predict(start = traindf.shape[0], end = traindf.shape[0] + testdf.shape[0] - 1, typ='levels')

        ### getting actual data from previous data
        testdf = testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'})

        ## createdf to output
        testdf['predictions'] = pred.values
        testdf['predictions (rounded)'] = np.around(pred).values
    
    # features #
    else: 
        print("yes features")
        ### get model parameters
        values_p = auto_arima(traindf[actualcol], 
                              exogenous=traindf.loc[:, (traindf.columns != actualcol) & (traindf.columns != date_col)],
                              d = 0, 
                              trace = True, 
                              suppress_warnings = True)
        print(values_p)
        p_order = values_p.get_params().get("order")
        print('order complete for ', station)

        ## fit model
        # parameters based on autoarima
        model = ARIMA(traindf[actualcol],
                      exog=traindf.loc[:, (traindf.columns != actualcol) & (traindf.columns != date_col)], 
                      order = p_order)
        print('model created')
        model = model.fit()
        print('model fit')
        # model.summary()
    
        ### get predictions
        pred = model.predict(start = traindf.shape[0], 
                             end = traindf.shape[0] + testdf.shape[0] - 1, 
                             exog = testdf.loc[:, (testdf.columns != actualcol) & (testdf.columns != date_col)], 
                             typ='levels')

        ### getting actual data from previous data
        testdf = testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'})

        ## createdf to output
        testdf['predictions'] = pred.values
        testdf['predictions (rounded)'] = np.around(pred).values
    
    #############
    
    ###### Evaluation Metrics #########
    MSE_raw = mse(testdf['Actuals'], testdf['predictions'])
    MSE_rounded = mse(testdf['Actuals'], testdf['predictions (rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    Evals = dict({station: 
                 dict({'MSE_Raw': MSE_raw,
                      'MSE_round': MSE_rounded,
                      'RMSE_Raw': RMSE_raw,
                      'RMSE': RMSE_rounded,
                      'PDQ_ARIMAOrder': p_order}) 
                 }) 
    
    print(Evals)
    
    return model, testdf, Evals, p_order

# COMMAND ----------

def arima_fit(traindf, testdf, actualcol, date_col, station, p, d, q):
    ## fit model
    # parameters based on autoarima
    model = ARIMA(traindf[actualcol], order = (p, d, q))
    print('model created')
    model = model.fit()
    print('model fit')
    # model.summary()
    
    return model

# COMMAND ----------

def arima_eval(traindf, testdf, actualcol, date_col, station, model):
    ### get predictions
    pred = model.predict(start = traindf.shape[0], end = traindf.shape[0] + testdf.shape[0] - 1, typ='levels')

    ### getting actual data from previous data
    testdf = testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'})
    
    ## createdf to output
    testdf['predictions'] = pred.values
    testdf['predictions (rounded)'] = np.around(pred).values
    
    ## Evaluation Metrics ###
    MSE_raw = mse(testdf['Actuals'], testdf['predictions'])
    MSE_rounded = mse(testdf['Actuals'], testdf['predictions (rounded)'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    print(f'RMSE (rounded):\t{RMSE_rounded}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet

# COMMAND ----------

def run_prophet(traindf, testdf, date_col, output_col, station, no_features = True):
    if date_col != 'ds':
        traindf = traindf.rename(columns={date_col: 'ds'})
    if output_col != 'y':
        traindf = traindf.rename(columns={output_col: "y"})
    
    print(traindf.columns)
    
    # create model
    m = Prophet()
    
    #### check if there are features
    if no_features: 
        print('no features')
        m.fit(traindf)
        # make predictions
        future = m.make_future_dataframe(periods = testdf.shape[0], freq = '10min', include_history=False)
    
    else:
        print("features")
        ## add features
        features = traindf.loc[:, (traindf.columns != "y") & (traindf.columns != "ds")]
        print(features.columns)
        for col in features.columns:
            m.add_regressor(col)
        m.fit(traindf)
        
        # make predictions
        future = m.make_future_dataframe(periods = testdf.shape[0], freq = '10min', include_history=False)
        pred_features = testdf.loc[:, (testdf.columns != output_col) & (testdf.columns != date_col)]
        for col in features.columns:
            future[col] = pred_features[col].values
    
    ### predict
    forecast = m.predict(future)
    
    ### get predictopms
    preds = forecast[(forecast['ds'] <= testdf[date_col].max()) & (forecast['ds'] >= testdf[date_col].min())]
    
    # rounding predictions
    ## need to change how we are rounding if there is more than 1 station being predicted for
    ## this assumes that there is at least one point in time when all stations are available (probably a fair assumption)
    preds['rounded'] = np.around(preds['yhat']).clip(upper = traindf['y'].max())
    preds = preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'rounded']]
    
    ##### create dataframe to output
    testdf = testdf.rename(columns = {output_col: "Actuals", date_col: 'DateTime'})
    testdf = testdf[['DateTime', 'Actuals']].merge(preds, left_on = 'DateTime', right_on = 'ds')
    
    pred_col = 'yhat'
    
    ####### Evaluation Metrics ###
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
# MAGIC ## Calc RMSE

# COMMAND ----------

def calc_rmse(preds, actuals):
    """
    Calculate the RMSE between predictions and the actual values
    preds: series/array of predictions
    df: dataframe with column 'ports_available' to be used in calculation
    """
    
    return np.sqrt(mse(preds, actuals['power_W']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark Models

# COMMAND ----------

# DBTITLE 1,Benchmark Model 1: Predict Average of All Training Data Only
# slrp_test = slrp_6m[slrp_6m['datetime'] > slrp_seed['datetime'].max()]
slrp_preds_1 = predict_average_overall(slrp_seed, slrp_test.shape[0])
slrp_rmse_1 = calc_rmse(slrp_preds_1, slrp_test)
print(slrp_rmse_1)

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2A: Predict Average of Last n Timestamps (No Streaming)
n_in = 12
n_out = slrp_test.shape[0]
slrp_preds_2a = predict_average_n_timestamps(slrp_seed, n_in, n_out)
slrp_rmse_2a = calc_rmse(slrp_preds_2a, slrp_test.head(n_out))
print(slrp_rmse_2a)

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2B: Predict Average of Last n Timestamps (With Streaming)
streaming_frequency = 6 # get streaming updates each hour with values from the past hour
n_in = 12
n_out = streaming_frequency
results = slrp_test.copy().reset_index(drop = True)
results['predicted'] = ['']*results.shape[0]
all_rmses = []

for i in range(int(np.ceil(slrp_test.shape[0] / streaming_frequency))):
    slrp_preds_2b = predict_average_n_timestamps(pd.concat([slrp_seed, slrp_test.head(streaming_frequency*i)]), n_in, n_out)
    all_rmses.append(calc_rmse(slrp_preds_2b, slrp_test.iloc[i:i+n_out,:]))
    for pred_num in range(n_out):
        results.loc[i*n_out + pred_num, 'predicted'] = slrp_preds_2b[pred_num]

results = results.dropna()
results.head()

# COMMAND ----------

slrp_rmse_2b = calc_rmse(results['predicted'], results)
slrp_rmse_2b

# COMMAND ----------

plt.subplots(figsize = (10,5))
subset = results.head(750)
plt.plot(subset['datetime'], subset['power_W'], label = 'actual');
plt.plot(subset['datetime'], subset['predicted'], label = 'predictions');
plt.xlabel('Datetime');
plt.ylabel('Power (W)');
plt.title(f'SlrpEV Benchmark 1: Predict Average of All Training Data  |  RMSE = {np.round(slrp_rmse_2b, 4)}');
plt.legend();

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3A: Predict Average by Day of Week and Hour (No Streaming)
slrp_preds_3 = predict_avg_by_day_hour(slrp_seed, slrp_test)
slrp_rmse_3 = calc_rmse(slrp_preds_3, slrp_test)
print(slrp_rmse_3)

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3B: Predict Average by Day of Week and Hour (With Streaming)
# Retrain Monthly

month_list = slrp['datetime'].dt.month.unique()
weights = []
rmses = []

for month in range(3, len(slrp['datetime'].dt.month.unique())):
    train_temp = slrp[(slrp['datetime'].dt.month == month_list[month - 3]) | 
                      (slrp['datetime'].dt.month == month_list[month - 2]) |
                      (slrp['datetime'].dt.month == month_list[month - 1])]
    test_temp = slrp[(slrp['datetime'].dt.month == month_list[month])]
    weights.append(test_temp.shape[0])
    slrp_preds_3b = predict_avg_by_day_hour(train_temp, test_temp)
    slrp_rmse_3b = calc_rmse(slrp_preds_3b, test_temp)
    rmses.append(slrp_rmse_3b)
print(sum([rmses[i] * weights[i] for i in range(len(rmses))]) / sum(weights))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ARIMA

# COMMAND ----------

actual_col = 'power_W'
date_col = 'datetime'
station = 'Slrp'

# COMMAND ----------

#run model
berk_model, b_testdf, bevals = run_arima(slrp_seed, slrp_test, actual_col, date_col, station)


info = 'MSE Predictions: ' + str(bevals['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(bevals['Slrp']['MSE_round'])

# COMMAND ----------

print("Just so we don't have to rerun the ARIMA models...")
print("auto_arima found the best model to be:  ARIMA(5,0,3)(0,0,0)")
print("Evaluation Metrics:")
print("{'Slrp': {'MSE_Raw': 26657986.05945017, 'MSE_round': 26657830.72488486, 'RMSE_Raw': 5163.137230352314, 'RMSE': 5163.122187677226}}")

# COMMAND ----------

model = arima_fit(slrp_seed, slrp_test, actual_col, date_col, station, 5, 0, 3)

# COMMAND ----------

arima_eval(slrp_seed, slrp_test, actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*2), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*3), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*4), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*5), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*6), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*7), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*8), actual_col, date_col, station, model)
arima_eval(slrp_seed, slrp_test.head(1008*9), actual_col, date_col, station, model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ARIMA with Features

# COMMAND ----------

date_col = 'datetime'
actualcol= 'power_W'
station = 'Slrp'
df = slrp_feat.drop(columns = 'station')
 
## run EDA
# arima_eda(df, 'power_W', 25, df.shape[0]-1)
 
## split into train and test
traindf, testdf = split2_TrainTest(df, 0.7)
 
## run model
berk_model2, b_testdf2, bevals2, berk_pqd2 = run_arima(traindf, testdf, actualcol, date_col, station, False)
 
## plot
info = 'MSE Predictions: ' + str(bevals2['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(bevals2['Slrp']['MSE_round']) + '\n arima order = ' + str(berk_pqd2)
size_ = (15, 7)
plot_predsActuals(b_testdf2, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet

# COMMAND ----------

# prophet_model, prophet_testdf, prophet_Evals = run_prophet(slrp_seed, slrp_test, date_col, actual_col, station)

## No Features
date_col = 'datetime'
actualcol= 'power_W'
station = 'Slrp'
df = slrp_feat[[date_col, actualcol]]

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(df, 0.7)

prophet_model, prophet_testdf, prophet_Evals = run_prophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station)

plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# COMMAND ----------

print("Just so we don't have to rerun the Prophet models...")
print("Evaluation Metrics:")
print("{'Slrp': {'MSE_raw': 23219139.299191058, 'MSE_round': 23219153.478376098, 'RMSE_raw': 4818.624212282076, 'RMSE_round': 4818.625683571624}}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prophet with Features

# COMMAND ----------

## Features
date_col = 'datetime'
actual_col= 'power_W'
station = 'Slrp'
df = slrp_feat.drop(columns = ['station'])

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(df, 0.7)

prophet_model2, prophet_testdf2, prophet_Evals2 = run_prophet(slrp_prophet_train, slrp_prophet_test, date_col, actual_col, station, False)

plot_predsActuals(prophet_testdf2, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM -- No Streaming

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single Step Predictions, `n_outputs=1`

# COMMAND ----------

## No features
date_col = 'datetime'
actualcol= 'power_W'
station = 'Slrp'
n_inputs = 12
n_outputs = 1
loc_ = 'Berkeley'

df = slrp_feat[[date_col, actualcol]]
lstm1_slrp_metrics = dict()


##### split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)

#### split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

### run model
lstm1_modelslrp = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

### plot and get results df and metrics
df_out, evals = plot_predictions(lstm1_modelslrp, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture agg metrics
lstm1_slrp_metrics.update(evals)

#### add additional dataframe columns for visualizations
# df['Location'] = loc_
# df['SiteName'] = station
# df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm1_slrp_metrics)
df.head()

#{'Slrp': {'MSE_Raw': 0.14015938871232805, 'MSE_round': 0.16337522441651706, 'RMSE_Raw': 0.3743786702155026, 'RMSE': 0.404197011884696}}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Six Step Predictions, `n_outputs=6`

# COMMAND ----------

## No features
date_col = 'datetime'
actualcol= 'power_W'
station = 'Slrp'
n_inputs = 12
n_outputs = 6
loc_ = 'Berkeley'

df = slrp_feat[[date_col, actualcol]]
lstm6_slrp_metrics = dict()


# split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)

# split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

lstm6_modelslrp = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

df_out, evals = plot_predictions(lstm6_modelslrp, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
lstm6_slrp_metrics.update(evals)

# add additional dataframe columns for visualizations
# df['Location'] = loc_
# df['SiteName'] = station
# df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm6_slrp_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Six Step Predictions with Features, `n_outputs=6`

# COMMAND ----------

## features
date_col = 'datetime'
actualcol= 'power_W'
station = 'Slrp'
n_inputs = 12
n_outputs = 6
loc_ = 'Berkeley'

df = slrp_feat.drop(columns = ['station'])
lstm6_slrp_metrics2 = dict()


# split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)

# split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

lstm6_modelslrp2 = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

df_out, evals = plot_predictions(lstm6_modelslrp2, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
lstm6_slrp_metrics2.update(evals)

# add additional dataframe columns for visualizations
df_out['Location'] = loc_
df_out['SiteName'] = station
df_out['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm6_slrp_metrics2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Stations Model Results

# COMMAND ----------

slrp_1StepNoFeatures_ports = spark.read.parquet(f"/mnt/{mount_name}/data/batch/Berkeley_LSTM_1StepNoFeatures")
slrp_1StepNoFeatures_ports = slrp_1StepNoFeatures_ports.toPandas()

# COMMAND ----------

merged = slrp_1StepNoFeatures_ports.merge(slrp, left_on = 'DateTime', right_on = 'datetime', how = 'left')
merged['power_pred'] = (8-merged['Predictions']) * 6000
merged['power_pred_round'] = (8-merged['Predictions_(rounded)']) * 6000
merged.head()

# COMMAND ----------

print(calc_rmse(merged['power_pred'], merged))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LSTM -- Streaming

# COMMAND ----------

# MAGIC %run "./retrainer_power"

# COMMAND ----------

seed_model = train_seed_model(slrp_seed)

# COMMAND ----------

model_to_use = seed_model
total_data = slrp_seed.copy()
prediction_results = pd.DataFrame(data = {'datetime': [], 'station': [], 'power_W': [], 'predicted': [],
                                          'predicted_rounded': [], 'stream_count': [], 'timesteps_out': []})
predict_flag = 0
retrain_flag = 0

for row in slrp_test.index:
    
    total_data = total_data.append(slrp_test.loc[row,:]).reset_index(drop = True)
    
    timestep = slrp_test.loc[row, 'datetime']
    
    if timestep.minute == 0:
        predict_flag = 1
        print(f'{timestep}\tmake prediction')
    if (timestep.day == 15) and (timestep.hour == 0) and (timestep.minute == 0):
        retrain_flag = 1
        print(f'{timestep}\tretrain')
        
    if retrain_flag:
        retrain_df = total_data[total_data["datetime"] >= total_data["datetime"].max() - dt.timedelta(days=90)]
        model_to_use = train_model(retrain_df)
        
    if predict_flag:
        prediction_results = prediction_results.append(make_predictions(model_to_use, slrp_seed, slrp_test.head(row+1), slrp))
        prediction_results = prediction_results.reset_index(drop = True)
        
    predict_flag = 0
    retrain_flag = 0
    

# COMMAND ----------

save_final_results_to_s3(prediction_results)

# COMMAND ----------

