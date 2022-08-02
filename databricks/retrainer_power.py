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
import folium
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

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# def arima_filter(df, start_date, end_date, date_col):
#     ''' 
#     filter data frame to be between the start date and end date, not including the end date. 
#     date_col is the date column used to filter the dataframe
#     '''
#     return df[(df[date_col] >= start_date) & (df[date_col] < end_date)]

  
# def dfSplit_Xy(df, date_col='datetime', n_input=6, n_out=36):
#     """ 
#     Tranform pandas dataframe into arrays of inputs and outputs. 
#     The output (value predicting) must be located at the end of the df
#     n_inputs, is the the number of inputs, to use to predict n_input+1 (or i + n_input) aka window size
#     n_outs, is the number of steps out you want to predict. Default is 1
#     Returns 2 numpy arrays. Inputs (features), and actual labels
#     """
    
#     ind_dates = df[date_col].to_numpy() ####
#     df_as_np = df.set_index(date_col).to_numpy()
    
#     #X is matrix of inputs
#     #y is actual outputs, labels
#     X = []
#     y = []
#     y_dates = [] ####
    
    
#     for i in range(len(df_as_np)): 
#         #print(i)
#         start_out = i + n_input
#         start_end = start_out + n_out

#         # to make sure we always have n_out values in label array
#         if start_end > len(df_as_np):
#             break

#         #take the i and the next values, makes a list of list for multiple inputs
#         row = df_as_np[i:start_out, :]
#         #print(row)
#         X.append(row)

#         # Creating label outputs extended n_out steps. -1, last column is label
#         label = df_as_np[start_out:start_end, -1]
#         #print(label)
#         y.append(label)
        
#         # array of dates
#         label_dates = ind_dates[start_out:start_end]####
#         #print(label_dates)###
#         y_dates.append(label_dates) #### 
    
#     return np.array(X), np.array(y), np.array(y_dates)


def split_train_test(X, y):
    '''
    Split inputs and labels into train, validation and test sets for LSTMs
    Returns x and y arrays for train, validation, and test.
    '''
    
    dev_prop, train_prop = 14/90, 1 - (14/90)
    
    # get size of Array
    num_timestamps = X.shape[0]
    
    # define split points
    train_start = 0 # do we need this? can we always start at 0?
    train_end = int(num_timestamps * train_prop)
    dev_end = int(num_timestamps * (train_prop + dev_prop))
    
    # splitting
    X_train, y_train = X[train_start:train_end], y[train_start:train_end]#, y_dates[train_start:train_end]
    X_val, y_val = X[train_end:dev_end], y[train_end:dev_end]#, y_dates[train_end:dev_end]
    # include dates for plotting later
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_train.shape[0])
    
    return X_train, y_train, X_val, y_val


### SAME MODEL IS ALREADY DEFINED IN THE MODELING_POWER NOTEBOOK
# def run_lstm(n_inputs, n_features, n_outputs, X_train, y_train, X_val, y_val, n_epochs = 10):
#     '''Run lstm model, and get fitted model'''
    
#     # Build LSTM Model
#     model = Sequential()
#     model.add(InputLayer((n_inputs, n_features))) 
#     model.add(LSTM(64, input_shape = (n_inputs, n_features), activation = 'relu'))
#     model.add(Dense(8, 'relu'))
#     model.add(Dense(n_outputs, 'linear'))
#     model.summary()
    
#     ### checkpoints to save best model
#     cp = ModelCheckpoint('model/', save_best_only=True)
#     model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    
#     ## Fit model
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, callbacks=[cp])
    
#     # load model in the event the last epoch is not the best model, if it is will give same results if skip
#     model = load_model('model/')
    
#     return model
  
    
# def write_model_to_s3(model, model_name):
    
#     bucket='w210v2'
#     key=f'models/power/{model_name}.pkl'
    
#     client = boto3.client("s3",
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )

#     pickle_byte_obj = pickle.dumps(model) 

#     client.put_object(Key=key, Body=pickle_byte_obj, Bucket="w210v2")
    
    
# def write_seed_model_to_s3(model, model_name):
    
#     bucket='w210v2'
#     key=f'models/power/seed_models/{model_name}.pkl'
    
#     client = boto3.client("s3",
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )

#     pickle_byte_obj = pickle.dumps(model) 

#     client.put_object(Key=key, Body=pickle_byte_obj, Bucket="w210v2")
    
    
# def get_model_from_s3(model_name):
    
#     bucket='w210v2'
#     key=f'models/power/{model_name}.pkl'

#     client = boto3.client("s3",
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )
    
#     response = client.get_object(Bucket=bucket,Key=key)

#     model = pickle.loads(response['Body'].read())
#     return model

# COMMAND ----------

# def save_fig_to_s3(fig_name):
#     bucket='w210v2'
#     key=f'figures/{fig_name}.png'
    
#     client = boto3.client("s3",
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#     )
#     plt.plot(range(10), range(10))
#     img_data = io.BytesIO()
#     plt.savefig(img_data, format='png')
#     img_data.seek(0)
#     client.put_object(Key=key, Body=img_data, Bucket="w210v2", ContentType = 'image/png')

# COMMAND ----------

# DBTITLE 1,Write to S3 Buckets
# # write results to s3 bucket

# def write_6m_to_s3(df, save_filename):

#     # create spark data frame
#     results_ab = spark.createDataFrame(df) 

#     ## Write to AWS S3
#     (results_ab
#         .repartition(1)
#         .write
#         .format("parquet")
#         .mode("overwrite")
#         .save(f"/mnt/{mount_name}/data/{save_filename}"))

# COMMAND ----------

# # write results to s3 bucket

# def write_seed_to_s3(df, save_filename):

#     # create spark data frame
#     results_ab = spark.createDataFrame(df) 

#     ## Write to AWS S3
#     (results_ab
#         .repartition(1)
#         .write
#         .format("parquet")
#         .mode("overwrite")
#         .save(f"/mnt/{mount_name}/data/{save_filename}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # AWS Setup

# COMMAND ----------

# ## Connect to AWS
# access_key = "AKIA3CS2H32VF7XY33S2"
# secret_key = "0ZgHc4WyyfQn7uylzrSSdjPwIgJpvukdQZysZWWI"
# encoded_secret_key = secret_key.replace("/", "%2F")
# aws_bucket_name = "w210v2"
# mount_name = "w210v2"

# try:
#     dbutils.fs.mount("s3a://%s:%s@%s" % (access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
# except Exception as e:
#     print("Already mounted :)")

# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Model on Seed Data

# COMMAND ----------

## Define timestep constants
n_inputs = 12
n_outputs = 36

date_col = "datetime"
station_col = "station"

# models_dir_entries = dbutils.fs.ls(f"/mnt/{mount_name}/models/power")
# seed_model_dir_entries = dbutils.fs.ls(f"/mnt/{mount_name}/models/power/seed_models/")

## If there is more than just the "seed_models" folder in the "models" directory
# if len(models_dir_entries) > 1:
    
#     ## Loop through all filesystem objects within the models directory
#     for models_dir_entry in models_dir_entries:
        
#         ## Delete existing model files
#         if models_dir_entry.isFile():
#             try:
#                 dbutils.fs.rm(models_dir_entry.path)
#             except:
#                 print(f"Couldn't delete: {models_dir_entry.path}")
        
#     ## Copy seed models back into directory
#     for seed_model_dir_entry in seed_model_dir_entries:
#         dbutils.fs.cp(seed_model_dir_entry.path, f"/mnt/{mount_name}/models/power/")

        
def train_seed_model(seed_df):

    ## Drop the station column
    seed_df = seed_df.drop(columns = ['station'])

    ## Split the data
    X, y, dates = dfSplit_Xy(seed_df, date_col="datetime", n_input=n_inputs, n_out=n_outputs)

    ## Split the data into train, validation -- use the last two weeks of the data as the validation set
    X_train, y_train, X_val, y_val = split_train_test(X, y)
    # print(f"2nd: {X_train}, {y_train}, {X_val}, {y_val}")

    ## Fit the model
    model = run_lstm(n_inputs=n_inputs, 
                     n_features=X_train.shape[2],
                     n_outputs=n_outputs, 
                     X_train=X_train,
                     y_train=y_train,
                     X_val=X_val, 
                     y_val=y_val, 
                     n_epochs = 10)

    return model
        
        
def train_model(station_df):

    ## Get a list of all the unique stations 
    stations = list(station_df[station_col].unique())
    if len(stations) > 1:
        return "Too many stations."
    station_name = stations[0]

    station_df = station_df.drop(station_col, axis=1)

    ## Split the data
#         global X
    X, y, dates = dfSplit_Xy(station_df, date_col="datetime", n_input=n_inputs, n_out=n_outputs)
    # print(f"1st: {X}, {y}, {dates}")

    ## Split the data into train, validation, (and test)
#         global X_val
    X_train, y_train, X_val, y_val = split_train_test(X, y)
    # print(f"2nd: {X_train}, {y_train}, {X_val}, {y_val}")

    ## Fit the model
    model = run_lstm(n_inputs=n_inputs, 
                     n_features=X_train.shape[2],
                     n_outputs=n_outputs, 
                     X_train=X_train,
                     y_train=y_train,
                     X_val=X_val, 
                     y_val=y_val, 
                     n_epochs = 10)

    return model

# COMMAND ----------

# MAGIC %md 
# MAGIC # Make Predictions from LSTM Model

# COMMAND ----------

def make_predictions(model, location_df, streamed_df, location_df_6m):
    """
    model: LSTM model to make predictions from
    location_df: training data
    streamed_data: the data that has been streamed so far
    location_df_6m: all data (training + streaming)
    
    Make predictions from an LSTM model by getting the station information from the last `n_inputs` timestamps
    and running it through the model for predictions
    
    Use the location_df_6m dataframe to calculate the errors (MSE, RMSE) of the predictions
    
    This function assumes that we are only predicting using a single feature (no time-based features)
    """
    
    # For testing, concatenate the original dataframe (seed) with the streamed data, drop the station column
    station_historical = pd.concat([location_df, streamed_df]).sort_values(by = 'datetime').drop(columns = ['station'])

    # Develop a testing point from which to predict out n_output time steps
    test = np.array(station_historical.set_index('datetime', drop = True).tail(n_inputs))
    test.resize(1, n_inputs, test.shape[1]) # 1 record, number of timesteps back (can also be test.shape[0]), number of features

    # Make predictions from the model
    predictions = model.predict(test)[0]
    predictions_rounded = np.clip(predictions, 0, 8 * 10000)

    # Get the actual values from the 6 month data to calculate the MSE
    actuals = location_df_6m[(location_df_6m['datetime'] > station_historical['datetime'].max()) &
                             (location_df_6m['datetime'] <= station_historical['datetime'].max() + 
                              dt.timedelta(minutes = 10*n_outputs))]

    # Add the predictions to this data frame
    actuals['predicted'] = predictions
    actuals['predicted_rounded'] = predictions_rounded
    actuals = actuals.reset_index(drop = True)

    # Evaluation Metrics
    MSE_raw = mse(actuals['power_W'], actuals['predicted'])
    MSE_rounded = mse(actuals['power_W'], actuals['predicted_rounded'])
    RMSE_raw = math.sqrt(MSE_raw)
    RMSE_rounded = math.sqrt(MSE_rounded)

    timestep_errors = pd.DataFrame(data = {'datetime': actuals['datetime'],
                                           'station': [station]*n_outputs,
                                           'power_W': actuals['power_W'],
                                           'predicted': actuals['predicted'],
                                           'predicted_rounded': actuals['predicted_rounded'],
                                           'stream_count': streamed_df.shape[0],
                                           'timesteps_out': range(1, n_outputs+1)})
    
    return timestep_errors

# COMMAND ----------

## Save Results to S3
def save_final_results_to_s3(df):
    df_spark = spark.createDataFrame(df)
    (df_spark.repartition(1)
        .write
        .format("csv")
        .mode("overwrite")
        .save(f"/mnt/{mount_name}/data/streaming_predictions/slrp_pred_power_errors"))

# COMMAND ----------

