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
# MAGIC # Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed Data

# COMMAND ----------

slrp_seed = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_seed")
slrp_seed = slrp_seed.toPandas()
slrp_seed.head()

# COMMAND ----------

palo_alto_seed = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_seed")
palo_alto_seed = palo_alto_seed.toPandas()
palo_alto_seed.head()

# COMMAND ----------

boulder_seed = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_seed")
boulder_seed = boulder_seed.toPandas()
boulder_seed.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6-Month Data

# COMMAND ----------

slrp_6m = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_6m")
slrp_6m = slrp_6m.toPandas()
slrp_6m.head()

# COMMAND ----------

palo_alto_6m = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_6m")
palo_alto_6m = palo_alto_6m.toPandas()
palo_alto_6m.head()

# COMMAND ----------

boulder_6m = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_6m")
boulder_6m = boulder_6m.toPandas()
boulder_6m.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Reference Functions

# COMMAND ----------

def calc_rmse(preds, actuals):
    """
    Calculate the RMSE between predictions and the actual values
    preds: series/array of predictions
    df: dataframe with column 'ports_available' to be used in calculation
    """
    
    return np.sqrt(mse(preds, actuals['ports_available']))

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Overall Average

# COMMAND ----------

def predict_average_overall(df, n_out):
    """
    Use the entire training set to make predictions of ports available
    """
    return [np.round(df['ports_available'].mean())] * n_out

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
    train_set = list(df.tail(n_in)['ports_available'])
    
    # Define list for the predictions
    preds = []
    
    # For each prediction you want to make
    for i in range(n_out):
        # Make the prediction based on the mean of the train set
        prediction = np.round(np.mean(train_set))
        
        # Update the predictions list
        preds.append(prediction)
        
        # Update the training set by using the prediction from the last timestep and dropping the first timestep
        train_set.append(prediction)
        train_set.pop(0)
    
    return preds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Average by Day/Hour

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
    df_grouped = df_grouped.rename(columns = {'ports_available': 'prediction'})
    df_grouped = df_grouped.round({'prediction': 0})
    df_grouped = df_grouped.reset_index()
    
    df_preds = df_test_mod.merge(df_grouped, how = 'left', on = ['day_of_week', 'hour'])
    
    return df_preds['prediction']

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## SlrpEV

# COMMAND ----------

# DBTITLE 1,Benchmark Model 1: Predict Average of All Training Data Only
slrp_test = slrp_6m[slrp_6m['datetime'] > slrp_seed['datetime'].max()]
slrp_preds_1 = predict_average_overall(slrp_seed, slrp_test.shape[0])
slrp_rmse_1 = calc_rmse(slrp_preds_1, slrp_test)
print(slrp_rmse_1)

# COMMAND ----------

## Using two weeks as testing data only
slrp_preds_1 = predict_average_overall(slrp_seed, 6*24*14)
slrp_rmse_1 = calc_rmse(slrp_preds_1, slrp_test.head(6*24*14))
print(slrp_rmse_1)

# COMMAND ----------

# plt.subplots(figsize = (10,5))
# plt.plot(slrp_test['datetime'], slrp_test['ports_available'], label = 'actual');
# plt.plot(slrp_test['datetime'], slrp_preds_1, label = 'predictions');
# plt.xlabel('Datetime');
# plt.ylabel('Ports Available');
# plt.title(f'SlrpEV Benchmark 1: Predict Average of All Training Data  |  RMSE = {np.round(slrp_rmse_1, 4)}');
# plt.legend();

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2A: Predict Average of Last n Timestamps (No Streaming)
n_in = 12
n_out = slrp_test.shape[0]
slrp_preds_2a = predict_average_n_timestamps(slrp_seed, n_in, n_out)
slrp_rmse_2a = calc_rmse(slrp_preds_2a, slrp_test.head(n_out))
print(slrp_rmse_2a)

# COMMAND ----------

## Using two weeks as testing data only
slrp_preds_2a = predict_average_n_timestamps(slrp_seed, n_in, 6*24*14)
slrp_rmse_2a = calc_rmse(slrp_preds_2a, slrp_test.head(6*24*14))
print(slrp_rmse_2a)

# COMMAND ----------

slrp_preds_2a

# COMMAND ----------

# plt.subplots(figsize = (10,5))
# plt.plot(slrp_test['datetime'], slrp_test['ports_available'], label = 'actual');
# plt.plot(slrp_test['datetime'], slrp_preds_2a, label = 'predictions');
# plt.xlabel('Datetime');
# plt.ylabel('Ports Available');
# plt.title(f'SlrpEV Benchmark 2A: Predict Average of All Training Data  |  RMSE = {np.round(slrp_rmse_2a, 4)}');
# plt.legend();

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
plt.plot(subset['datetime'], subset['ports_available'], label = 'actual');
plt.plot(subset['datetime'], subset['predicted'], label = 'predictions');
plt.xlabel('Datetime');
plt.ylabel('Ports Available');
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

month_list = slrp_6m['datetime'].dt.month.unique()
weights = []
rmses = []

for month in range(3, len(slrp_6m['datetime'].dt.month.unique())):
    train_temp = slrp_6m[(slrp_6m['datetime'].dt.month == month_list[month - 3]) | 
                         (slrp_6m['datetime'].dt.month == month_list[month - 2]) |
                         (slrp_6m['datetime'].dt.month == month_list[month - 1])]
    test_temp = slrp_6m[(slrp_6m['datetime'].dt.month == month_list[month])]
    weights.append(test_temp.shape[0])
    slrp_preds_3b = predict_avg_by_day_hour(train_temp, test_temp)
    slrp_rmse_3b = calc_rmse(slrp_preds_3b, test_temp)
    rmses.append(slrp_rmse_3b)
print(sum([rmses[i] * weights[i] for i in range(len(rmses))]) / sum(weights))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

# DBTITLE 1,Benchmark Model 1: Predict Average of All Training Data Only
palo_alto_rmses_1 = []
for station in palo_alto_seed['station'].unique():
    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]
    palo_alto_preds_1 = predict_average_overall(palo_alto_seed_selected, palo_alto_test.shape[0])
    palo_alto_rmse_1 = calc_rmse(palo_alto_preds_1, palo_alto_test)
    if len(station) < 7:
        print(f'{station}:\t\t{palo_alto_rmse_1}')
        palo_alto_rmses_1.append(palo_alto_rmse_1)
    else:
        print(f'{station}:\t{palo_alto_rmse_1}')
        palo_alto_rmses_1.append(palo_alto_rmse_1)

print(f"Mean:\t\t{np.mean(palo_alto_rmses_1)}")

# COMMAND ----------

# Test on two weeks only
palo_alto_rmses_1 = []
for station in palo_alto_seed['station'].unique():
    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]\
                     .head(6*24*14)
    palo_alto_preds_1 = predict_average_overall(palo_alto_seed_selected, palo_alto_test.shape[0])
    palo_alto_rmse_1 = calc_rmse(palo_alto_preds_1, palo_alto_test)
    if len(station) < 7:
        print(f'{station}:\t\t{palo_alto_rmse_1}')
        palo_alto_rmses_1.append(palo_alto_rmse_1)
    else:
        print(f'{station}:\t{palo_alto_rmse_1}')
        palo_alto_rmses_1.append(palo_alto_rmse_1)

print(f"Mean:\t\t{np.mean(palo_alto_rmses_1)}")

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2A: Predict Average of Last n Timestamps (No Streaming)
n_in = 12

for station in palo_alto_seed['station'].unique():

    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]
    
    n_out = palo_alto_test.shape[0]
    palo_alto_preds_2a = predict_average_n_timestamps(palo_alto_seed_selected, n_in, n_out)
    palo_alto_rmse_2a = calc_rmse(palo_alto_preds_2a, palo_alto_test.head(n_out))
    if len(station) < 7:
        print(f'{station}:\t\t{palo_alto_rmse_2a}')
    else:
        print(f'{station}:\t{palo_alto_rmse_2a}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2B: Predict Average of Last n Timestamps (With Streaming)
for station in palo_alto_seed['station'].unique():

    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]
    
    
    streaming_frequency = 6 # get streaming updates each hour with values from the past hour
    n_in = 12
    n_out = streaming_frequency
    results = palo_alto_test.copy().reset_index(drop = True)
    results['predicted'] = ['']*results.shape[0]
#     all_rmses = []

    for i in range(int(np.ceil(palo_alto_test.shape[0] / streaming_frequency))):
        palo_alto_preds_2b = predict_average_n_timestamps(pd.concat([palo_alto_seed_selected, 
                                                                palo_alto_test.head(streaming_frequency*i)]), n_in, n_out)
#         all_rmses.append(calc_rmse(palo_alto_preds_2b, palo_alto_test.iloc[i:i+n_out,:]))
        for pred_num in range(n_out):
            results.loc[i*n_out + pred_num, 'predicted'] = palo_alto_preds_2b[pred_num]
    results = results.dropna()
    palo_alto_rmse_2b = calc_rmse(results['predicted'], results)
    if len(station) < 7:
        print(f'{station}:\t\t{palo_alto_rmse_2b}')
    else:
        print(f'{station}:\t{palo_alto_rmse_2b}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3A: Predict Average by Day of Week and Hour (No Streaming)
for station in palo_alto_seed['station'].unique():

    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]
    
    palo_alto_preds_3 = predict_avg_by_day_hour(palo_alto_seed_selected, palo_alto_test)
    palo_alto_rmse_3 = calc_rmse(palo_alto_preds_3, palo_alto_test)
    if len(station) < 7:
        print(f'{station}:\t\t{palo_alto_rmse_3}')
    else:
        print(f'{station}:\t{palo_alto_rmse_3}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3B: Predict Average by Day of Week and Hour (With Streaming)
# Retrain Monthly

month_list = palo_alto_6m['datetime'].dt.month.unique()

for station in palo_alto_seed['station'].unique():

    palo_alto_seed_selected = palo_alto_seed[palo_alto_seed['station'] == station]
    palo_alto_6m_selected = palo_alto_6m[palo_alto_6m['station'] == station]
    
    palo_alto_test = palo_alto_6m_selected[palo_alto_6m_selected['datetime'] > palo_alto_seed_selected['datetime'].max()]

    weights = []
    rmses = []

    for month in range(3, len(palo_alto_6m_selected['datetime'].dt.month.unique())):
        train_temp = palo_alto_6m_selected[(palo_alto_6m_selected['datetime'].dt.month == month_list[month - 3]) | 
                                           (palo_alto_6m_selected['datetime'].dt.month == month_list[month - 2]) |
                                           (palo_alto_6m_selected['datetime'].dt.month == month_list[month - 1])]
        test_temp = palo_alto_6m_selected[(palo_alto_6m_selected['datetime'].dt.month == month_list[month])]
        weights.append(test_temp.shape[0])
        palo_alto_preds_3b = predict_avg_by_day_hour(train_temp, test_temp)
        palo_alto_rmse_3b = calc_rmse(palo_alto_preds_3b, test_temp)
        rmses.append(palo_alto_rmse_3b)
    weighted_rmse_3b = sum([rmses[i] * weights[i] for i in range(len(rmses))]) / sum(weights)
    if len(station) < 7:
        print(f'{station}:\t\t{weighted_rmse_3b}')
    else:
        print(f'{station}:\t{weighted_rmse_3b}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

# DBTITLE 1,Benchmark Model 1: Predict Average of All Training Data Only
boulder_rmses_1 = []
for station in boulder_seed['station'].unique():
    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    boulder_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]
    boulder_preds_1 = predict_average_overall(boulder_seed_selected, boulder_test.shape[0])
    boulder_rmse_1 = calc_rmse(boulder_preds_1, boulder_test)
    if len(station) < 15:
        print(f'{station}:\t\t{boulder_rmse_1}')
        boulder_rmses_1.append(boulder_rmse_1)
    else:
        print(f'{station}:\t{boulder_rmse_1}')
        boulder_rmses_1.append(boulder_rmse_1)
        
print(f"Mean:\t\t\t{np.mean(boulder_rmses_1)}")

# COMMAND ----------

boulder_rmses_1 = []
for station in boulder_seed['station'].unique():
    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    boulder_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]\
                   .head(6*24*14)
    boulder_preds_1 = predict_average_overall(boulder_seed_selected, boulder_test.shape[0])
    boulder_rmse_1 = calc_rmse(boulder_preds_1, boulder_test)
    if len(station) < 15:
        print(f'{station}:\t\t{boulder_rmse_1}')
        boulder_rmses_1.append(boulder_rmse_1)
    else:
        print(f'{station}:\t{boulder_rmse_1}')
        boulder_rmses_1.append(boulder_rmse_1)
        
print(f"Mean:\t\t\t{np.mean(boulder_rmses_1)}")

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2A: Predict Average of Last n Timestamps (No Streaming)
n_in = 12

for station in boulder_seed['station'].unique():
    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    boulder_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]
    
    n_out = boulder_test.shape[0]
    boulder_preds_2a = predict_average_n_timestamps(boulder_seed_selected, n_in, n_out)
    boulder_rmse_2a = calc_rmse(boulder_preds_2a, boulder_test.head(n_out))
    if len(station) < 15:
        print(f'{station}:\t\t{boulder_rmse_2a}')
    else:
        print(f'{station}:\t{boulder_rmse_2a}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 2B: Predict Average of Last n Timestamps (With Streaming)
for station in boulder_seed['station'].unique():
    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    boulder_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]
    
    
    streaming_frequency = 6 # get streaming updates each hour with values from the past hour
    n_in = 12
    n_out = streaming_frequency
    results = boulder_test.copy().reset_index(drop = True)
    results['predicted'] = ['']*results.shape[0]
#     all_rmses = []

    for i in range(int(np.ceil(boulder_test.shape[0] / streaming_frequency))):
        boulder_preds_2b = predict_average_n_timestamps(pd.concat([boulder_seed_selected, 
                                                                boulder_test.head(streaming_frequency*i)]), n_in, n_out)
#         all_rmses.append(calc_rmse(palo_alto_preds_2b, palo_alto_test.iloc[i:i+n_out,:]))
        for pred_num in range(n_out):
            results.loc[i*n_out + pred_num, 'predicted'] = boulder_preds_2b[pred_num]
    
    results = results.dropna()
    boulder_rmse_2b = calc_rmse(results['predicted'], results)
    if len(station) < 15:
        print(f'{station}:\t\t{boulder_rmse_2b}')
    else:
        print(f'{station}:\t{boulder_rmse_2b}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3A: Predict Average by Day of Week and Hour (No Streaming)
for station in boulder_seed['station'].unique():
    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    boulder_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]
    
    boulder_preds_3 = predict_avg_by_day_hour(boulder_seed_selected, boulder_test)
    boulder_rmse_3 = calc_rmse(boulder_preds_3, boulder_test)
    if len(station) < 15:
        print(f'{station}:\t\t{boulder_rmse_3}')
    else:
        print(f'{station}:\t{boulder_rmse_3}')

# COMMAND ----------

# DBTITLE 1,Benchmark Model 3B: Predict Average by Day of Week and Hour (With Streaming)
# Retrain Monthly

month_list = boulder_6m['datetime'].dt.month.unique()

for station in boulder_seed['station'].unique():

    boulder_seed_selected = boulder_seed[boulder_seed['station'] == station]
    boulder_6m_selected = boulder_6m[boulder_6m['station'] == station]
    
    palo_alto_test = boulder_6m_selected[boulder_6m_selected['datetime'] > boulder_seed_selected['datetime'].max()]

    weights = []
    rmses = []

    for month in range(3, len(boulder_6m_selected['datetime'].dt.month.unique())):
        train_temp = boulder_6m_selected[(boulder_6m_selected['datetime'].dt.month == month_list[month - 3]) | 
                                           (boulder_6m_selected['datetime'].dt.month == month_list[month - 2]) |
                                           (boulder_6m_selected['datetime'].dt.month == month_list[month - 1])]
        test_temp = boulder_6m_selected[(boulder_6m_selected['datetime'].dt.month == month_list[month])]
        weights.append(test_temp.shape[0])
        boulder_preds_3b = predict_avg_by_day_hour(train_temp, test_temp)
        boulder_rmse_3b = calc_rmse(boulder_preds_3b, test_temp)
        rmses.append(boulder_rmse_3b)
    weighted_rmse_3b = sum([rmses[i] * weights[i] for i in range(len(rmses))]) / sum(weights)
    if len(station) < 15:
        print(f'{station}:\t\t{weighted_rmse_3b}')
    else:
        print(f'{station}:\t{weighted_rmse_3b}')

# COMMAND ----------

