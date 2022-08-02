# Databricks notebook source
# MAGIC %md
# MAGIC # Modeling
# MAGIC 
# MAGIC ARIMA, Prophet, and LSTM

# COMMAND ----------

# MAGIC %md
# MAGIC # Import & Setup

# COMMAND ----------

pip install markupsafe==2.0.1

# COMMAND ----------

# import libraries
import pandas as pd
import numpy as np
import json
import math
import datetime as dt
import time
import holidays
from calendar import monthrange
from datetime import timedelta
import urllib
import requests
import seaborn as sns
from pandas.io.json import json_normalize
import pytz
import warnings
import altair as alt
from altair import datum
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
import boto3
import pyspark.sql.functions as F
import pickle

from statsmodels.tsa.seasonal import seasonal_decompose


############## Modeling Libraries  #####################

### Model Evaluation
from sklearn.metrics import mean_squared_error as mse

#### ARIMA/VAR Models
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

#### LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from keras.models import Model, Sequential

#### Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet


#### hide warnings
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)
##
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)


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
    
### display aws
display(dbutils.fs.ls(f"/mnt/{mount_name}/data/"))

# COMMAND ----------

# DBTITLE 1,General Functions
############## Filter Data ####################
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



def arima_filter(df, start_date, end_date, date_col):
    ''' 
    filter data frame to be between the start date and end date, not including the end date. 
    date_col is the date column used to filter the dataframe
    '''
    return df[(df[date_col] >= start_date) & (df[date_col] < end_date)] 



### split into train dev and test
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
    
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_train.shape[0]




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
    
    
    
    
#### Write to s3 ####
def write_to_s3(df, save_filename):

    # create spark data frame
    results_ab = spark.createDataFrame(df)
    
    # replace any spaces
    cols = results_ab.columns
    for col in cols:
        results_ab = results_ab.withColumnRenamed(col, col.replace(" ", "_"))
    

    ## Write to AWS S3
    (results_ab
        .repartition(1)
        .write
        .format("parquet")
        .mode("overwrite")
        .save(f"/mnt/{mount_name}/data/batch/{save_filename}"))
    
    
def write_model_to_s3(model, model_name):
    
    bucket='w210v2'
    key=f'models/{model_name}.pkl'
    
    client = boto3.client("s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    pickle_byte_obj = pickle.dumps(model) 

    client.put_object(Key=key, Body=pickle_byte_obj, Bucket="w210v2")
    
    

    
def dictstodf(dict1, dict2, dict3, modeltype, features='No'):
    emptyd = dict()
    emptyd.update(dict1)
    emptyd.update(dict2)
    emptyd.update(dict3)
    
    
    df = pd.DataFrame.from_dict({(i,j): emptyd[i][j] 
                           for i in emptyd.keys() 
                           for j in emptyd[i].keys()},
                       orient='index')

    df.reset_index(inplace=True)

    df = df.rename(columns={"level_0":"city", "level_1":"station", "RMSE":"RMSE_round"})
    df['Model'] = str(modeltype)
    df['Features'] = str(features)
    
    print(df.shape)
    return df

# COMMAND ----------

# DBTITLE 1,Model Functions
################################# ARIMA FUNCTIONS ##################################################
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
        
###### ARIMA Model   #### need to add in features here possibly in an if statement
def run_arima(traindf, testdf, actualcol, date_col, station, no_features = True):
    '''
    run arima model given a training df, a testing df. 
    as a string provide the name of the actualcol aka output column, the date column, 
    and the station name
    '''
    print(station)
    #### new
    traindf.set_index(date_col, drop=False, inplace=True)
    testdf.set_index(date_col, drop = False, inplace = True)
    
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
        testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'}, inplace = True)

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
        testdf.rename(columns={actualcol: "Actuals", date_col: 'DateTime'}, inplace = True)

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

#############################################################################################################





######################################## PROPHET ##############################################################

### how do I add features here???
## output model, dataframe with dt, actuals, predictions, and predictions rounded, metrics dict
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


#### PROPHET MODEL
def run_prophet_boulder(df, dt_col, station_col, output_col, loc, nofeatures = True):
    
    #collect df vales and metrics
    results_df = pd.DataFrame()
    metrics = dict()
    
    for station in df[station_col].unique():
        
        #get station df
        stationdf = df[df[station_col] == station]
        stationdf = stationdf.drop([station_col], axis = 1)
        
        ## split data
        station_train, station_test = split2_TrainTest(stationdf, 0.7)
        
        ## run model
        prophet_model, prophet_testdf, prophet_Evals = run_prophet(station_train, station_test, dt_col, output_col, station, nofeatures)
        
        ### write model to s3
        if nofeatures:
            modelname = 'BatchModel_'+ str(loc)+str(station)+"_Prophet"
        else: 
            modelname = 'BatchModel_'+ str(loc)+str(station)+"_Prophet_Features"
        write_model_to_s3(prophet_model, modelname)
        
        ## plot
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
#########################################################################################



################################# LSTMs ####################################################
# def select_months(df, date_col, train_months, start_date = '2021-01-01'):
#     """
#     Filter the dataframe to only a certain number of months, after start date
    
#     date_col is the column of the dataframe with the datetime info
    
#     start_date should have format of YYYY-MM-DD
    
#     returns the filtered dataframe
#     """
    
#     # converting start date to date time format
#     split = start_date.split('-')
#     start_year = int(split[0])
#     start_month = int(split[1])
#     start_day = int(split[2])
#     start_date = dt.datetime(year = start_year, month = start_month, day = start_day)
    
#     #total months to pull if train_months represents 70%
#     total_months = train_months / 0.7
    
#     end_date = start_date + dt.timedelta(days = int(total_months * 30))
    
    
#     print(start_date)
#     print(end_date)
    
    
#     # filter df to dates from date_col equal to or after start and before end date
#     temp = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
    
#     return temp


#### get arrays
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



#### split arrays ####
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


#### run LSTM models
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


### plots
def plot_predictions(model, X_test, y_test, train_df, station, train_end, date_col,y_dates, start=0, end=1000):
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
    predictions = model.predict(X_test)
    predictions = predictions.flatten()
    
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

# DBTITLE 1,Benchmark Model functions
def calc_rmse(preds, actuals):
    """
    Calculate the RMSE between predictions and the actual values
    preds: series/array of predictions
    df: dataframe with column 'ports_available' to be used in calculation
    """
    mse_score = mse(preds, actuals)
    rmse_score = np.sqrt(mse_score)
    
    return mse_score, rmse_score


def predict_average_overall(df, n_out, actualcol):
    """
    Use the entire training set to make predictions of ports available
    """
    predictions = [df[actualcol].mean()] * n_out
    roundedpredictions = [np.round(df[actualcol].mean())] * n_out
    
    return predictions, roundedpredictions


def predict_average_n_timestamps(df, n_in, n_out, actualcol):
    """
    Use the last n_in timesteps only to make predictions of ports available for n_out timesteps out
    """
    
    # Get the last n_in entries from the ports available column
    train_set = list(df.tail(n_in)[actualcol])
    
    # Define list for the predictions
    preds = []
    preds_round = []
    
    # For each prediction you want to make
    for i in range(n_out):
        # Make the prediction based on the mean of the train set
        prediction = np.mean(train_set)
        prediction_round = np.round(prediction)
        
        # Update the predictions list
        preds.append(prediction)
        preds_round.append(prediction_round)
        
        # Update the training set by using the prediction from the last timestep and dropping the first timestep
        train_set.append(prediction)
        train_set.pop(0)
    
    return preds, preds_round




def predict_avg_by_day_hour(df, df_test, datecol, actualcol):
    """
    Make predictions based on the day of week and the hour of day -- return the average
    """
    df_mod = df.copy()
    df_test_mod = df_test.copy()
    
    # Add day of week and hour columns
    df_mod.loc[:,'day_of_week'] = df[datecol].dt.dayofweek
    df_mod.loc[:,'Hour'] = df[datecol].dt.hour
    df_test_mod.loc[:,'day_of_week'] = df_test[datecol].dt.dayofweek
    df_test_mod.loc[:,'Hour'] = df_test[datecol].dt.hour
    
    # Group by these features, calculate the mean, rename the column
    df_grouped = df_mod.groupby(['day_of_week', 'Hour']).mean()
    df_grouped = df_grouped.rename(columns = {actualcol: 'Predictions'})
    df_grouped.loc[:,'Predictions_Rounded'] = df_grouped['Predictions']
    df_grouped = df_grouped.round({'Predictions_Rounded': 0})
    df_grouped = df_grouped.reset_index()
    
    df_preds = df_test_mod.merge(df_grouped, how = 'left', on = ['day_of_week', 'Hour'])
    
    return df_preds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Datasets

# COMMAND ----------

slrp = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_ts")
boulder = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_ts_clean")
palo_alto = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_ts")

#convert to Dataframe
slrp = slrp.toPandas()
palo_alto = palo_alto.toPandas()
boulder = boulder.toPandas()


### holidays df	
holiday_list = []
for ptr in holidays.US(years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).items():
    holiday_list.append(ptr)
us_holidays = pd.DataFrame(holiday_list).rename(columns = {0: 'date', 1: 'holiday_name'})
us_holidays['holiday'] = 1

### berkeley
print("SLRP: Min Date: ",slrp['DateTime'].min() ," Max Date: ", slrp['DateTime'].max())

#### boulder
date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder
result = test.groupby('proper_name').agg({date_col: ['min', 'max']})
 
print()
print("Boulder Dates for each stations")
print(result)


### palo alto
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

test = palo_alto
result = test.groupby('Station_Location').agg({date_col: ['min', 'max']})

print()
print("Palo Alto Dates for each stations")
print(result)


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
# MAGIC ## Berkeley, SlrpEV

# COMMAND ----------

# DBTITLE 1,Berkeley, SlrpEV
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

#move output to end of col for LSTMs
slrp_transformed = slrp_transformed[[c for c in slrp_transformed if c not in ['Ports Available']] + ['Ports Available']]

slrp_transformed.head()




# COMMAND ----------

# MAGIC %md
# MAGIC ##Palo Alto

# COMMAND ----------

### Palo Alto ####
# pa_transformed = palo_alto[['Station_Location', 'datetime_pac', 'Ports_Available']].sort_values(by = ['datetime_pac', 'Station_Location']).copy()
# pa_transformed

pa_transformed = palo_alto.copy()
## drop unneeded colums
pa_transformed = pa_transformed.drop(['datetime_utc', 'DateTime', 'Station_Name', 'Port_Type', 'Plug_Type', 'Ports_Perc_Available' , 'Ports_Perc_Occupied', 'Port_Number', 'Ports_Occupied', 'Total_Ports'], axis=1)


## add datetime parts
pa_transformed['Date'] = pa_transformed['datetime_pac'].dt.date
pa_transformed['Month'] = pa_transformed['datetime_pac'].dt.month
pa_transformed['Year'] = pa_transformed['datetime_pac'].dt.year
pa_transformed['Year-Month'] = pa_transformed['Year'].astype(str) + '-' + pa_transformed['Month'].astype(str)
pa_transformed['DayofWeek'] = pa_transformed['datetime_pac'].dt.weekday
pa_transformed['IsWeekend'] = pa_transformed['DayofWeek'] > 4
pa_transformed['Hour'] = pa_transformed['datetime_pac'].dt.hour

## holidays
pa_transformed = pa_transformed.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)


# Apply cosine and sine transformations to cyclical features
pa_transformed['month_cosine'] = np.cos(2 * math.pi * pa_transformed['Month'] / pa_transformed['Month'].max())

pa_transformed['month_sine'] = np.sin(2 * math.pi * pa_transformed['Month'] / pa_transformed['Month'].max())

pa_transformed['hour_cosine'] = np.cos(2 * math.pi * pa_transformed['datetime_pac'].dt.hour / 
                                         pa_transformed['datetime_pac'].dt.hour.max())

pa_transformed['hour_sine'] = np.sin(2 * math.pi * pa_transformed['datetime_pac'].dt.hour / 
                                       pa_transformed['datetime_pac'].dt.hour.max())

pa_transformed['dayofweek_cosine'] = np.cos(2 * math.pi * pa_transformed['DayofWeek'] / 
                                              pa_transformed['DayofWeek'].max())

pa_transformed['dayofweek_sine'] = np.sin(2 * math.pi * pa_transformed['DayofWeek'] / 
                                            pa_transformed['DayofWeek'].max())

pa_transformed['IsWeekend'] = pa_transformed['IsWeekend'].astype(int)


# Drop unnecessary columns
pa_transformed = pa_transformed.drop(columns = ['DayofWeek', 'Month', 'Year', 'Date', 'Year-Month'])

# Sort by DateTime
pa_transformed = pa_transformed.sort_values(by = 'datetime_pac').reset_index(drop=True)

# move output to end of col for LSTMs
pa_transformed = pa_transformed[[c for c in pa_transformed if c not in ['Ports_Available']] + ['Ports_Available']]

pa_transformed.head()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

##### Boulder
# boulder = boulder.sort_values(by = ['Date Time', 'Station'])
# boulder.head()

boulder_transformed = boulder.copy()
## drop unneeded colums
boulder_transformed = boulder_transformed.drop(['Ports Occupied', 'Plugs', 'Fee', 'Notes', 'Date Hour'], axis=1)


## add datetime parts
boulder_transformed['Year-Month'] = boulder_transformed['Year'].astype(str) + '-' + boulder_transformed['Month'].astype(str)
boulder_transformed['DayofWeek'] = boulder_transformed['Date Time'].dt.weekday
boulder_transformed['IsWeekend'] = boulder_transformed['DayofWeek'] > 4

## holidays
boulder_transformed = boulder_transformed.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)


# Apply cosine and sine transformations to cyclical features
boulder_transformed['month_cosine'] = np.cos(2 * math.pi * boulder_transformed['Month'] / boulder_transformed['Month'].max())

boulder_transformed['month_sine'] = np.sin(2 * math.pi * boulder_transformed['Month'] / boulder_transformed['Month'].max())

boulder_transformed['hour_cosine'] = np.cos(2 * math.pi * boulder_transformed['Date Time'].dt.hour / 
                                         boulder_transformed['Date Time'].dt.hour.max())

boulder_transformed['hour_sine'] = np.sin(2 * math.pi * boulder_transformed['Date Time'].dt.hour / 
                                       boulder_transformed['Date Time'].dt.hour.max())

boulder_transformed['dayofweek_cosine'] = np.cos(2 * math.pi * boulder_transformed['DayofWeek'] / 
                                              boulder_transformed['DayofWeek'].max())

boulder_transformed['dayofweek_sine'] = np.sin(2 * math.pi * boulder_transformed['DayofWeek'] / 
                                            boulder_transformed['DayofWeek'].max())

boulder_transformed['IsWeekend'] = boulder_transformed['IsWeekend'].astype(int)


# Drop unnecessary columns
boulder_transformed = boulder_transformed.drop(columns = ['DayofWeek', 'Month', 'Year', 'Date', 'Year-Month'])

# Sort by DateTime
boulder_transformed = boulder_transformed.sort_values(by = 'Date Time').reset_index(drop=True)

# move output to end of col for LSTMs
boulder_transformed = boulder_transformed[[c for c in boulder_transformed if c not in ['Ports Available']] + ['Ports Available']]

boulder_transformed.head()


# COMMAND ----------

# MAGIC %md
# MAGIC #Data Filtering
# MAGIC 
# MAGIC Filtering data to 129 days. 3-months will be used for training. Will use the following time periods
# MAGIC 
# MAGIC Palo Alto starting on 2019-01-01 <br>
# MAGIC Berkeley, SlrpEV starts on 2021-11-01 <br>
# MAGIC Boulder Starts on 2021-11-24<br>

# COMMAND ----------

####### Filtering data to 129 days ###########
## 3 months will be used for training


########### Berkeley ###########
start = dt.datetime(2021, 11, 1)
end_date = start + dt.timedelta(days = int(129))
end  = end_date.replace(hour=0, minute=0, second = 0)
date_col = 'DateTime'

## filter data set
slrp_feat = arima_filter(slrp_transformed, start, end, date_col)



########## Palo Alto ###########
start = dt.datetime(2019, 1, 1)
end_date = start + dt.timedelta(days = int(129))
end  = end_date.replace(hour=0, minute=0, second = 0)
date_col = 'datetime_pac'

## filter dataset
paloalto_feat = arima_filter(pa_transformed, start, end, date_col)



########## Boulder #############
start = dt.datetime(2021, 11, 24)
end_date = start + dt.timedelta(days = int(129))
end  = end_date.replace(hour=0, minute=0, second = 0)
date_col = 'Date Time'

## filter dataset
boulder_feat = arima_filter(boulder_transformed, start, end, date_col).drop_duplicates()





########### check ############
### berkeley
print("SLRP: Min Date: ",slrp_feat['DateTime'].min() ," Max Date: ", slrp_feat['DateTime'].max())


#### boulder
date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder_feat
result = test.groupby('proper_name').agg({date_col: ['min', 'max']})
 
print()
print("Boulder Dates for each stations")
print(result)


### palo alto
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

test = paloalto_feat
result = test.groupby('Station_Location').agg({date_col: ['min', 'max']})

print()
print("Palo Alto Dates for each stations")
print(result)

# COMMAND ----------

print(slrp.shape)
print(paloalto_feat.shape)
print(boulder_feat.shape)


########### check ############

#### boulder
date_col = 'Date Time'
actualcol= 'Ports Available'

test = boulder_feat
result = test.groupby('proper_name').agg({date_col: ['count']})
print()
print("Boulder")
print(result)


### palo alto
date_col = 'datetime_pac'
actualcol= 'Ports_Available'

test = paloalto_feat
result = test.groupby('Station_Location').agg({date_col: ['count']})

print()
print("Palo Alto")
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

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Average

# COMMAND ----------

actual_col = 'Ports Available'

## split into train and test
traindf, testdf = split2_TrainTest(slrp_feat, 0.7)

avg_predictions, avg_roundpredictions = predict_average_overall(traindf, testdf.shape[0], actual_col)

testdf.loc[:,'Predictions'] = avg_predictions
testdf.loc[:,'Predictions_Rounded'] = avg_roundpredictions

testdf = testdf[['DateTime', actual_col, 'Predictions', 'Predictions_Rounded']]
testdf.loc[:,'Location'] = 'Berkeley'
testdf.loc[:,'SiteName'] = 'Slrp'
testdf.loc[:,'Features'] = 'No'
testdf.loc[:,'Model'] = 'Simple Average'

slrp_avg_model = testdf.rename(columns={actual_col: "Actuals"})

#### write dataframe to s3
#write_to_s3(slrp_avg_model, "BerkeleySlrp_BenchmarkSimpleAvg")




MSE_Raw, RMSE_Raw = calc_rmse(slrp_avg_model['Predictions'], slrp_avg_model['Actuals'])
MSE_round, RMSE_round = calc_rmse(slrp_avg_model['Predictions_Rounded'], slrp_avg_model['Actuals'])

Berkeley_simpleavg = dict({'Slrp': 
                 dict({'MSE_Raw': MSE_Raw,
                      'MSE_round': MSE_round,
                      'RMSE_Raw': RMSE_Raw,
                      'RMSE': RMSE_round}) 
                 }) 

print(Berkeley_simpleavg)

# COMMAND ----------

actual_col = 'Ports Available'

## split into train and test
traindf, testdf = split2_TrainTest(slrp_feat, 0.7)

avg_predictions, avg_roundpredictions = predict_average_overall(traindf, testdf.shape[0], actual_col)

testdf.loc[:,'Predictions'] = avg_predictions
testdf.loc[:,'Predictions_Rounded'] = avg_roundpredictions

testdf = testdf[['DateTime', actual_col, 'Predictions', 'Predictions_Rounded']]
testdf.loc[:,'Location'] = 'Berkeley'
testdf.loc[:,'SiteName'] = 'Slrp'
testdf.loc[:,'Features'] = 'No'
testdf.loc[:,'Model'] = 'Simple Average'

slrp_avg_model = testdf.rename(columns={actual_col: "Actuals"})

#### write dataframe to s3
write_to_s3(slrp_avg_model, "BerkeleySlrp_BenchmarkSimpleAvg")




MSE_Raw, RMSE_Raw = calc_rmse(slrp_avg_model['Predictions'], slrp_avg_model['Actuals'])

MSE_round, RMSE_round = calc_rmse(slrp_avg_model['Predictions_Rounded'], slrp_avg_model['Actuals'])

Berkeley_simpleavg = dict({'Slrp': 
                 dict({'MSE_Raw': MSE_Raw,
                      'MSE_round': MSE_round,
                      'RMSE_Raw': RMSE_Raw,
                      'RMSE': RMSE_round}) 
                 }) 

print(Berkeley_simpleavg)

# COMMAND ----------

### simple average
pa_results_df = pd.DataFrame()
pa_metrics = dict()

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actual_col= 'Ports_Available'
stationcol = 'Station_Location'

stations = paloalto_feat[stationcol].unique()


for station in stations: 
    # split into train and test
    stationdf = paloalto_feat[paloalto_feat[stationcol] == station]
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    # predict
    avg_predictions, avg_roundpredictions = predict_average_overall(traindf, testdf.shape[0], actual_col)
    testdf.loc[:,'Predictions'] = avg_predictions
    testdf.loc[:,'Predictions_Rounded'] = avg_roundpredictions

    testdf = testdf[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    testdf.loc[:,'Location'] = str(loc_)
    testdf.loc[:,'SiteName'] = str(station)
    testdf.loc[:,'Features'] = 'No'
    testdf.loc[:,'Model'] = 'Simple Average'

    testdf = testdf.rename(columns={actual_col: "Actuals"})
    
    # metrics
    MSE_Raw, RMSE_Raw = calc_rmse(testdf['Predictions'], testdf['Actuals'])
    MSE_round, RMSE_round = calc_rmse(testdf['Predictions_Rounded'], testdf['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 
    print(evals)
    
    pa_metrics.update(evals)
    print(pa_metrics)
    ##### append each station df to results df
    pa_results_df = pa_results_df.append(testdf)


#### write dataframe to s3
write_to_s3(pa_results_df, "PaloAlto_BenchmarkSimpleAvg")

# COMMAND ----------

### simple average
bo_results_df = pd.DataFrame()
bo_metrics = dict()


loc_ = 'Boulder'
date_col = 'Date Time'
actual_col= 'Ports Available'
stationcol = 'Station'


stations = boulder_feat[stationcol].unique()


for station in stations: 
    stationdf = boulder_feat[boulder_feat[stationcol] == station]
    # split into train and test
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    # predict
    avg_predictions, avg_roundpredictions = predict_average_overall(traindf, testdf.shape[0], actual_col)
    testdf.loc[:,'Predictions'] = avg_predictions
    testdf.loc[:,'Predictions_Rounded'] = avg_roundpredictions

    testdf = testdf[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    testdf.loc[:,'Location'] = str(loc_)
    testdf.loc[:,'SiteName'] = str(station)
    testdf.loc[:,'Features'] = 'No'
    testdf.loc[:,'Model'] = 'Simple Average'

    testdf = testdf.rename(columns={actual_col: "Actuals"})
    
    # metrics
    MSE_Raw, RMSE_Raw = calc_rmse(testdf['Predictions'], testdf['Actuals'])
    MSE_round, RMSE_round = calc_rmse(testdf['Predictions_Rounded'], testdf['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 
    print(evals)
    
    bo_metrics.update(evals)
    print(bo_metrics)
    ##### append each station df to results df
    bo_results_df = bo_results_df.append(testdf)


# #### write dataframe to s3
write_to_s3(bo_results_df, "Boulder_BenchmarkSimpleAvg")

# COMMAND ----------

berkeley_simp = {'Berkeley': {'Slrp': {'MSE_Raw': 1.617500235370586, 'MSE_round': 1.559842095819128, 'RMSE_Raw': 1.2718098267314126, 'RMSE': 1.2489363858175995}}}
paloalto_simp = {'Palo Alto': {'BRYANT': {'MSE_Raw': 6.073218129737852, 'MSE_round': 5.996411268616544, 'RMSE_Raw': 2.464390011694142, 'RMSE': 2.448757086486233}, 'HAMILTON': {'MSE_Raw': 1.379941540385773, 'MSE_round': 1.458819307374843, 'RMSE_Raw': 1.1747091301193555, 'RMSE': 1.207815924458211}, 'CAMBRIDGE': {'MSE_Raw': 4.9843585087397635, 'MSE_round': 4.89413242418805, 'RMSE_Raw': 2.232567694100173, 'RMSE': 2.212268614835922}, 'HIGH': {'MSE_Raw': 4.1348313109883845, 'MSE_round': 4.145881930737485, 'RMSE_Raw': 2.033428462225407, 'RMSE': 2.036143887532874}, 'MPL': {'MSE_Raw': 0.9773685162786255, 'MSE_round': 0.997308451462408, 'RMSE_Raw': 0.9886195002520562, 'RMSE': 0.9986533189562873}, 'TED_THOMPSON': {'MSE_Raw': 3.0784780918514723, 'MSE_round': 3.2971469585501527, 'RMSE_Raw': 1.754559230077877, 'RMSE': 1.8158047688422212}, 'WEBSTER': {'MSE_Raw': 3.820571119040337, 'MSE_round': 3.8273820204557687, 'RMSE_Raw': 1.954628128069464, 'RMSE': 1.9563696022111385}, 'RINCONADA_LIB': {'MSE_Raw': 0.7501454139488949, 'MSE_round': 0.8133859680602907, 'RMSE_Raw': 0.8661093544979727, 'RMSE': 0.901879131624793}}}

boulder_simp = {'Boulder' : {'1505 30th St': {'MSE_Raw': 0.9194707186270493, 'MSE_round': 0.9700340929481428, 'RMSE_Raw': 0.9588903579800192, 'RMSE': 0.9849030880996072}, '600 Baseline Rd': {'MSE_Raw': 0.1729638840956741, 'MSE_round': 0.1733357258209223, 'RMSE_Raw': 0.4158892690316427, 'RMSE': 0.41633607316796645}, '1400 Walnut St': {'MSE_Raw': 0.11128820622952669, 'MSE_round': 0.12076081105329266, 'RMSE_Raw': 0.33359887024617857, 'RMSE': 0.34750656260464013}, '1739 Broadway': {'MSE_Raw': 0.1345069095544081, 'MSE_round': 0.13978108738560918, 'RMSE_Raw': 0.36675183647039605, 'RMSE': 0.37387308994578516}, '3172 Broadway': {'MSE_Raw': 0.38147901168742665, 'MSE_round': 0.5058316884981159, 'RMSE_Raw': 0.6176398721645379, 'RMSE': 0.7112184534291246}, '900 Walnut St': {'MSE_Raw': 0.3327507475240159, 'MSE_round': 0.4017584783778934, 'RMSE_Raw': 0.5768455144352046, 'RMSE': 0.633844206708473}, '1770 13th St': {'MSE_Raw': 0.10640227101757901, 'MSE_round': 0.11232729230217119, 'RMSE_Raw': 0.32619360971297245, 'RMSE': 0.33515264030314784}, '3335 Airport Rd': {'MSE_Raw': 0.11758828628160437, 'MSE_round': 0.12757940068185897, 'RMSE_Raw': 0.34291148461608045, 'RMSE': 0.35718258731614977}, '1100 Walnut': {'MSE_Raw': 0.2762144421998442, 'MSE_round': 0.3324959626771936, 'RMSE_Raw': 0.5255610737106052, 'RMSE': 0.5766246289200572}, '1500 Pearl St': {'MSE_Raw': 0.4961217917069463, 'MSE_round': 0.7611699264310067, 'RMSE_Raw': 0.7043591354607012, 'RMSE': 0.8724505295035396}, '1745 14th street': {'MSE_Raw': 0.23469423954535246, 'MSE_round': 0.35618158980800285, 'RMSE_Raw': 0.48445251526372785, 'RMSE': 0.59680950881165}, '5660 Sioux Dr': {'MSE_Raw': 0.12977262666556802, 'MSE_round': 0.1449847478916203, 'RMSE_Raw': 0.3602396794712765, 'RMSE': 0.38076862776707365}, '2667 Broadway': {'MSE_Raw': 0.06380580105242177, 'MSE_round': 0.06818589628566302, 'RMSE_Raw': 0.25259810183851694, 'RMSE': 0.26112429279112087}, '1360 Gillaspie Dr': {'MSE_Raw': 0.2392659857827969, 'MSE_round': 0.2822537233088103, 'RMSE_Raw': 0.4891482247568695, 'RMSE': 0.5312755624991707}, '5565 51st St': {'MSE_Raw': 0.018889062821175306, 'MSE_round': 0.019020276332316528, 'RMSE_Raw': 0.13743748695743568, 'RMSE': 0.13791401789635646}, '5333 Valmont Rd': {'MSE_Raw': 0.34890315218262796, 'MSE_round': 0.39422214247263593, 'RMSE_Raw': 0.5906802452957336, 'RMSE': 0.6278711193172019}, '1100 Spruce St': {'MSE_Raw': 0.3796244133161643, 'MSE_round': 0.48968239727256413, 'RMSE_Raw': 0.6161366839558933, 'RMSE': 0.699773104136308}, '2052 Junction Pl': {'MSE_Raw': 0.15150822022974003, 'MSE_round': 0.17190023326753992, 'RMSE_Raw': 0.38924056858161643, 'RMSE': 0.4146085301432424}}}

simpleavg =  dictstodf(berkeley_simp, paloalto_simp, boulder_simp, 'Simple Average','No')
simpleavg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Average by Day/Hour

# COMMAND ----------

date_col = 'DateTime'
actual_col = 'Ports Available'


traindf, testdf = split2_TrainTest(slrp_feat, 0.7)
Berkely_avgDayHour =  predict_avg_by_day_hour(traindf, testdf, date_col, actual_col)


Berkely_avgDayHour = Berkely_avgDayHour[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
Berkely_avgDayHour.loc[:,'Location'] = 'Berkeley'
Berkely_avgDayHour.loc[:,'SiteName'] = 'Slrp'
Berkely_avgDayHour.loc[:,'Features'] = 'Yes'
Berkely_avgDayHour.loc[:,'Model'] = 'Average: Day of Week and Hour'

Berkely_avgDayHour = Berkely_avgDayHour.rename(columns={actual_col: "Actuals"})

# #### write dataframe to s3
write_to_s3(Berkely_avgDayHour, "BerkeleySlrp_BenchmarkDayHourAvg")




MSE_Raw, RMSE_Raw = calc_rmse(Berkely_avgDayHour['Predictions'], Berkely_avgDayHour['Actuals'])
MSE_round, RMSE_round = calc_rmse(Berkely_avgDayHour['Predictions_Rounded'], Berkely_avgDayHour['Actuals'])

Berkeley_avgDayWeekHour = dict({'Slrp': 
                 dict({'MSE_Raw': MSE_Raw,
                      'MSE_round': MSE_round,
                      'RMSE_Raw': RMSE_Raw,
                      'RMSE': RMSE_round}) 
                 }) 

print(Berkeley_avgDayWeekHour)

# COMMAND ----------

### average day of week and hour
pa_results_df = pd.DataFrame()
pa_metrics = dict()

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actual_col= 'Ports_Available'
stationcol = 'Station_Location'

stations = paloalto_feat[stationcol].unique()



for station in stations:
    stationdf = paloalto_feat[paloalto_feat[stationcol] == station]
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    station_avgDayHour =  predict_avg_by_day_hour(traindf, testdf, date_col, actual_col)
    station_avgDayHour = station_avgDayHour[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    station_avgDayHour.loc[:,'Location'] = str(loc_)
    station_avgDayHour.loc[:,'SiteName'] = str(station)
    station_avgDayHour.loc[:,'Features'] = 'Yes'
    station_avgDayHour.loc[:,'Model'] = 'Average: Day of Week and Hour'

    station_avgDayHour = station_avgDayHour.rename(columns={actual_col: "Actuals", date_col: 'DateTime'})


    MSE_Raw, RMSE_Raw = calc_rmse(station_avgDayHour['Predictions'], station_avgDayHour['Actuals'])
    MSE_round, RMSE_round = calc_rmse(station_avgDayHour['Predictions_Rounded'], station_avgDayHour['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 

    print(evals)
    
    pa_metrics.update(evals)
    ##### append each station df to results df
    pa_results_df = pa_results_df.append(station_avgDayHour)


print(pa_metrics)
pa_results_df     
    
# #### write dataframe to s3
write_to_s3(pa_results_df , "PaloAlto_BenchmarkDayHourAvg")


# COMMAND ----------

### average day of week and hour
####### No Features ######### 
bo_results_df = pd.DataFrame()
bo_metrics = dict()

loc_ = 'Boulder'
date_col = 'Date Time'
actual_col= 'Ports Available'
stationcol = 'Station'

stations = boulder_feat[stationcol].unique()


for station in stations:
    stationdf = boulder_feat[boulder_feat[stationcol] == station]
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    station_avgDayHour =  predict_avg_by_day_hour(traindf, testdf, date_col, actual_col)
    station_avgDayHour = station_avgDayHour[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    station_avgDayHour.loc[:,'Location'] = str(loc_)
    station_avgDayHour.loc[:,'SiteName'] = str(station)
    station_avgDayHour.loc[:,'Features'] = 'Yes'
    station_avgDayHour.loc[:,'Model'] = 'Average: Day of Week and Hour'

    station_avgDayHour = station_avgDayHour.rename(columns={actual_col: "Actuals", date_col: 'DateTime'})


    MSE_Raw, RMSE_Raw = calc_rmse(station_avgDayHour['Predictions'], station_avgDayHour['Actuals'])
    MSE_round, RMSE_round = calc_rmse(station_avgDayHour['Predictions_Rounded'], station_avgDayHour['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 

    print(evals)
    
    bo_metrics.update(evals)
    ##### append each station df to results df
    bo_results_df = bo_results_df.append(station_avgDayHour)


print(bo_metrics)
bo_results_df     
    
# #### write dataframe to s3
write_to_s3(bo_results_df , "Boulder_BenchmarkDayHourAvg")

# COMMAND ----------

Berkeley_avgDayWeekHour = {'Berkeley': {'Slrp': {'MSE_Raw': 0.8515220380201735, 'MSE_round': 0.9438363538489144, 'RMSE_Raw': 0.9227795175556149, 'RMSE': 0.9715124054014516}}}

PaloAlto_avgDayWeekHour = {'Palo Alto' : {'BRYANT': {'MSE_Raw': 2.845388021784769, 'MSE_round': 2.923739458101561, 'RMSE_Raw': 1.6868277984977509, 'RMSE': 1.7098945751424446}, 'HAMILTON': {'MSE_Raw': 0.6081415235769522, 'MSE_round': 0.6666068544769423, 'RMSE_Raw': 0.7798342923832936, 'RMSE': 0.816459952769848}, 'CAMBRIDGE': {'MSE_Raw': 2.3142438100288527, 'MSE_round': 2.388839045397452, 'RMSE_Raw': 1.5212638857308263, 'RMSE': 1.545586958212786}, 'HIGH': {'MSE_Raw': 1.8663400034481212, 'MSE_round': 1.9502960703391352, 'RMSE_Raw': 1.3661405504003317, 'RMSE': 1.3965300105401013}, 'MPL': {'MSE_Raw': 0.6200379789930767, 'MSE_round': 0.710389377355105, 'RMSE_Raw': 0.7874249037165872, 'RMSE': 0.8428459985994505}, 'TED_THOMPSON': {'MSE_Raw': 1.4617882862764269, 'MSE_round': 1.5309528081823076, 'RMSE_Raw': 1.2090443690272192, 'RMSE': 1.2373167776209566}, 'WEBSTER': {'MSE_Raw': 1.4339645865582613, 'MSE_round': 1.504755069083079, 'RMSE_Raw': 1.197482603864566, 'RMSE': 1.2266845841874263}, 'RINCONADA_LIB': {'MSE_Raw': 0.6699918853407785, 'MSE_round': 0.8462228602189126, 'RMSE_Raw': 0.8185303203551952, 'RMSE': 0.9199037233422379}}}

Boulder_avgDayWeekHour = {'Boulder' : {'1505 30th St': {'MSE_Raw': 0.7749564046694093, 'MSE_round': 0.8867755248519649, 'RMSE_Raw': 0.8803160822508068, 'RMSE': 0.941687594084134}, '600 Baseline Rd': {'MSE_Raw': 0.16506183792957613, 'MSE_round': 0.20958191279382737, 'RMSE_Raw': 0.40627803033092513, 'RMSE': 0.4578011716824536}, '1400 Walnut St': {'MSE_Raw': 0.10766495718237634, 'MSE_round': 0.12076081105329266, 'RMSE_Raw': 0.3281233871310857, 'RMSE': 0.34750656260464013}, '1739 Broadway': {'MSE_Raw': 0.13529551970127407, 'MSE_round': 0.13978108738560918, 'RMSE_Raw': 0.36782539295333333, 'RMSE': 0.37387308994578516}, '3172 Broadway': {'MSE_Raw': 0.2920019348272951, 'MSE_round': 0.3830970751839225, 'RMSE_Raw': 0.540372033720561, 'RMSE': 0.6189483622919787}, '900 Walnut St': {'MSE_Raw': 0.28208770614117823, 'MSE_round': 0.32424188049524494, 'RMSE_Raw': 0.5311192955835612, 'RMSE': 0.5694224095478198}, '1770 13th St': {'MSE_Raw': 0.10650559462223531, 'MSE_round': 0.11232729230217119, 'RMSE_Raw': 0.326351949009402, 'RMSE': 0.33515264030314784}, '3335 Airport Rd': {'MSE_Raw': 0.10615413164792727, 'MSE_round': 0.1394222142472636, 'RMSE_Raw': 0.325813031734348, 'RMSE': 0.3733928417193661}, '1100 Walnut': {'MSE_Raw': 0.23898444342946074, 'MSE_round': 0.31024582809976675, 'RMSE_Raw': 0.4888603516644204, 'RMSE': 0.5569971526855113}, '1500 Pearl St': {'MSE_Raw': 0.45666009305369665, 'MSE_round': 0.5684550511394222, 'RMSE_Raw': 0.6757663006200417, 'RMSE': 0.7539595819003975}, '1745 14th street': {'MSE_Raw': 0.25758945768183905, 'MSE_round': 0.49345056522519287, 'RMSE_Raw': 0.5075327158734095, 'RMSE': 0.7024603655902537}, '5660 Sioux Dr': {'MSE_Raw': 0.12442226365026655, 'MSE_round': 0.1530593935043962, 'RMSE_Raw': 0.35273540175359, 'RMSE': 0.39122805817629724}, '2667 Broadway': {'MSE_Raw': 0.07468010960147081, 'MSE_round': 0.06818589628566302, 'RMSE_Raw': 0.2732766173705149, 'RMSE': 0.26112429279112087}, '1360 Gillaspie Dr': {'MSE_Raw': 0.21791167515766227, 'MSE_round': 0.28010048447873676, 'RMSE_Raw': 0.46681010610060947, 'RMSE': 0.5292452026034216}, '5565 51st St': {'MSE_Raw': 0.01884190151799091, 'MSE_round': 0.019020276332316528, 'RMSE_Raw': 0.13726580607708136, 'RMSE': 0.13791401789635646}, '5333 Valmont Rd': {'MSE_Raw': 0.3237217497545316, 'MSE_round': 0.35725820922303964, 'RMSE_Raw': 0.568965508404975, 'RMSE': 0.5977108073500426}, '1100 Spruce St': {'MSE_Raw': 0.3012725423487835, 'MSE_round': 0.428673963753813, 'RMSE_Raw': 0.5488829951353781, 'RMSE': 0.6547319785636051}, '2052 Junction Pl': {'MSE_Raw': 0.15517270934023814, 'MSE_round': 0.17190023326753992, 'RMSE_Raw': 0.39391967371564235, 'RMSE': 0.4146085301432424}}}

avgDayHour =  dictstodf(Berkeley_avgDayWeekHour, PaloAlto_avgDayWeekHour, Boulder_avgDayWeekHour, 'Average DayofWeek Hour','Yes')
avgDayHour

# COMMAND ----------

# MAGIC %md
# MAGIC ## Average last 12 inputs and predict next

# COMMAND ----------

date_col = 'DateTime'
actual_col = 'Ports Available'


traindf, testdf = split2_TrainTest(slrp_feat, 0.7)
predictions, predictions_round =  predict_average_n_timestamps(traindf, 12, testdf.shape[0], actual_col)

testdf.loc[:, 'Predictions'] = predictions
testdf.loc[:, 'Predictions_Rounded'] = predictions_round

Berkeley_last12df = testdf[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
Berkeley_last12df.loc[:,'Location'] = 'Berkeley'
Berkeley_last12df.loc[:,'SiteName'] = 'Slrp'
Berkeley_last12df.loc[:,'Features'] = 'No'
Berkeley_last12df.loc[:,'Model'] = 'Average: Last 12'

Berkeley_last12df = Berkeley_last12df.rename(columns={actual_col: "Actuals", date_col: 'DateTime'})


# #### write dataframe to s3
write_to_s3(Berkeley_last12df, "BerkeleySlrp_BenchmarkAvgLast12")



MSE_Raw, RMSE_Raw = calc_rmse(Berkeley_last12df['Predictions'], Berkeley_last12df['Actuals'])
MSE_round, RMSE_round = calc_rmse(Berkeley_last12df['Predictions_Rounded'], Berkeley_last12df['Actuals'])

Berkeley_last12 = dict({'Slrp': 
                 dict({'MSE_Raw': MSE_Raw,
                      'MSE_round': MSE_round,
                      'RMSE_Raw': RMSE_Raw,
                      'RMSE': RMSE_round}) 
                 }) 

print(Berkeley_last12)
Berkeley_last12df

# COMMAND ----------

## no features last 12
pa_results_df = pd.DataFrame()
pa_metrics = dict()

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actual_col= 'Ports_Available'
stationcol = 'Station_Location'


stations = paloalto_feat[stationcol].unique()

for station in stations:
    stationdf = paloalto_feat[paloalto_feat[stationcol] == station]
    
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    predictions, predictions_round =  predict_average_n_timestamps(traindf, 12, testdf.shape[0], actual_col)

    testdf.loc[:, 'Predictions'] = predictions
    testdf.loc[:, 'Predictions_Rounded'] = predictions_round

    testdf = testdf[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    testdf.loc[:,'Location'] = str(loc_)
    testdf.loc[:,'SiteName'] = str(station)
    testdf.loc[:,'Features'] = 'No'
    testdf.loc[:,'Model'] = 'Average: Last 12'

    testdf = testdf.rename(columns={actual_col: "Actuals", date_col: 'DateTime'})


    ## Metrics
    MSE_Raw, RMSE_Raw = calc_rmse(testdf['Predictions'], testdf['Actuals'])
    MSE_round, RMSE_round = calc_rmse(testdf['Predictions_Rounded'], testdf['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 

    print(evals)
    
    pa_metrics.update(evals)
    ##### append each station df to results df
    pa_results_df = pa_results_df.append(testdf)


print(pa_metrics)
pa_results_df     
    
# #### write dataframe to s3
write_to_s3(pa_results_df ,  "PaloAlto_BenchmarkAvgLast12")

# COMMAND ----------

## no features last 12
bo_results_df = pd.DataFrame()
bo_metrics = dict()

loc_ = 'Boulder'
date_col = 'Date Time'
actual_col= 'Ports Available'
stationcol = 'Station'

stations = boulder_feat[stationcol].unique()


for station in stations:
    stationdf = boulder_feat[boulder_feat[stationcol] == station]
    
    traindf, testdf = split2_TrainTest(stationdf, 0.7)
    
    predictions, predictions_round =  predict_average_n_timestamps(traindf, 12, testdf.shape[0], actual_col)

    testdf.loc[:, 'Predictions'] = predictions
    testdf.loc[:, 'Predictions_Rounded'] = predictions_round

    testdf = testdf[[date_col, actual_col, 'Predictions', 'Predictions_Rounded']]
    testdf.loc[:,'Location'] = str(loc_)
    testdf.loc[:,'SiteName'] = str(station)
    testdf.loc[:,'Features'] = 'No'
    testdf.loc[:,'Model'] = 'Average: Last 12'

    testdf = testdf.rename(columns={actual_col: "Actuals", date_col: 'DateTime'})


    ## Metrics
    MSE_Raw, RMSE_Raw = calc_rmse(testdf['Predictions'], testdf['Actuals'])
    MSE_round, RMSE_round = calc_rmse(testdf['Predictions_Rounded'], testdf['Actuals'])

    evals = dict({str(station): 
                     dict({'MSE_Raw': MSE_Raw,
                          'MSE_round': MSE_round,
                          'RMSE_Raw': RMSE_Raw,
                          'RMSE': RMSE_round}) 
                     }) 

    print(evals)
    
    bo_metrics.update(evals)
    ##### append each station df to results df
    bo_results_df = bo_results_df.append(testdf)


print(bo_metrics)
bo_results_df     
    
# #### write dataframe to s3
write_to_s3(bo_results_df ,  "Boulder_BenchmarkAvgLast12")

# COMMAND ----------

Berkeley_Last12 = {'Berkeley': {'Slrp': {'MSE_Raw': 2.350619056163646, 'MSE_round': 2.350619056163646, 'RMSE_Raw': 1.5331728722370632, 'RMSE': 1.5331728722370632}}}

PaloAlto_Last12 = {'Palo Alto' : {'BRYANT': {'MSE_Raw': 9.131182168621645, 'MSE_round': 10.15718643459537, 'RMSE_Raw': 3.0217845999709585, 'RMSE': 3.1870341125559625}, 'HAMILTON': {'MSE_Raw': 1.458819307374843, 'MSE_round': 1.458819307374843, 'RMSE_Raw': 1.207815924458211, 'RMSE': 1.207815924458211}, 'CAMBRIDGE': {'MSE_Raw': 9.11735151623901, 'MSE_round': 9.11735151623901, 'RMSE_Raw': 3.0194952419633005, 'RMSE': 3.0194952419633005}, 'HIGH': {'MSE_Raw': 7.236497398169747, 'MSE_round': 7.236497398169747, 'RMSE_Raw': 2.6900738648166795, 'RMSE': 2.6900738648166795}, 'MPL': {'MSE_Raw': 0.997308451462408, 'MSE_round': 0.997308451462408, 'RMSE_Raw': 0.9986533189562873, 'RMSE': 0.9986533189562873}, 'TED_THOMPSON': {'MSE_Raw': 3.4648407753905452, 'MSE_round': 3.2971469585501527, 'RMSE_Raw': 1.8614082774583725, 'RMSE': 1.8158047688422212}, 'WEBSTER': {'MSE_Raw': 6.598725923509916, 'MSE_round': 7.090974340570608, 'RMSE_Raw': 2.5687985369642976, 'RMSE': 2.6628883454945322}, 'RINCONADA_LIB': {'MSE_Raw': 2.3671272205275433, 'MSE_round': 2.3671272205275433, 'RMSE_Raw': 1.5385471135222162, 'RMSE': 1.5385471135222162}}}

Boulder_Last12 = {'Boulder' : {'1505 30th St': {'MSE_Raw': 1.5070877444823254, 'MSE_round': 1.5070877444823254, 'RMSE_Raw': 1.2276350208764515, 'RMSE': 1.2276350208764515}, '600 Baseline Rd': {'MSE_Raw': 0.1733357258209223, 'MSE_round': 0.1733357258209223, 'RMSE_Raw': 0.41633607316796645, 'RMSE': 0.41633607316796645}, '1400 Walnut St': {'MSE_Raw': 0.12076081105329266, 'MSE_round': 0.12076081105329266, 'RMSE_Raw': 0.34750656260464013, 'RMSE': 0.34750656260464013}, '1739 Broadway': {'MSE_Raw': 0.13978108738560918, 'MSE_round': 0.13978108738560918, 'RMSE_Raw': 0.37387308994578516, 'RMSE': 0.37387308994578516}, '3172 Broadway': {'MSE_Raw': 0.383870308530452, 'MSE_round': 0.5060111250672887, 'RMSE_Raw': 0.6195726822015735, 'RMSE': 0.7113445895396188}, '900 Walnut St': {'MSE_Raw': 0.4017584783778934, 'MSE_round': 0.4017584783778934, 'RMSE_Raw': 0.633844206708473, 'RMSE': 0.633844206708473}, '1770 13th St': {'MSE_Raw': 0.11232729230217119, 'MSE_round': 0.11232729230217119, 'RMSE_Raw': 0.33515264030314784, 'RMSE': 0.33515264030314784}, '3335 Airport Rd': {'MSE_Raw': 0.12757940068185897, 'MSE_round': 0.12757940068185897, 'RMSE_Raw': 0.35718258731614977, 'RMSE': 0.35718258731614977}, '1100 Walnut': {'MSE_Raw': 0.3324959626771936, 'MSE_round': 0.3324959626771936, 'RMSE_Raw': 0.5766246289200572, 'RMSE': 0.5766246289200572}, '1500 Pearl St': {'MSE_Raw': 0.7611699264310067, 'MSE_round': 0.7611699264310067, 'RMSE_Raw': 0.8724505295035396, 'RMSE': 0.8724505295035396}, '1745 14th street': {'MSE_Raw': 0.35618158980800285, 'MSE_round': 0.35618158980800285, 'RMSE_Raw': 0.59680950881165, 'RMSE': 0.59680950881165}, '5660 Sioux Dr': {'MSE_Raw': 0.1449847478916203, 'MSE_round': 0.1449847478916203, 'RMSE_Raw': 0.38076862776707365, 'RMSE': 0.38076862776707365}, '2667 Broadway': {'MSE_Raw': 0.9325318499910281, 'MSE_round': 0.9325318499910281, 'RMSE_Raw': 0.9656768869508207, 'RMSE': 0.9656768869508207}, '1360 Gillaspie Dr': {'MSE_Raw': 0.8512470841557509, 'MSE_round': 0.8512470841557509, 'RMSE_Raw': 0.922630524183842, 'RMSE': 0.922630524183842}, '5565 51st St': {'MSE_Raw': 0.019020276332316528, 'MSE_round': 0.019020276332316528, 'RMSE_Raw': 0.13791401789635646, 'RMSE': 0.13791401789635646}, '5333 Valmont Rd': {'MSE_Raw': 0.39422214247263593, 'MSE_round': 0.39422214247263593, 'RMSE_Raw': 0.6278711193172019, 'RMSE': 0.6278711193172019}, '1100 Spruce St': {'MSE_Raw': 0.48968239727256413, 'MSE_round': 0.48968239727256413, 'RMSE_Raw': 0.699773104136308, 'RMSE': 0.699773104136308}, '2052 Junction Pl': {'MSE_Raw': 0.17190023326753992, 'MSE_round': 0.17190023326753992, 'RMSE_Raw': 0.4146085301432424, 'RMSE': 0.4146085301432424}}}

Last12Avg =  dictstodf(Berkeley_Last12, PaloAlto_Last12, Boulder_Last12, 'Average Last 12','No')
Last12Avg

# COMMAND ----------

# MAGIC %md
# MAGIC # ARIMA Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley

# COMMAND ----------

date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
df = slrp_feat

######### No Features #########
## run EDA
arima_eda(df, 'Ports Available', 25, df.shape[0]-1)

## split into train and test
traindf, testdf = split2_TrainTest(df, 0.7)

## run model
berk_model, b_testdf, bevals, berk_pqd = run_arima(traindf, testdf, actualcol, date_col, station)

## plot
info = 'MSE Predictions: ' + str(bevals['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(bevals['Slrp']['MSE_round']) + '\n arima order = ' + str(berk_pqd)
size_ = (15, 7)
plot_predsActuals(b_testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)


#### write dataframe to s3
write_to_s3(b_testdf, "BerkeleySlrp_ARIMA")
## write model to s3
write_model_to_s3(berk_model, 'BatchModel_BerkeleySlrp_ARIMA')

# COMMAND ----------

# DBTITLE 1,Features
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
df = slrp_feat
 
######### No Features #########
## run EDA
arima_eda(df, 'Ports Available', 25, df.shape[0]-1)
 
## split into train and test
traindf, testdf = split2_TrainTest(df, 0.7)
 
## run model
berk_model2, b_testdf2, bevals2, berk_pqd2 = run_arima(traindf, testdf, actualcol, date_col, station, False)
 
## plot
info = 'MSE Predictions: ' + str(bevals2['Slrp']['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(bevals2['Slrp']['MSE_round']) + '\n arima order = ' + str(berk_pqd2)
size_ = (15, 7)
plot_predsActuals(b_testdf2, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)


#### write dataframe to s3
write_to_s3(b_testdf2, "BerkeleySlrp_ARIMA_Features")

## write model to s3
write_model_to_s3(berk_model2, 'BatchModel_BerkeleySlrp_ARIMA_Features')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

### no features
pa_results_df = pd.DataFrame()
pa_metrics = dict()
pa_station_error = dict()

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actualcol= 'Ports_Available'
stationcol = 'Station_Location'

df = paloalto_feat[[date_col, stationcol, actualcol]]


stations = df[stationcol].unique()

for station in stations:
    try:
        stationdf = df[df[stationcol] == station]
        stationdf = stationdf.drop(columns = [stationcol], axis = 1)
        
        ## run EDA
        try: 
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1)
        except:
            print('OLS used in EDA')
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1, 'ols') 
        
        ## split into train and test
        traindf, testdf = split2_TrainTest(stationdf, 0.7)
        #run model
        model, testdf, evals, pdq = run_arima(traindf, testdf, actualcol, date_col, station)
        
        ### write model to s3
        modelname = 'BatchModel_PaloAlto'+ str(station)+"_ARIMA"
        write_model_to_s3(model, modelname)

        ### plot
        #subtitle
        info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])+ '\n arima order = ' + str(pdq)
        size_ = (15, 7)
        plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

        ###### capture metrics
        pa_metrics.update(evals)
        print(pa_metrics)

        # add additional dataframe columns for visualizations
        testdf['Location'] = loc_
        testdf['SiteName'] = station

        ##### append each station df to results df
        pa_results_df = pa_results_df.append(testdf)
        
    except Exception as e:         
        print(station, " errored:")
        print(e)
        station_error = dict({station: e})
        pa_station_error.update(station_error)

        
print(pa_metrics)
pa_results_df.head()
print(pa_station_error)

#### write dataframe to s3
write_to_s3(pa_results_df, "PaloAlto_ARIMA")

# COMMAND ----------

### features
pa_results_df2 = pd.DataFrame()
pa_metrics2 = dict()
pa_station_error2 = dict()

loc_ = 'PaloAlto'
date_col = 'datetime_pac'
actualcol= 'Ports_Available'
stationcol = 'Station_Location'

df = paloalto_feat


stations = df[stationcol].unique()

for station in stations:
    try:
        stationdf = df[df[stationcol] == station]
        stationdf = stationdf.drop(columns = [stationcol], axis = 1)
        
        ## run EDA
        try: 
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1)
        except:
            print('OLS used in EDA')
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1, 'ols') 
        
        ## split into train and test
        traindf, testdf = split2_TrainTest(stationdf, 0.7)
        #run model
        model, testdf, evals, pdq = run_arima(traindf, testdf, actualcol, date_col, station, False)
        
        ### write model to s3
        modelname = 'BatchModel_PaloAlto'+ str(station)+"_ARIMAFeatures"
        write_model_to_s3(model, modelname)

        ### plot
        #subtitle
        info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])+ '\n arima order = ' + str(pdq)
        size_ = (15, 7)
        plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

        ###### capture metrics
        pa_metrics2.update(evals)
        print(pa_metrics2)

        # add additional dataframe columns for visualizations
        testdf['Location'] = loc_
        testdf['SiteName'] = station

        ##### append each station df to results df
        pa_results_df2 = pa_results_df2.append(testdf)
        
    except Exception as e:         
        print(station, " errored:")
        print(e)
        station_error = dict({station: e})
        pa_station_error2.update(station_error)

        
print(pa_metrics2)
pa_results_df2.head()
print(pa_station_error2)

#### write dataframe to s3
write_to_s3(pa_results_df2, "PaloAlto_ARIMA_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

####### No Features ######### 
bo_results_df = pd.DataFrame()
bo_metrics = dict()
bo_station_error = dict()

loc_ = 'Boulder'
date_col = 'Date Time'
actualcol= 'Ports Available'
stationcol = 'Station'

df = boulder_feat[[date_col, stationcol, actualcol]]


stations = df[stationcol].unique()

for station in stations:
    try:
        stationdf = df[df[stationcol] == station]
        stationdf = stationdf.drop(columns = [stationcol], axis = 1)
        
        ## run EDA
        try: 
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1)
        except:
            print('OLS used in EDA')
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1, 'ols') 
        
        ## split into train and test
        traindf, testdf = split2_TrainTest(stationdf, 0.7)
        #run model
        model, testdf, evals, pdq = run_arima(traindf, testdf, actualcol, date_col, station)
        
        ### write model to s3
        modelname = 'BatchModel_Boulder'+ str(station)+"_ARIMA"
        write_model_to_s3(model, modelname)

        ### plot
        #subtitle
        info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])+ '\n arima order = ' + str(pdq)
        size_ = (15, 7)
        plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

        ###### capture metrics
        bo_metrics.update(evals)
        print(bo_metrics)

        # add additional dataframe columns for visualizations
        testdf['Location'] = loc_
        testdf['SiteName'] = station

        ##### append each station df to results df
        bo_results_df = bo_results_df.append(testdf)
        
    except Exception as e:         
        print(station, " errored:")
        print(e)
        station_error = dict({station: e})
        bo_station_error.update(station_error)

        
print(bo_metrics)
bo_results_df.head()
print(bo_station_error)


#### write dataframe to s3
write_to_s3(bo_results_df, "Boulder_ARIMA")

# COMMAND ----------

####### Features ######### 
bo_results_df2 = pd.DataFrame()
bo_metrics2 = dict()
bo_station_error2 = dict()

loc_ = 'Boulder'
date_col = 'Date Time'
actualcol= 'Ports Available'
stationcol = 'Station'

df = boulder_feat.loc[:, (boulder_feat.columns != 'proper_name')]


stations = df[stationcol].unique()

for station in stations:
    try:
        stationdf = df[df[stationcol] == station]
        stationdf = stationdf.drop(columns = [stationcol], axis = 1)
        
        ## run EDA
        try: 
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1)
        except:
            print('OLS used in EDA')
            arima_eda(stationdf, actualcol, 25, stationdf.shape[0]-1, 'ols') 
        
        ## split into train and test
        traindf, testdf = split2_TrainTest(stationdf, 0.7)
        #run model
        model, testdf, evals, pdq = run_arima(traindf, testdf, actualcol, date_col, station, False)
        
        ### write model to s3
        modelname = 'BatchModel_Boulder'+ str(station)+"_ARIMA_Features"
        write_model_to_s3(model, modelname)

        ### plot
        #subtitle
        info = 'MSE Predictions: ' + str(evals[station]['MSE_Raw']) + '\n MSE Predictions Rounded: ' + str(evals[station]['MSE_round'])+ '\n arima order = ' + str(pdq)
        size_ = (15, 7)
        plot_predsActuals(testdf, 'predictions', 'predictions (rounded)', 'Actuals', 'DateTime', station, info, size_)

        ###### capture metrics
        bo_metrics2.update(evals)
        print(bo_metrics2)

        # add additional dataframe columns for visualizations
        testdf['Location'] = loc_
        testdf['SiteName'] = station

        ##### append each station df to results df
        bo_results_df2 = bo_results_df2.append(testdf)
        
    except Exception as e:         
        print(station, " errored:")
        print(e)
        station_error = dict({station: e})
        bo_station_error2.update(station_error)

        
print(bo_metrics2)
bo_results_df2.head()
print(bo_station_error2)


#### write dataframe to s3
write_to_s3(bo_results_df2, "Boulder_ARIMA_Features")

# COMMAND ----------

##### add traffic

# COMMAND ----------

## Metrics
berkeley_arima_nf = {'Berkeley': {'Slrp': {'MSE_Raw': 1.618060564722533, 'MSE_round': 1.5557150547281535, 'RMSE_Raw': 1.2720300958399267, 'RMSE': 1.247283069206086, 'PDQ_ARIMAOrder': (2, 0, 1)}}}

berkeley_arima_f = {'Berkeley':{'Slrp': {'MSE_Raw': 1.531123605437244, 'MSE_round': 1.4950654943477482, 'RMSE_Raw': 1.2373857949068447, 'RMSE': 1.2227287084009062, 'PDQ_ARIMAOrder': (4, 0, 2)}}}



## Palo ALto
## no features
paloalto_arima_nf = {'Palo Alto': {
    'BRYANT': {'MSE_Raw': 6.01709609601273, 'MSE_round': 5.954781984568455, 'RMSE_Raw': 2.4529769864417257, 'RMSE': 2.4402421979320934, 'PDQ_ARIMAOrder': (2, 0, 3)}, 
    'HAMILTON': {'MSE_Raw': 1.3779245992121416, 'MSE_round': 1.458819307374843, 'RMSE_Raw': 1.1738503308395587, 'RMSE': 1.207815924458211, 'PDQ_ARIMAOrder': (2, 0, 1)}, 
    'CAMBRIDGE': {'MSE_Raw': 4.975115117055593, 'MSE_round': 4.874753274717388, 'RMSE_Raw': 2.230496607721158, 'RMSE': 2.2078843436007665, 'PDQ_ARIMAOrder': (3, 0, 2)}, 
    'HIGH': {'MSE_Raw': 4.11212334968201, 'MSE_round': 4.12829714695855, 'RMSE_Raw': 2.0278371112300935, 'RMSE': 2.031821140494052, 'PDQ_ARIMAOrder': (2, 0, 3)}, 
    'MPL': {'MSE_Raw': 0.977897243198588, 'MSE_round': 0.997308451462408, 'RMSE_Raw': 0.9888868707787499, 'RMSE': 0.9986533189562873, 'PDQ_ARIMAOrder': (2, 0, 1)}, 
    'TED_THOMPSON': {'MSE_Raw': 3.0508481152390368, 'MSE_round': 3.2413421855374125, 'RMSE_Raw': 1.7466677174663292, 'RMSE': 1.8003727907123603, 'PDQ_ARIMAOrder': (2, 0, 2)}, 
    'WEBSTER': {'MSE_Raw': 3.8043438280449258, 'MSE_round': 3.804055266463305, 'RMSE_Raw': 1.9504727191234759, 'RMSE': 1.9503987455039302, 'PDQ_ARIMAOrder': (1, 0, 2)}, 
    'RINCONADA_LIB': {'MSE_Raw': 0.744484850134153, 'MSE_round': 0.8117710389377355, 'RMSE_Raw': 0.86283535517163, 'RMSE': 0.9009833732859533, 'PDQ_ARIMAOrder': (1, 0, 0)}}}


## with features
paloalto_arima_f = {'Palo Alto': {'BRYANT': {'MSE_Raw': 4.782053260959337, 'MSE_round': 4.892517495065494, 'RMSE_Raw': 2.186790630343778, 'RMSE': 2.2119035908161764, 'PDQ_ARIMAOrder': (1, 0, 3)}, 'HAMILTON': {'MSE_Raw': 0.9114100216934025, 'MSE_round': 0.9416831150188408, 'RMSE_Raw': 0.9546779675332423, 'RMSE': 0.970403583576875, 'PDQ_ARIMAOrder': (1, 0, 3)}, 'CAMBRIDGE': {'MSE_Raw': 3.9319077706645333, 'MSE_round': 4.147855732998385, 'RMSE_Raw': 1.9829038732789175, 'RMSE': 2.0366285211099213, 'PDQ_ARIMAOrder': (1, 0, 1)}, 'HIGH': {'MSE_Raw': 3.478576323958279, 'MSE_round': 3.537771397810874, 'RMSE_Raw': 1.8650941863504586, 'RMSE': 1.8808964346318682, 'PDQ_ARIMAOrder': (4, 0, 0)}, 'MPL': {'MSE_Raw': 1.2829604563843497, 'MSE_round': 1.3554638435313118, 'RMSE_Raw': 1.1326784435065187, 'RMSE': 1.1642438934910984, 'PDQ_ARIMAOrder': (2, 0, 0)}, 'TED_THOMPSON': {'MSE_Raw': 2.738235940582761, 'MSE_round': 2.6675040373228063, 'RMSE_Raw': 1.654761596298017, 'RMSE': 1.6332495330851333, 'PDQ_ARIMAOrder': (3, 0, 2)}, 'WEBSTER': {'MSE_Raw': 2.8961434905201675, 'MSE_round': 2.9163825587654766, 'RMSE_Raw': 1.7018059497252227, 'RMSE': 1.7077419473578193, 'PDQ_ARIMAOrder': (1, 0, 0)}, 'RINCONADA_LIB': {'MSE_Raw': 1.2588837396152175, 'MSE_round': 1.2768706262336265, 'RMSE_Raw': 1.1219998839639946, 'RMSE': 1.129987002683494, 'PDQ_ARIMAOrder': (1, 0, 0)}}}



### Boulder
## no features
boulder_arima_nf = {'Boulder': {
    '1505 30th St': {'MSE_Raw': 0.9220103410455158, 'MSE_round': 0.9741611340391172, 'RMSE_Raw': 0.9602136955102837, 'RMSE': 0.9869960152093408, 'PDQ_ARIMAOrder': (1, 0, 0)}, 
    '600 Baseline Rd': {'MSE_Raw': 0.172582856896137, 'MSE_round': 0.1733357258209223, 'RMSE_Raw': 0.4154309291520517, 'RMSE': 0.41633607316796645, 'PDQ_ARIMAOrder': (2, 0, 0)}, 
    '1739 Broadway': {'MSE_Raw': 0.13451025609779138, 'MSE_round': 0.13978108738560918, 'RMSE_Raw': 0.3667563988505059, 'RMSE': 0.37387308994578516, 'PDQ_ARIMAOrder': (1, 0, 2)}, 
    '3172 Broadway': {'MSE_Raw': 0.3810888973748628, 'MSE_round': 0.5074466176206711, 'RMSE_Raw': 0.6173239808843188, 'RMSE': 0.7123528743682243, 'PDQ_ARIMAOrder': (1, 0, 0)}, 
    '1770 13th St': {'MSE_Raw': 0.10640634357378084, 'MSE_round': 0.11232729230217119, 'RMSE_Raw': 0.3261998521976686, 'RMSE': 0.33515264030314784, 'PDQ_ARIMAOrder': (1, 0, 5)}, 
    '3335 Airport Rd': {'MSE_Raw': 0.11770560984886409, 'MSE_round': 0.12757940068185897, 'RMSE_Raw': 0.3430825117211078, 'RMSE': 0.35718258731614977, 'PDQ_ARIMAOrder': (1, 0, 2)}, 
    '1100 Walnut': {'MSE_Raw': 0.2762145426689134, 'MSE_round': 0.3324959626771936, 'RMSE_Raw': 0.525561169293274, 'RMSE': 0.5766246289200572, 'PDQ_ARIMAOrder': (4, 0, 3)}, 
    '1500 Pearl St': {'MSE_Raw': 0.4956849691640468, 'MSE_round': 0.7611699264310067, 'RMSE_Raw': 0.7040489820772748, 'RMSE': 0.8724505295035396, 'PDQ_ARIMAOrder': (2, 0, 0)}, 
    '1745 14th street': {'MSE_Raw': 0.24719715639480447, 'MSE_round': 0.35618158980800285, 'RMSE_Raw': 0.4971892561136096, 'RMSE': 0.59680950881165, 'PDQ_ARIMAOrder': (1, 0, 2)}, 
    '5660 Sioux Dr': {'MSE_Raw': 0.1298131270314375, 'MSE_round': 0.1449847478916203, 'RMSE_Raw': 0.36029588816892916, 'RMSE': 0.38076862776707365, 'PDQ_ARIMAOrder': (3, 0, 3)}, 
    '2667 Broadway': {'MSE_Raw': 0.0645983257475192, 'MSE_round': 0.07428673963753812, 'RMSE_Raw': 0.25416200689229534, 'RMSE': 0.27255593854755417, 'PDQ_ARIMAOrder': (1, 0, 3)}, 
    '1360 Gillaspie Dr': {'MSE_Raw': 0.23834588940746054, 'MSE_round': 0.28153597703211913, 'RMSE_Raw': 0.4882068100789465, 'RMSE': 0.5305996391179691, 'PDQ_ARIMAOrder': (1, 0, 3)}, 
    '5565 51st St': {'MSE_Raw': 0.01888911152847075, 'MSE_round': 0.019020276332316528, 'RMSE_Raw': 0.1374376641553208, 'RMSE': 0.13791401789635646, 'PDQ_ARIMAOrder': (3, 0, 0)}, 
    '5333 Valmont Rd': {'MSE_Raw': 0.3489601592151525, 'MSE_round': 0.39422214247263593, 'RMSE_Raw': 0.590728498732838, 'RMSE': 0.6278711193172019, 'PDQ_ARIMAOrder': (5, 0, 3)}, 
    '1100 Spruce St': {'MSE_Raw': 0.37945905946146513, 'MSE_round': 0.48968239727256413, 'RMSE_Raw': 0.6160024833241057, 'RMSE': 0.699773104136308, 'PDQ_ARIMAOrder': (1, 0, 0)}, 
    '2052 Junction Pl': {'MSE_Raw': 0.15149686135066795, 'MSE_round': 0.17190023326753992, 'RMSE_Raw': 0.389225977230025, 'RMSE': 0.4146085301432424, 'PDQ_ARIMAOrder': (1, 0, 0)}}}

# # {'1400 Walnut St': ValueError('The computed initial AR coefficients are not stationary\nYou should induce stationarity, choose a different model order, or you can\npass your own start_params.'), '900 Walnut St': ValueError('The computed initial AR coefficients are not stationary\nYou should induce stationarity, choose a different model order, or you can\npass your own start_params.')}


# ## with features
boulder_arima_f = {'Boulder': {'1505 30th St': {'MSE_Raw': 0.788780971941492, 'MSE_round': 0.8234344159339674, 'RMSE_Raw': 0.8881334201241906, 'RMSE': 0.907432871310031, 'PDQ_ARIMAOrder': (1, 0, 0)}, '600 Baseline Rd': {'MSE_Raw': 0.19194405678326099, 'MSE_round': 0.1733357258209223, 'RMSE_Raw': 0.4381142051831474, 'RMSE': 0.41633607316796645, 'PDQ_ARIMAOrder': (2, 0, 1)}, '1400 Walnut St': {'MSE_Raw': 0.10790481701781555, 'MSE_round': 0.12076081105329266, 'RMSE_Raw': 0.32848868628586825, 'RMSE': 0.34750656260464013, 'PDQ_ARIMAOrder': (1, 0, 0)}, '1739 Broadway': {'MSE_Raw': 0.1330747839195548, 'MSE_round': 0.13978108738560918, 'RMSE_Raw': 0.36479416650976587, 'RMSE': 0.37387308994578516, 'PDQ_ARIMAOrder': (2, 0, 0)}, '3172 Broadway': {'MSE_Raw': 0.33730531203425096, 'MSE_round': 0.5017046474071416, 'RMSE_Raw': 0.5807799170376425, 'RMSE': 0.7083111233117418, 'PDQ_ARIMAOrder': (2, 0, 1)}, '900 Walnut St': {'MSE_Raw': 0.3162120803970932, 'MSE_round': 0.34290328368921585, 'RMSE_Raw': 0.5623273783100847, 'RMSE': 0.5855794426798262, 'PDQ_ARIMAOrder': (1, 0, 0)}, '1770 13th St': {'MSE_Raw': 0.10662107613901174, 'MSE_round': 0.11232729230217119, 'RMSE_Raw': 0.32652882895544116, 'RMSE': 0.33515264030314784, 'PDQ_ARIMAOrder': (1, 0, 1)}, '3335 Airport Rd': {'MSE_Raw': 0.10972860590322053, 'MSE_round': 0.12757940068185897, 'RMSE_Raw': 0.33125308436785994, 'RMSE': 0.35718258731614977, 'PDQ_ARIMAOrder': (2, 0, 2)}, '1100 Walnut': {'MSE_Raw': 0.24460620526186688, 'MSE_round': 0.3324959626771936, 'RMSE_Raw': 0.49457679409962907, 'RMSE': 0.5766246289200572, 'PDQ_ARIMAOrder': (1, 0, 3)}, '1500 Pearl St': {'MSE_Raw': 0.5460925774017968, 'MSE_round': 0.7611699264310067, 'RMSE_Raw': 0.7389807693044501, 'RMSE': 0.8724505295035396, 'PDQ_ARIMAOrder': (2, 0, 0)}, '1745 14th street': {'MSE_Raw': 0.23485652122363057, 'MSE_round': 0.35618158980800285, 'RMSE_Raw': 0.4846199760881, 'RMSE': 0.59680950881165, 'PDQ_ARIMAOrder': (1, 0, 4)}, '5660 Sioux Dr': {'MSE_Raw': 0.12056770644403453, 'MSE_round': 0.1449847478916203, 'RMSE_Raw': 0.3472286083317942, 'RMSE': 0.38076862776707365, 'PDQ_ARIMAOrder': (1, 0, 3)}, '2667 Broadway': {'MSE_Raw': 0.06515210690619425, 'MSE_round': 0.07410730306836533, 'RMSE_Raw': 0.2552491075521994, 'RMSE': 0.2722265656918247, 'PDQ_ARIMAOrder': (5, 0, 2)}, '1360 Gillaspie Dr': {'MSE_Raw': 0.22571495431347674, 'MSE_round': 0.2820742867396375, 'RMSE_Raw': 0.4750946793150569, 'RMSE': 0.5311066623001801, 'PDQ_ARIMAOrder': (1, 0, 0)}, '5565 51st St': {'MSE_Raw': 0.018705631330699843, 'MSE_round': 0.019020276332316528, 'RMSE_Raw': 0.13676853194613095, 'RMSE': 0.13791401789635646, 'PDQ_ARIMAOrder': (2, 0, 1)}, '5333 Valmont Rd': {'MSE_Raw': 0.30895969761326675, 'MSE_round': 0.3949398887493271, 'RMSE_Raw': 0.555841432076871, 'RMSE': 0.6284424307359642, 'PDQ_ARIMAOrder': (1, 0, 5)}, '1100 Spruce St': {'MSE_Raw': 0.3271457604535144, 'MSE_round': 0.48968239727256413, 'RMSE_Raw': 0.5719665728462761, 'RMSE': 0.699773104136308, 'PDQ_ARIMAOrder': (1, 0, 0)}, '2052 Junction Pl': {'MSE_Raw': 0.15510961931283443, 'MSE_round': 0.17190023326753992, 'RMSE_Raw': 0.39383958576155653, 'RMSE': 0.4146085301432424, 'PDQ_ARIMAOrder': (1, 0, 0)}}}


# {'1400 Walnut St': ValueError('The computed initial AR coefficients are not stationary\nYou should induce stationarity, choose a different model order, or you can\npass your own start_params.'), '900 Walnut St': ValueError('The computed initial AR coefficients are not stationary\nYou should induce stationarity, choose a different model order, or you can\npass your own start_params.')}



arima_nf = dict()
arima_nf.update(berkeley_arima_nf)
arima_nf.update(paloalto_arima_nf)
arima_nf.update(boulder_arima_nf)

arima_nf = pd.DataFrame.from_dict({(i,j): arima_nf[i][j] 
                           for i in arima_nf.keys() 
                           for j in arima_nf[i].keys()},
                       orient='index')

arima_nf.reset_index(inplace=True)

arima_nf = arima_nf.rename(columns={"level_0":"city", "level_1":"station", "RMSE":"RMSE_round"})
arima_nf['Model'] ='ARIMA' 
arima_nf['Features'] = 'No'
# arima_nf



arima_f = dict()
arima_f.update(berkeley_arima_f)
arima_f.update(paloalto_arima_f)
arima_f.update(boulder_arima_f)

arima_f = pd.DataFrame.from_dict({(i,j): arima_f[i][j] 
                           for i in arima_f.keys() 
                           for j in arima_f[i].keys()},
                       orient='index')

arima_f.reset_index(inplace=True)

arima_f = arima_f.rename(columns={"level_0":"city", "level_1":"station", "RMSE":"RMSE_round"})
arima_f['Model'] ='ARIMA' 
arima_f['Features'] = 'Yes'

print(arima_nf.shape)
print(arima_f.shape)


arima = pd.concat([arima_nf, arima_f], ignore_index=True, sort=False)
print(arima.shape)
arima = arima.sort_values(by=['city', 'station'], ignore_index=True)
arima

# COMMAND ----------

# MAGIC %md 
# MAGIC # Prophet
# MAGIC 
# MAGIC 
# MAGIC https://colab.research.google.com/github/Apress/hands-on-time-series-analylsis-python/blob/master/Chapter%208/4.%20fbprophet_with_exogenous_or_add_regressors.ipynb#scrollTo=4uSAA5VLfHm2 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors
# MAGIC 
# MAGIC 
# MAGIC https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#coefficients-of-additional-regressors
# MAGIC 
# MAGIC https://github.com/facebook/prophet/issues/725

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

## No Features
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
df = slrp_feat[[date_col, actualcol]]

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(df, 0.7)

prophet_model, prophet_testdf, prophet_Evals = run_prophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station)

plot_predsActuals(prophet_testdf, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# #### write dataframe to s3
write_to_s3(prophet_testdf, "BerkeleySlrp_Prophet")

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_Prophet'
write_model_to_s3(prophet_model, modelname)

# COMMAND ----------

## Features
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
df = slrp_feat

slrp_prophet_train, slrp_prophet_test = split2_TrainTest(df, 0.7)

prophet_model2, prophet_testdf2, prophet_Evals2 = run_prophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station, False)

plot_predsActuals(prophet_testdf2, 'yhat', 'rounded', 'Actuals', 'DateTime', 'SlrpEV')

# #### write dataframe to s3
write_to_s3(prophet_testdf2, "BerkeleySlrp_Prophet_Features")

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_Prophet_Features'
write_model_to_s3(prophet_model2, modelname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder

# COMMAND ----------

## no features
station_col = 'Station'
location_ = 'Boulder'
outputcol = 'Ports Available'
date_col = 'Date Time'
df = boulder_feat[[date_col,station_col, outputcol]]

### run prophet mode
boulderprophet, boulder_prophet_res = run_prophet_boulder(df, date_col, station_col, outputcol, location_)
print(boulder_prophet_res)

## write results df to s3
write_to_s3(boulderprophet, "Boulder_Prophet")

# COMMAND ----------

## features
station_col = 'Station'
location_ = 'Boulder'
outputcol = 'Ports Available'
date_col = 'Date Time'
df = boulder_feat.loc[:, (boulder_feat.columns != 'proper_name')]

### run prophet mode
boulderprophet2, boulder_prophet_res2 = run_prophet_boulder(df, date_col, station_col, outputcol, location_, False)
print(boulder_prophet_res2)

## write results df to s3
write_to_s3(boulderprophet2, "Boulder_Prophet_Features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

## No features
station_col = 'Station_Location'
location_ = 'PaloAlto'
outputcol = 'Ports_Available'
date_col = 'datetime_pac'
df = paloalto_feat[[date_col,station_col, outputcol]]

### run prophet mode
paloaltoprophet, paloalto_prophet_res = run_prophet_boulder(df, date_col, station_col, outputcol, location_)
print(paloalto_prophet_res)

## write results df to s3
write_to_s3(paloaltoprophet, "PaloAlto_Prophet")

# COMMAND ----------

### Features
station_col = 'Station_Location'
location_ = 'PaloAlto'
outputcol = 'Ports_Available'
date_col = 'datetime_pac'
df = paloalto_feat

### run prophet mode
paloaltoprophet2, paloalto_prophet_res2 = run_prophet_boulder(df, date_col, station_col, outputcol, location_, False)
print(paloalto_prophet_res2)

## write results df to s3
write_to_s3(paloaltoprophet2, "PaloAlto_Prophet_Features")

# COMMAND ----------

write_to_s3(paloaltoprophet2, "PaloAlto_Prophet_Features")

# COMMAND ----------

## berkeley
berkeley_prophet_nf = {'Berkeley': {'Slrp': {'MSE_raw': 1.306307041774391, 'MSE_round': 1.154853759196124, 'RMSE_raw': 1.1429378993516626, 'RMSE_round': 1.074641223476991}}}
berkeley_prophet_f = {'Berkeley': {'Slrp': {'MSE_raw': 3.329712021866425, 'MSE_round': 3.310963574376458, 'RMSE_raw': 1.8247498518609133, 'RMSE_round': 1.8196053347845675}}}


## boulder no features
boulder_prophet_nf = {'Boulder': {'1505 30th St': {'MSE_raw': 0.9064583262711643, 'MSE_round': 0.8991566481248878, 'RMSE_raw': 0.9520810502636655, 'RMSE_round': 0.9482387084088519}, '600 Baseline Rd': {'MSE_raw': 0.16683676859986327, 'MSE_round': 0.16974699443746635, 'RMSE_raw': 0.4084565688048893, 'RMSE_round': 0.41200363401002466}, '1400 Walnut St': {'MSE_raw': 0.12823550639861805, 'MSE_round': 0.12076081105329266, 'RMSE_raw': 0.3580998553457095, 'RMSE_round': 0.34750656260464013}, '1739 Broadway': {'MSE_raw': 0.1301235118606662, 'MSE_round': 0.13978108738560918, 'RMSE_raw': 0.3607263670161445, 'RMSE_round': 0.37387308994578516}, '3172 Broadway': {'MSE_raw': 0.3003432217908115, 'MSE_round': 0.37879059752377536, 'RMSE_raw': 0.5480357851370762, 'RMSE_round': 0.615459663604184}, '900 Walnut St': {'MSE_raw': 0.41292238479996063, 'MSE_round': 0.5894491297326395, 'RMSE_raw': 0.6425903709206672, 'RMSE_round': 0.7677559050457635}, '1770 13th St': {'MSE_raw': 0.1011463241992596, 'MSE_round': 0.11232729230217119, 'RMSE_raw': 0.31803509900521926, 'RMSE_round': 0.33515264030314784}, '3335 Airport Rd': {'MSE_raw': 0.12113697185614422, 'MSE_round': 0.12757940068185897, 'RMSE_raw': 0.3480473701324925, 'RMSE_round': 0.35718258731614977}, '1100 Walnut': {'MSE_raw': 0.22622746177543823, 'MSE_round': 0.32316526108020815, 'RMSE_raw': 0.4756337475152896, 'RMSE_round': 0.5684762625477058}, '1500 Pearl St': {'MSE_raw': 0.9778041929181932, 'MSE_round': 0.7495065494347748, 'RMSE_raw': 0.9888398216689057, 'RMSE_round': 0.8657404630920139}, '1745 14th street': {'MSE_raw': 1.2134489758458151, 'MSE_round': 0.35618158980800285, 'RMSE_raw': 1.1015666007308933, 'RMSE_round': 0.59680950881165}, '5660 Sioux Dr': {'MSE_raw': 0.12449733378588489, 'MSE_round': 0.1449847478916203, 'RMSE_raw': 0.35284179710726576, 'RMSE_round': 0.38076862776707365}, '2667 Broadway': {'MSE_raw': 0.06990120255619543, 'MSE_round': 0.06818589628566302, 'RMSE_raw': 0.26438835556089724, 'RMSE_round': 0.26112429279112087}, '1360 Gillaspie Dr': {'MSE_raw': 0.23480640720305498, 'MSE_round': 0.2967880854118069, 'RMSE_raw': 0.4845682688776216, 'RMSE_round': 0.544782603807984}, '5565 51st St': {'MSE_raw': 0.018802074916649383, 'MSE_round': 0.019020276332316528, 'RMSE_raw': 0.13712065824174482, 'RMSE_round': 0.13791401789635646}, '5333 Valmont Rd': {'MSE_raw': 0.3055789346463945, 'MSE_round': 0.3908128476583528, 'RMSE_raw': 0.5527919451714131, 'RMSE_round': 0.6251502600642127}, '1100 Spruce St': {'MSE_raw': 0.3005234720565079, 'MSE_round': 0.43154494886057776, 'RMSE_raw': 0.548200211653104, 'RMSE_round': 0.6569208086676641}, '2052 Junction Pl': {'MSE_raw': 0.15785990949821754, 'MSE_round': 0.17172079669836712, 'RMSE_raw': 0.39731588125598194, 'RMSE_round': 0.414392080882788}}}

### Boulder features
boulder_prophet_f = {'Boulder': {'1505 30th St': {'MSE_raw': 0.90944681316505, 'MSE_round': 0.8937735510497039, 'RMSE_raw': 0.953649208653292, 'RMSE_round': 0.9453959757951712}, '600 Baseline Rd': {'MSE_raw': 0.15692234804539437, 'MSE_round': 0.1740534720976135, 'RMSE_raw': 0.39613425507698063, 'RMSE_round': 0.4171971621399329}, '1400 Walnut St': {'MSE_raw': 0.1282582794660727, 'MSE_round': 0.12076081105329266, 'RMSE_raw': 0.35813165102525174, 'RMSE_round': 0.34750656260464013}, '1739 Broadway': {'MSE_raw': 0.13581427240398714, 'MSE_round': 0.13978108738560918, 'RMSE_raw': 0.36852987993375397, 'RMSE_round': 0.37387308994578516}, '3172 Broadway': {'MSE_raw': 0.31784410986922057, 'MSE_round': 0.39996411268616544, 'RMSE_raw': 0.563776648921557, 'RMSE_round': 0.6324271599845831}, '900 Walnut St': {'MSE_raw': 0.4782012611492521, 'MSE_round': 0.6732460075363359, 'RMSE_raw': 0.6915209766516501, 'RMSE_round': 0.820515696093826}, '1770 13th St': {'MSE_raw': 0.10383946250100791, 'MSE_round': 0.11232729230217119, 'RMSE_raw': 0.3222413109782914, 'RMSE_round': 0.33515264030314784}, '3335 Airport Rd': {'MSE_raw': 0.12384824828324655, 'MSE_round': 0.12668221783599498, 'RMSE_raw': 0.35192079831014045, 'RMSE_round': 0.35592445523733685}, '1100 Walnut': {'MSE_raw': 0.22523664070049274, 'MSE_round': 0.305580477301274, 'RMSE_raw': 0.474591024673342, 'RMSE_round': 0.5527933405001131}, '1500 Pearl St': {'MSE_raw': 0.8861821942802427, 'MSE_round': 0.7401758478377893, 'RMSE_raw': 0.9413725055897069, 'RMSE_round': 0.8603347301125239}, '1745 14th street': {'MSE_raw': 1.9273339525728128, 'MSE_round': 0.35618158980800285, 'RMSE_raw': 1.3882845358833371, 'RMSE_round': 0.59680950881165}, '5660 Sioux Dr': {'MSE_raw': 0.12042727204901293, 'MSE_round': 0.1449847478916203, 'RMSE_raw': 0.3470263276021186, 'RMSE_round': 0.38076862776707365}, '2667 Broadway': {'MSE_raw': 0.06797202948147012, 'MSE_round': 0.06818589628566302, 'RMSE_raw': 0.26071445967086315, 'RMSE_round': 0.26112429279112087}, '1360 Gillaspie Dr': {'MSE_raw': 0.23322414608456646, 'MSE_round': 0.29876188767270767, 'RMSE_raw': 0.4829328587749714, 'RMSE_round': 0.5465911522085842}, '5565 51st St': {'MSE_raw': 0.02017458628154388, 'MSE_round': 0.019020276332316528, 'RMSE_raw': 0.14203727074801134, 'RMSE_round': 0.13791401789635646}, '5333 Valmont Rd': {'MSE_raw': 0.3184942908229757, 'MSE_round': 0.42670016149291223, 'RMSE_raw': 0.5643529842421104, 'RMSE_round': 0.6532229033744241}, '1100 Spruce St': {'MSE_raw': 0.2872876043145408, 'MSE_round': 0.3811232729230217, 'RMSE_raw': 0.5359921681466445, 'RMSE_round': 0.6173518226449337}, '2052 Junction Pl': {'MSE_raw': 0.16103189928891737, 'MSE_round': 0.17190023326753992, 'RMSE_raw': 0.4012878010716465, 'RMSE_round': 0.4146085301432424}}}



# Palo alto no features
paloalto_prophet_nf = {'Palo Alto': {'BRYANT': {'MSE_raw': 3.2450610368136736, 'MSE_round': 3.0649560380405525, 'RMSE_raw': 1.8014052949887966, 'RMSE_round': 1.7507015845199183}, 'HAMILTON': {'MSE_raw': 0.7171381641169542, 'MSE_round': 0.7979544231114301, 'RMSE_raw': 0.8468401054018133, 'RMSE_round': 0.8932829468379154}, 'CAMBRIDGE': {'MSE_raw': 2.5556675899804193, 'MSE_round': 2.589090256594294, 'RMSE_raw': 1.5986455485755493, 'RMSE_round': 1.6090650255953902}, 'HIGH': {'MSE_raw': 2.2715132391585557, 'MSE_round': 2.0977929301991747, 'RMSE_raw': 1.5071540197201332, 'RMSE_round': 1.4483759630010347}, 'MPL': {'MSE_raw': 0.6667158919104537, 'MSE_round': 0.7959806208505293, 'RMSE_raw': 0.8165267245537366, 'RMSE_round': 0.892177460402654}, 'TED_THOMPSON': {'MSE_raw': 1.6485984349469993, 'MSE_round': 1.6987260003588731, 'RMSE_raw': 1.2839775835064253, 'RMSE_round': 1.303351832913459}, 'WEBSTER': {'MSE_raw': 1.7725090530683303, 'MSE_round': 1.6938812129912075, 'RMSE_raw': 1.331356095516271, 'RMSE_round': 1.3014919181428701}, 'RINCONADA_LIB': {'MSE_raw': 0.6642276761789367, 'MSE_round': 0.7746276691189664, 'RMSE_raw': 0.8150016418259148, 'RMSE_round': 0.8801293479477698}}}




# features
paloalto_prophet_f = {'Palo Alto': {'BRYANT': {'MSE_raw': 3.08476395062346, 'MSE_round': 3.128476583527723, 'RMSE_raw': 1.7563496094523607, 'RMSE_round': 1.7687500059442327}, 'HAMILTON': {'MSE_raw': 0.680066708407249, 'MSE_round': 0.7198995155212632, 'RMSE_raw': 0.8246615720447079, 'RMSE_round': 0.8484689243108808}, 'CAMBRIDGE': {'MSE_raw': 3.090373444500062, 'MSE_round': 3.0502422393683832, 'RMSE_raw': 1.7579458024922332, 'RMSE_round': 1.7464942712097236}, 'HIGH': {'MSE_raw': 3.205390991098312, 'MSE_round': 2.535079849273282, 'RMSE_raw': 1.790360575721637, 'RMSE_round': 1.5921934082495386}, 'MPL': {'MSE_raw': 0.7584239167315235, 'MSE_round': 0.9296608648842634, 'RMSE_raw': 0.8708753738230997, 'RMSE_round': 0.9641892266999582}, 'TED_THOMPSON': {'MSE_raw': 1.6350460347616003, 'MSE_round': 1.7172079669836713, 'RMSE_raw': 1.2786891861439982, 'RMSE_round': 1.3104228199263286}, 'WEBSTER': {'MSE_raw': 1.7318025845630456, 'MSE_round': 1.6321550331957653, 'RMSE_raw': 1.3159797052246078, 'RMSE_round': 1.2775582308434184}, 'RINCONADA_LIB': {'MSE_raw': 1.308694473182666, 'MSE_round': 1.5024223936838328, 'RMSE_raw': 1.143981850023271, 'RMSE_round': 1.225733410527686}}}



prophet_nf = dict()
prophet_nf.update(berkeley_prophet_nf)
prophet_nf.update(paloalto_prophet_nf)
prophet_nf.update(boulder_prophet_nf)

prophet_nf = pd.DataFrame.from_dict({(i,j): prophet_nf[i][j] 
                           for i in prophet_nf.keys() 
                           for j in prophet_nf[i].keys()},
                       orient='index')

prophet_nf.reset_index(inplace=True)

prophet_nf = prophet_nf.rename(columns={"level_0":"city", "level_1":"station", "RMSE":"RMSE_round"})
prophet_nf['Model'] ='Prophet' 
prophet_nf['Features'] = 'No'
# # prophet_nf



prophet_f = dict()
prophet_f.update(berkeley_prophet_f)
prophet_f.update(paloalto_prophet_f)
prophet_f.update(boulder_prophet_f)

prophet_f = pd.DataFrame.from_dict({(i,j): prophet_f[i][j] 
                           for i in prophet_f.keys() 
                           for j in prophet_f[i].keys()},
                       orient='index')

prophet_f.reset_index(inplace=True)

prophet_f = prophet_f.rename(columns={"level_0":"city", "level_1":"station", "RMSE":"RMSE_round"})
prophet_f['Model'] ='Prophet' 
prophet_f['Features'] = 'Yes'

print(prophet_nf.shape)
print(prophet_f.shape)


prophet = pd.concat([prophet_nf, prophet_f], ignore_index=True, sort=False)
print(prophet.shape)

prophet = prophet.sort_values(by=['city', 'station'], ignore_index=True)
prophet = prophet.rename(columns={"MSE_raw":"MSE_Raw", "RMSE_raw":"RMSE_Raw"})
prophet



# COMMAND ----------

# MAGIC %md
# MAGIC # LSTM Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berkeley SlrpEV

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single Step predictions , n_outputs = 1

# COMMAND ----------

## No features
date_col = 'DateTime'
actualcol= 'Ports Available'
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
df, evals = plot_predictions(lstm1_modelslrp, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture agg metrics
lstm1_slrp_metrics.update(evals)

#### add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm1_slrp_metrics)
df.head()

#{'Slrp': {'MSE_Raw': 0.14015938871232805, 'MSE_round': 0.16337522441651706, 'RMSE_Raw': 0.3743786702155026, 'RMSE': 0.404197011884696}}

# #### write dataframe to s3
write_to_s3(df, "Berkeley_LSTM_1StepNoFeatures")
Berkeley_LSTM_1StepNoFeatures = df

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_LSTM1'
write_model_to_s3(lstm1_modelslrp, modelname)

# COMMAND ----------

## features
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
n_inputs = 12
n_outputs = 1
loc_ = 'Berkeley'

df = slrp_feat
lstm1_slrp_metrics2 = dict()



##### split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)

#### split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

### run model
lstm1_modelslrp2 = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

### plot and get results df and metrics
df, evals = plot_predictions(lstm1_modelslrp2, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture agg metrics
lstm1_slrp_metrics2.update(evals)

#### add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm1_slrp_metrics2)
df.head()

#{'Slrp': {'MSE_Raw': 0.14015938871232805, 'MSE_round': 0.16337522441651706, 'RMSE_Raw': 0.3743786702155026, 'RMSE': 0.404197011884696}}


# #### write dataframe to s3
write_to_s3(df, "Berkeley_LSTM_1StepFeatures")
Berkeley_LSTM_1StepFeatures = df

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_LSTM1_Features'
write_model_to_s3(lstm1_modelslrp2, modelname)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multistep Predictions

# COMMAND ----------

## No features
date_col = 'DateTime'
actualcol= 'Ports Available'
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

df, evals = plot_predictions(lstm6_modelslrp, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
lstm6_slrp_metrics.update(evals)

# add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm6_slrp_metrics)


# #### write dataframe to s3
write_to_s3(df, "Berkeley_LSTM_6StepNoFeatures")
Berkeley_LSTM_6StepNoFeatures = df

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_LSTM6'
write_model_to_s3(lstm6_modelslrp, modelname)

# COMMAND ----------

## features
date_col = 'DateTime'
actualcol= 'Ports Available'
station = 'Slrp'
n_inputs = 12
n_outputs = 6
loc_ = 'Berkeley'

df = slrp_feat
lstm6_slrp_metrics2 = dict()


# split into array format
X, y, y_dates = dfSplit_Xy(df, date_col, n_inputs, n_outputs)

# split into train, val, and test
X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)

print()
print(station)

lstm6_modelslrp2 = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)

df, evals = plot_predictions(lstm6_modelslrp2, X_test, y_test, df, station, train_end, date_col, y_dates, 0, y_test.shape[0])

# capture metrics
lstm6_slrp_metrics2.update(evals)

# add additional dataframe columns for visualizations
df['Location'] = loc_
df['SiteName'] = station
df['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])

print(lstm6_slrp_metrics2)


# #### write dataframe to s3
write_to_s3(df, "Berkeley_LSTM_6StepFeatures")
Berkeley_LSTM_6StepFeatures = df

### write model to s3
modelname = 'BatchModel_BerkeleySlrp_LSTM6_Features'
write_model_to_s3(lstm6_modelslrp2, modelname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Palo Alto

# COMMAND ----------

def lstm_multistation(n_inputs, n_outputs, stationcol, date_col, actual_col, loc_, df, nofeatures = True):
    stations = df[stationcol].unique()
    
    # empty to save results
    lstm_results_df = pd.DataFrame()
    lstm_metrics = dict()


    for station in stations:
        stationdf = pd.DataFrame(df[df[stationcol] == station]).drop([stationcol], axis = 1)
        stationdf = stationdf.sort_values(by=[date_col])
    
        # split into array format
        X, y, y_dates = dfSplit_Xy(stationdf, date_col, n_inputs, n_outputs)
    
        # split into train, val, and test
        X_train, y_train, X_val, y_val, X_test, y_test, train_end, y_dates = split_train_test(X, y, y_dates)
    
        print()
        print(station)
    
        lstm_model = run_lstm(X_train.shape[1], X_train.shape[2], y_train.shape[1], X_train, y_train, X_val, y_val, n_epochs = 10)
        
        print('Model Created')
        
        # get station results
        resultsdf, evals = plot_predictions(lstm_model, X_test, y_test, stationdf, station, train_end, date_col, y_dates, 0, y_test.shape[0])
    
        # capture metrics
        lstm_metrics.update(evals)
    
        # add additional dataframe columns for visualizations
        resultsdf['Location'] = loc_
        resultsdf['SiteName'] = station
        resultsdf['StepsOut'] = np.tile(np.arange(1, y_test.shape[1]+1, 1),  y_test.shape[0])
    
        # append each station df to results df
        lstm_results_df = lstm_results_df.append(resultsdf)
        
        ### write model to s3
        # check if features
        if nofeatures:
            modelname = 'BatchModel_'+ str(loc_) + str(station) + '_LSTM'+str(n_outputs)
        else: 
            modelname = 'BatchModel_'+ str(loc_) + str(station) + '_LSTM'+str(n_outputs) +"features"
        # write
        write_model_to_s3(lstm_model, modelname)

    print(lstm_metrics)
    lstm_results_df.head()
    
    return lstm_results_df, lstm_metrics 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single Step Predictions

# COMMAND ----------

## No Features
stationcol_ = 'Station_Location'
date_col_ = 'datetime_pac'
actual_col_ = 'Ports_Available'
n_inputs_ = 12
n_outputs_ = 1
loc__ = 'PaloAlto'

# df_ = paloalto_feat[[date_col_, stationcol_, actual_col_]]

## run model
lstm1_pa_results_df, lstm1_pa_metrics = lstm_multistation(n_inputs_, n_outputs_, stationcol_, date_col_, actual_col_, loc__, 
                                                          paloalto_feat[[date_col_, stationcol_, actual_col_]])

# #### write dataframe to s3
write_to_s3(lstm1_pa_results_df, "PaloAlto_LSTM_1StepNoFeatures")

# COMMAND ----------

## Features
stationcolumn = 'Station_Location'
date_column = 'datetime_pac'
actualcol1 = 'Ports_Available'
n_inputs1 = 12
n_outputs1 = 1
loc_1 = 'PaloAlto'

# df = paloalto_feat

## run model
lstm1_pa_results_df2, lstm1_pa_metrics2 = lstm_multistation(n_inputs1, n_outputs1, stationcolumn, date_column, actualcol1, loc_1, paloalto_feat, False)


# #### write dataframe to s3
write_to_s3(lstm1_pa_results_df2, "PaloAlto_LSTM_1StepFeatures")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Multistep Predictions
# MAGIC 6 steps, 10 minutes to 1 hour

# COMMAND ----------

## No Features
stationcol1 = 'Station_Location'
date_col1 = 'datetime_pac'
actualcol1 = 'Ports_Available'
n_inputs1 = 12
n_outputs1 = 6
loc_1 = 'PaloAlto'

df = paloalto_feat[[date_col1, stationcol1, actualcol1]]


## run model
lstm6_pa_results_df, lstm6_pa_metrics = lstm_multistation(n_inputs1, n_outputs1, stationcol1, date_col1, actualcol1, loc_1, df)

# #### write dataframe to s3
write_to_s3(lstm6_pa_results_df, "PaloAlto_LSTM_6StepNoFeatures")

# COMMAND ----------

## Features
stationcol1 = 'Station_Location'
date_col1 = 'datetime_pac'
actualcol1 = 'Ports_Available'
n_inputs1 = 12
n_outputs1 = 6
loc_1 = 'PaloAlto'

df = paloalto_feat

## run model
lstm6_pa_results_df2, lstm6_pa_metrics2 = lstm_multistation(n_inputs1, n_outputs1, stationcol1, date_col1, actualcol1, loc_1, df, False)

# #### write dataframe to s3
write_to_s3(lstm6_pa_results_df2, "PaloAlto_LSTM_6StepFeatures")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Boulder CO

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single Step Predictions

# COMMAND ----------

## No Features
station_col = 'Station'
date_col = 'Date Time'
actualcol = 'Ports Available'
n_inputs = 12
n_outputs = 1
loc_ = 'Boulder'

df = boulder_feat[[date_col, station_col, actualcol]]


## run model
lstm1_bo_results_df, lstm1_bo_metrics = lstm_multistation(n_inputs, n_outputs, station_col, date_col, actualcol, loc_, df)

# #### write dataframe to s3
write_to_s3(lstm1_bo_results_df, "Boulder_LSTM_1StepNoFeatures")

# COMMAND ----------

## Features
station_col = 'Station'
date_col = 'Date Time'
actualcol = 'Ports Available'
n_inputs = 12
n_outputs = 1
loc_ = 'Boulder'

df = boulder_feat.loc[:, (boulder_feat.columns != 'proper_name')]


## run model
lstm1_bo_results_df2, lstm1_bo_metrics2 = lstm_multistation(n_inputs, n_outputs, station_col, date_col, actualcol, loc_, df, False)

# #### write dataframe to s3
write_to_s3(lstm1_bo_results_df2, "Boulder_LSTM_1StepFeatures")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multistep Predictions
# MAGIC 6 steps

# COMMAND ----------

## No Features
station_col = 'Station'
date_col = 'Date Time'
actualcol = 'Ports Available'
n_inputs = 12
n_outputs = 6
loc_ = 'Boulder'

df = boulder_feat[[date_col, station_col, actualcol]]


## run model
lstm6_bo_results_df, lstm6_bo_metrics = lstm_multistation(n_inputs, n_outputs, station_col, date_col, actualcol, loc_, df)

# #### write dataframe to s3
write_to_s3(lstm6_bo_results_df, "Boulder_LSTM_6StepNoFeatures")

# COMMAND ----------

## Features
station_col = 'Station'
date_col = 'Date Time'
actualcol = 'Ports Available'
n_inputs = 12
n_outputs = 6
loc_ = 'Boulder'

df = boulder_feat.loc[:, (boulder_feat.columns != 'proper_name')]


## run model
lstm6_bo_results_df2, lstm6_bo_metrics2 = lstm_multistation(n_inputs, n_outputs, station_col, date_col, actualcol, loc_, df, False)

# #### write dataframe to s3
write_to_s3(lstm6_bo_results_df2, "Boulder_LSTM_6StepFeatures")

# COMMAND ----------

## berkeley 1 step

berkeley_lstm1_nf =  {'Berkeley': {'Slrp': {'MSE_Raw': 0.1546694032730616, 'MSE_round': 0.16481149012567325, 'RMSE_Raw': 0.39328031132140545, 'RMSE': 0.4059698143035677}}} #nof
berkeley_lstm1_f = {'Berkeley': {'Slrp': {'MSE_Raw': 0.19958821234990962, 'MSE_round': 0.22262118491921004, 'RMSE_Raw': 0.4467529656867536, 'RMSE': 0.47182749487414366}}} #feat

## 6 steps
berkeley_lstm6_nf = {'Berkeley': {'Slrp': {'MSE_Raw': 0.35581580161441784, 'MSE_round': 0.37895114942528735, 'RMSE_Raw': 0.5965029770373471, 'RMSE': 0.6155900822993231}}}
berkeley_lstm6_f = {'Berkeley':{'Slrp': {'MSE_Raw': 0.39345046206407697, 'MSE_round': 0.4339080459770115, 'RMSE_Raw': 0.6272562969505184, 'RMSE': 0.658716969552942}}}



### PA
# 1step
#no features
paloalto_lstm1_nf = {'Palo Alto': {'BRYANT': {'MSE_Raw': 0.3322590452193334, 'MSE_round': 0.36912028725314183, 'RMSE_Raw': 0.5764191575748792, 'RMSE': 0.6075527032720222}, 'HAMILTON': {'MSE_Raw': 0.17354608493419882, 'MSE_round': 0.18168761220825852, 'RMSE_Raw': 0.4165886279463217, 'RMSE': 0.42624829877462095}, 'CAMBRIDGE': {'MSE_Raw': 0.3478720015832584, 'MSE_round': 0.3497307001795332, 'RMSE_Raw': 0.5898067493537679, 'RMSE': 0.5913803346236103}, 'HIGH': {'MSE_Raw': 0.29346470986535744, 'MSE_round': 0.2883303411131059, 'RMSE_Raw': 0.5417238317310376, 'RMSE': 0.5369640035543406}, 'MPL': {'MSE_Raw': 0.1395281849355635, 'MSE_round': 0.1421903052064632, 'RMSE_Raw': 0.37353471717574455, 'RMSE': 0.37708129787416295}, 'TED_THOMPSON': {'MSE_Raw': 0.1806262709302046, 'MSE_round': 0.1741472172351885, 'RMSE_Raw': 0.42500149520937525, 'RMSE': 0.41730949813680074}, 'WEBSTER': {'MSE_Raw': 0.26999869883965183, 'MSE_round': 0.2718132854578097, 'RMSE_Raw': 0.5196139902270259, 'RMSE': 0.5213571572902876}, 'RINCONADA_LIB': {'MSE_Raw': 0.0928559029210134, 'MSE_round': 0.09443447037701976, 'RMSE_Raw': 0.3047226655846483, 'RMSE': 0.3073019205553713}}}


#features
paloalto_lstm1_f = {'Palo Alto': {'BRYANT': {'MSE_Raw': 0.33572485124016355, 'MSE_round': 0.37378815080789946, 'RMSE_Raw': 0.5794176828852944, 'RMSE': 0.6113821642867082}, 'HAMILTON': {'MSE_Raw': 0.17584660393163168, 'MSE_round': 0.1910233393177738, 'RMSE_Raw': 0.4193406776496071, 'RMSE': 0.4370621687103264}, 'CAMBRIDGE': {'MSE_Raw': 0.4639497948868851, 'MSE_round': 0.503411131059246, 'RMSE_Raw': 0.6811386018182239, 'RMSE': 0.7095147151816134}, 'HIGH': {'MSE_Raw': 0.2713745110366625, 'MSE_round': 0.27971274685816877, 'RMSE_Raw': 0.52093618710612, 'RMSE': 0.5288787638563008}, 'MPL': {'MSE_Raw': 0.1365850769171972, 'MSE_round': 0.14326750448833034, 'RMSE_Raw': 0.36957418323957264, 'RMSE': 0.3785069411362628}, 'TED_THOMPSON': {'MSE_Raw': 0.22421727733728677, 'MSE_round': 0.2542190305206463, 'RMSE_Raw': 0.4735158680944988, 'RMSE': 0.5042013789356851}, 'WEBSTER': {'MSE_Raw': 0.24725046312201085, 'MSE_round': 0.26463195691202873, 'RMSE_Raw': 0.4972428613082453, 'RMSE': 0.5144239077959234}, 'RINCONADA_LIB': {'MSE_Raw': 0.0997399179225467, 'MSE_round': 0.09730700179533214, 'RMSE_Raw': 0.31581627241569854, 'RMSE': 0.31194070237038984}}}

#6steps
#nofeatures
paloalto_lstm6_nf = {'Palo Alto': {'BRYANT': {'MSE_Raw': 1.7404180745383109, 'MSE_round': 1.7009698275862069, 'RMSE_Raw': 1.3192490570541677, 'RMSE': 1.3042123399148648}, 'HAMILTON': {'MSE_Raw': 0.5118422840463651, 'MSE_round': 0.5648347701149425, 'RMSE_Raw': 0.7154315369386264, 'RMSE': 0.7515549015973102}, 'CAMBRIDGE': {'MSE_Raw': 1.2040678839325443, 'MSE_round': 1.2388050766283525, 'RMSE_Raw': 1.0973002706335875, 'RMSE': 1.1130162068129792}, 'HIGH': {'MSE_Raw': 0.9689644053225493, 'MSE_round': 0.999161877394636, 'RMSE_Raw': 0.9843598962384384, 'RMSE': 0.9995808508543148}, 'MPL': {'MSE_Raw': 0.388823823571824, 'MSE_round': 0.42738266283524906, 'RMSE_Raw': 0.6235573939677277, 'RMSE': 0.6537451054006057}, 'TED_THOMPSON': {'MSE_Raw': 0.6069105184855702, 'MSE_round': 0.6392480842911877, 'RMSE_Raw': 0.7790446190595056, 'RMSE': 0.7995299145692972}, 'WEBSTER': {'MSE_Raw': 0.7900008718149473, 'MSE_round': 0.8225574712643678, 'RMSE_Raw': 0.8888199321656481, 'RMSE': 0.9069495417410871}, 'RINCONADA_LIB': {'MSE_Raw': 0.2894968533645909, 'MSE_round': 0.327286877394636, 'RMSE_Raw': 0.5380491179851435, 'RMSE': 0.5720899207245623}}}

#features
paloalto_lstm6_f = {'Palo Alto': {'BRYANT': {'MSE_Raw': 3.569435162819382, 'MSE_round': 3.8144755747126435, 'RMSE_Raw': 1.8892948850879214, 'RMSE': 1.9530682463018654}, 'HAMILTON': {'MSE_Raw': 0.43194379572677194, 'MSE_round': 0.4966475095785441, 'RMSE_Raw': 0.657224311576171, 'RMSE': 0.7047322254434971}, 'CAMBRIDGE': {'MSE_Raw': 1.1711175978355068, 'MSE_round': 1.2486230842911878, 'RMSE_Raw': 1.0821818691123535, 'RMSE': 1.1174180436574255}, 'HIGH': {'MSE_Raw': 0.844905413312375, 'MSE_round': 0.897389846743295, 'RMSE_Raw': 0.9191873657271269, 'RMSE': 0.9473066276255514}, 'MPL': {'MSE_Raw': 0.37678501768442885, 'MSE_round': 0.4228927203065134, 'RMSE_Raw': 0.6138281662521107, 'RMSE': 0.6503020223761521}, 'TED_THOMPSON': {'MSE_Raw': 0.6987152539125446, 'MSE_round': 0.7710129310344828, 'RMSE_Raw': 0.8358918912829246, 'RMSE': 0.8780734200706014}, 'WEBSTER': {'MSE_Raw': 0.6809712534826488, 'MSE_round': 0.7352729885057471, 'RMSE_Raw': 0.8252098239130753, 'RMSE': 0.8574806053233782}, 'RINCONADA_LIB': {'MSE_Raw': 0.2850644505878719, 'MSE_round': 0.33075909961685823, 'RMSE_Raw': 0.533914272695413, 'RMSE': 0.5751165965409607}}}




#### boulder
# 1step
#nodeatures
boulder_lstm1_nf = {'Boulder': {'1505 30th St': {'MSE_Raw': 0.12042340634889744, 'MSE_round': 0.11310592459605028, 'RMSE_Raw': 0.3470207578069321, 'RMSE': 0.3363122427091382}, '600 Baseline Rd': {'MSE_Raw': 0.03547248604529577, 'MSE_round': 0.03303411131059246, 'RMSE_Raw': 0.18834140820673442, 'RMSE': 0.1817528852882189}, '1400 Walnut St': {'MSE_Raw': 0.02218794807424331, 'MSE_round': 0.02585278276481149, 'RMSE_Raw': 0.14895619515227726, 'RMSE': 0.1607880056621497}, '1739 Broadway': {'MSE_Raw': 0.023320129828827214, 'MSE_round': 0.022980251346499104, 'RMSE_Raw': 0.15270929843603898, 'RMSE': 0.15159238551622276}, '3172 Broadway': {'MSE_Raw': 0.0646636534862682, 'MSE_round': 0.06104129263913824, 'RMSE_Raw': 0.25429049035752044, 'RMSE': 0.24706536106694163}, '900 Walnut St': {'MSE_Raw': 0.053236526103199934, 'MSE_round': 0.05098743267504488, 'RMSE_Raw': 0.23073041867772862, 'RMSE': 0.22580396957326698}, '1770 13th St': {'MSE_Raw': 0.02184354287660462, 'MSE_round': 0.027648114901256734, 'RMSE_Raw': 0.1477956118313552, 'RMSE': 0.166277223038084}, '3335 Airport Rd': {'MSE_Raw': 0.014319529509875062, 'MSE_round': 0.01867145421903052, 'RMSE_Raw': 0.11966423655326207, 'RMSE': 0.13664352973716143}, '1100 Walnut': {'MSE_Raw': 0.04288015187825327, 'MSE_round': 0.04416517055655296, 'RMSE_Raw': 0.20707523241144332, 'RMSE': 0.21015511070766982}, '1500 Pearl St': {'MSE_Raw': 0.04877897867078649, 'MSE_round': 0.06068222621184919, 'RMSE_Raw': 0.22085963567566277, 'RMSE': 0.24633762646386198}, '1745 14th street': {'MSE_Raw': 0.0035189268715493044, 'MSE_round': 0.004308797127468581, 'RMSE_Raw': 0.05932054341920094, 'RMSE': 0.06564142843866655}, '5660 Sioux Dr': {'MSE_Raw': 0.022339669704398078, 'MSE_round': 0.022621184919210054, 'RMSE_Raw': 0.14946461020722623, 'RMSE': 0.1504034072726082}, '2667 Broadway': {'MSE_Raw': 0.0026175089504417146, 'MSE_round': 0.003231597845601436, 'RMSE_Raw': 0.05116159644148836, 'RMSE': 0.05684714456858353}, '1360 Gillaspie Dr': {'MSE_Raw': 0.03434795912396282, 'MSE_round': 0.03267504488330341, 'RMSE_Raw': 0.18533202401086227, 'RMSE': 0.18076239897529411}, '5565 51st St': {'MSE_Raw': 0.01792945273997833, 'MSE_round': 0.020107719928186715, 'RMSE_Raw': 0.13390090641955466, 'RMSE': 0.14180169226136446}, '5333 Valmont Rd': {'MSE_Raw': 0.07180971680443442, 'MSE_round': 0.08797127468581688, 'RMSE_Raw': 0.2679733509221289, 'RMSE': 0.29659951902492504}, '1100 Spruce St': {'MSE_Raw': 0.05729140698683341, 'MSE_round': 0.05601436265709156, 'RMSE_Raw': 0.23935623448498977, 'RMSE': 0.23667353603031235}, '2052 Junction Pl': {'MSE_Raw': 0.03162828404036058, 'MSE_round': 0.04021543985637343, 'RMSE_Raw': 0.17784342563153854, 'RMSE': 0.2005378763634776}}}

#features
boulder_lstm1_f = {'Boulder': {'1505 30th St': {'MSE_Raw': 0.13259828524769218, 'MSE_round': 0.1414721723518851, 'RMSE_Raw': 0.364140474607935, 'RMSE': 0.3761278670238156}, '600 Baseline Rd': {'MSE_Raw': 0.03645552107034044, 'MSE_round': 0.03159784560143627, 'RMSE_Raw': 0.19093328958131014, 'RMSE': 0.17775782852363006}, '1400 Walnut St': {'MSE_Raw': 0.02607352060156716, 'MSE_round': 0.02657091561938959, 'RMSE_Raw': 0.16147297173696643, 'RMSE': 0.16300587602718372}, '1739 Broadway': {'MSE_Raw': 0.10964403935501936, 'MSE_round': 0.13177737881508078, 'RMSE_Raw': 0.3311254133331046, 'RMSE': 0.36301154088414433}, '3172 Broadway': {'MSE_Raw': 0.05917879485147923, 'MSE_round': 0.05709156193895871, 'RMSE_Raw': 0.24326692099724373, 'RMSE': 0.2389384061614179}, '900 Walnut St': {'MSE_Raw': 0.057953978003101134, 'MSE_round': 0.06750448833034112, 'RMSE_Raw': 0.24073632464399952, 'RMSE': 0.2598162587875153}, '1770 13th St': {'MSE_Raw': 2.296939120131761, 'MSE_round': 3.594973070017953, 'RMSE_Raw': 1.5155656106324664, 'RMSE': 1.896041420965785}, '3335 Airport Rd': {'MSE_Raw': 0.015544984367610159, 'MSE_round': 0.017953321364452424, 'RMSE_Raw': 0.12467952665778836, 'RMSE': 0.1339900047184581}, '1100 Walnut': {'MSE_Raw': 0.04957216706166855, 'MSE_round': 0.05421903052064632, 'RMSE_Raw': 0.2226480789534654, 'RMSE': 0.23284980249217804}, '1500 Pearl St': {'MSE_Raw': 0.06325546200444368, 'MSE_round': 0.06283662477558348, 'RMSE_Raw': 0.2515063856136533, 'RMSE': 0.250672345454347}, '1745 14th street': {'MSE_Raw': 0.004619294019943933, 'MSE_round': 0.0039497307001795335, 'RMSE_Raw': 0.06796538839691812, 'RMSE': 0.06284688297902716}, '5660 Sioux Dr': {'MSE_Raw': 0.021588104297898196, 'MSE_round': 0.021903052064631955, 'RMSE_Raw': 0.1469289089930848, 'RMSE': 0.14799679748099942}, '2667 Broadway': {'MSE_Raw': 0.005046211763775287, 'MSE_round': 0.0021543985637342907, 'RMSE_Raw': 0.07103669308023346, 'RMSE': 0.046415499175752606}, '1360 Gillaspie Dr': {'MSE_Raw': 0.03909031958787962, 'MSE_round': 0.03770197486535009, 'RMSE_Raw': 0.19771271984341224, 'RMSE': 0.1941699638598877}, '5565 51st St': {'MSE_Raw': 0.022017128322618602, 'MSE_round': 0.026929982046678635, 'RMSE_Raw': 0.14838169807162405, 'RMSE': 0.16410357109666637}, '5333 Valmont Rd': {'MSE_Raw': 0.12091577870444811, 'MSE_round': 0.1393177737881508, 'RMSE_Raw': 0.3477294619448403, 'RMSE': 0.37325296219608334}, '1100 Spruce St': {'MSE_Raw': 0.05999675258235523, 'MSE_round': 0.05780969479353681, 'RMSE_Raw': 0.2449423454251127, 'RMSE': 0.24043646727053866}, '2052 Junction Pl': {'MSE_Raw': 0.03193327357659258, 'MSE_round': 0.0348294434470377, 'RMSE_Raw': 0.1786988348495663, 'RMSE': 0.18662648109804167}}}


#6step
#nofeatures
boulder_lstm6_nf = {'Boulder': {'1505 30th St': {'MSE_Raw': 0.3253508644786578, 'MSE_round': 0.34812021072796934, 'RMSE_Raw': 0.5703953580444513, 'RMSE': 0.5900171274869649}, '600 Baseline Rd': {'MSE_Raw': 0.09233811925277859, 'MSE_round': 0.10530411877394635, 'RMSE_Raw': 0.3038718796676958, 'RMSE': 0.3245059610761355}, '1400 Walnut St': {'MSE_Raw': 0.05208116183723117, 'MSE_round': 0.06208093869731801, 'RMSE_Raw': 0.22821297473463503, 'RMSE': 0.24916046776589182}, '1739 Broadway': {'MSE_Raw': 2.2300754812890964, 'MSE_round': 3.497844827586207, 'RMSE_Raw': 1.4933437250978412, 'RMSE': 1.8702526106347792}, '3172 Broadway': {'MSE_Raw': 0.18289194849339369, 'MSE_round': 0.20988984674329503, 'RMSE_Raw': 0.4276586822378258, 'RMSE': 0.458137366674336}, '900 Walnut St': {'MSE_Raw': 0.16103575132122644, 'MSE_round': 0.18624281609195403, 'RMSE_Raw': 0.4012926006310438, 'RMSE': 0.4315585894081521}, '1770 13th St': {'MSE_Raw': 0.06263250181884356, 'MSE_round': 0.07938218390804598, 'RMSE_Raw': 0.2502648633325173, 'RMSE': 0.28174844082629097}, '3335 Airport Rd': {'MSE_Raw': 0.03542647002190117, 'MSE_round': 0.040948275862068964, 'RMSE_Raw': 0.18821920736710473, 'RMSE': 0.20235680335009487}, '1100 Walnut': {'MSE_Raw': 0.13109256418499793, 'MSE_round': 0.1453544061302682, 'RMSE_Raw': 0.3620670713900919, 'RMSE': 0.3812537293329315}, '1500 Pearl St': {'MSE_Raw': 0.12882060644082885, 'MSE_round': 0.148227969348659, 'RMSE_Raw': 0.3589158765516355, 'RMSE': 0.38500385627764694}, '1745 14th street': {'MSE_Raw': 0.0073224944331188434, 'MSE_round': 0.008500957854406131, 'RMSE_Raw': 0.0855715749131617, 'RMSE': 0.09220063912146234}, '5660 Sioux Dr': {'MSE_Raw': 0.06491000744269106, 'MSE_round': 0.07291666666666667, 'RMSE_Raw': 0.2547744246243941, 'RMSE': 0.27003086243366087}, '2667 Broadway': {'MSE_Raw': 0.0039988629616142515, 'MSE_round': 0.004669540229885058, 'RMSE_Raw': 0.06323656348675386, 'RMSE': 0.06833403419881676}, '1360 Gillaspie Dr': {'MSE_Raw': 0.10664493006902132, 'MSE_round': 0.12362308429118773, 'RMSE_Raw': 0.32656535344249443, 'RMSE': 0.35160074557825916}, '5565 51st St': {'MSE_Raw': 0.02770552420288579, 'MSE_round': 0.028735632183908046, 'RMSE_Raw': 0.16644976480273496, 'RMSE': 0.1695158759052026}, '5333 Valmont Rd': {'MSE_Raw': 0.16951040736409162, 'MSE_round': 0.20504070881226052, 'RMSE_Raw': 0.41171641619455934, 'RMSE': 0.4528142100379145}, '1100 Spruce St': {'MSE_Raw': 0.16432289774586173, 'MSE_round': 0.1904932950191571, 'RMSE_Raw': 0.40536760816061973, 'RMSE': 0.43645537574780435}, '2052 Junction Pl': {'MSE_Raw': 0.0785375198688267, 'MSE_round': 0.0963242337164751, 'RMSE_Raw': 0.2802454636007989, 'RMSE': 0.3103614565574712}}}


## features
boulder_lstm6_f = {'Boulder': {'1505 30th St': {'MSE_Raw': 0.3426070271408745, 'MSE_round': 0.38553639846743293, 'RMSE_Raw': 0.5853264278510535, 'RMSE': 0.6209157740526753}, '600 Baseline Rd': {'MSE_Raw': 0.09919570284123788, 'MSE_round': 0.11925287356321838, 'RMSE_Raw': 0.3149534931402379, 'RMSE': 0.345330093625242}, '1400 Walnut St': {'MSE_Raw': 0.053222916428166865, 'MSE_round': 0.06675047892720307, 'RMSE_Raw': 0.23070092420310515, 'RMSE': 0.25836114051304826}, '1739 Broadway': {'MSE_Raw': 0.1629021604363653, 'MSE_round': 0.2192887931034483, 'RMSE_Raw': 0.40361139780284366, 'RMSE': 0.4682828131625677}, '3172 Broadway': {'MSE_Raw': 0.18839226921640498, 'MSE_round': 0.2246168582375479, 'RMSE_Raw': 0.43404178280023337, 'RMSE': 0.4739376100686122}, '900 Walnut St': {'MSE_Raw': 0.16672981032312392, 'MSE_round': 0.187080938697318, 'RMSE_Raw': 0.4083256180098475, 'RMSE': 0.4325285409048957}, '1770 13th St': {'MSE_Raw': 0.10872681026279839, 'MSE_round': 0.1434985632183908, 'RMSE_Raw': 0.32973748689343524, 'RMSE': 0.3788120420715144}, '3335 Airport Rd': {'MSE_Raw': 0.030466212207062707, 'MSE_round': 0.03562021072796935, 'RMSE_Raw': 0.17454573099065673, 'RMSE': 0.1887331733637978}, '1100 Walnut': {'MSE_Raw': 0.1355665982226208, 'MSE_round': 0.15193965517241378, 'RMSE_Raw': 0.3681936966090278, 'RMSE': 0.3897943755012555}, '1500 Pearl St': {'MSE_Raw': 0.8524487263942937, 'MSE_round': 0.7738266283524904, 'RMSE_Raw': 0.9232814989992455, 'RMSE': 0.8796741603301136}, '1745 14th street': {'MSE_Raw': 0.012097599934339828, 'MSE_round': 0.008800287356321839, 'RMSE_Raw': 0.10998909006960567, 'RMSE': 0.09380984679830705}, '5660 Sioux Dr': {'MSE_Raw': 0.07084767239257715, 'MSE_round': 0.08369252873563218, 'RMSE_Raw': 0.2661722607496453, 'RMSE': 0.28929661030788484}, '2667 Broadway': {'MSE_Raw': 0.00612849234436516, 'MSE_round': 0.005387931034482759, 'RMSE_Raw': 0.07828468780269332, 'RMSE': 0.07340252743933794}, '1360 Gillaspie Dr': {'MSE_Raw': 0.10188880847768336, 'MSE_round': 0.12134818007662836, 'RMSE_Raw': 0.3192002639060365, 'RMSE': 0.34835065677651356}, '5565 51st St': {'MSE_Raw': 0.026387518274417914, 'MSE_round': 0.02747844827586207, 'RMSE_Raw': 0.16244235369637414, 'RMSE': 0.16576624588818456}, '5333 Valmont Rd': {'MSE_Raw': 0.3042947045527833, 'MSE_round': 0.3887691570881226, 'RMSE_Raw': 0.5516291367873739, 'RMSE': 0.623513558062792}, '1100 Spruce St': {'MSE_Raw': 0.15563337954208326, 'MSE_round': 0.18396791187739464, 'RMSE_Raw': 0.39450396644657865, 'RMSE': 0.42891480724893916}, '2052 Junction Pl': {'MSE_Raw': 0.07924594927680424, 'MSE_round': 0.09446839080459771, 'RMSE_Raw': 0.2815065705748344, 'RMSE': 0.3073571063186887}}}


                   
                   
                   
lstm1_nf = dictstodf(berkeley_lstm1_nf, paloalto_lstm1_nf, boulder_lstm1_nf, 'LSTM 1-step', 'No')
lstm1_f = dictstodf(berkeley_lstm1_f, paloalto_lstm1_f, boulder_lstm1_f, 'LSTM 1-step', 'Yes')
lstm6_nf = dictstodf(berkeley_lstm6_nf, paloalto_lstm6_nf, boulder_lstm6_nf, 'LSTM 6-step', 'No')
lstm6_f = dictstodf(berkeley_lstm6_f, paloalto_lstm6_f, boulder_lstm6_f, 'LSTM 6-step', 'Yes')


lstm = pd.concat([lstm1_nf, lstm1_f, lstm6_nf, lstm6_f], ignore_index=True, sort=False)
print(lstm.shape)

lstm = lstm.sort_values(by=['city', 'station'], ignore_index=True)
lstm

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Results Visual

# COMMAND ----------

### results dictionary metrics
## replace MSE_raw with MSE_RAW, replace RMSE_raw with RMSE_RAW

BatchModels_Evals = pd.concat([arima, prophet, lstm, simpleavg, avgDayHour, Last12Avg], ignore_index=True, sort=False)
BatchModels_Evals = BatchModels_Evals.sort_values(by=['city', 'station'], ignore_index=True)
BatchModels_Evals

# COMMAND ----------

# MAGIC %md
# MAGIC ## Charts

# COMMAND ----------

# df = BatchModels_Evals[BatchModels_Evals['Model'] == 'ARIMA' ]
# df2 = BatchModels_Evals[BatchModels_Evals['Model'] == 'Prophet' ]
# df3 = BatchModels_Evals[BatchModels_Evals['Model'] == 'LSTM 1-step' ]

# selection = alt.selection_multi(fields=['Features'], bind='legend')

arima = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['ARIMA', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'ARIMA') & (datum.Features == 'No')
)


prophet = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['Prophet', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'Prophet') & (datum.Features == 'No')
)



lstm = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['LSTM', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'LSTM 1-step') & (datum.Features == 'No')
)



alt.hconcat(
    arima,
    prophet,
    lstm,
    data = BatchModels_Evals
).resolve_scale(
    y='shared'
)

# COMMAND ----------

summary_res = BatchModels_Evals[(BatchModels_Evals['Features'] == 'No') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].groupby(['city', 'Model']).agg(MeanRMSE=('RMSE_round', np.mean), MedianRMSE= ('RMSE_round',np.median)).reset_index()
summary_res.sort_values(by = ['city', 'MedianRMSE'], ascending=False).reset_index(drop=True)

# COMMAND ----------

BatchModels_Evals[(BatchModels_Evals['city'] == 'Berkeley') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].sort_values(by = ['RMSE_round'], ascending=False)

# COMMAND ----------

BatchModels_Evals[(BatchModels_Evals['city'] == 'Palo Alto') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].sort_values(by = ['station', 'RMSE_round'], ascending=False)

# COMMAND ----------

BatchModels_Evals[(BatchModels_Evals['city'] == 'Boulder') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].sort_values(by = ['station', 'RMSE_round'], ascending=False).head(53)

# COMMAND ----------

BatchModels_Evals[(BatchModels_Evals['city'] == 'Boulder') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].sort_values(by = ['station', 'RMSE_round'], ascending=False).tail(53)

# COMMAND ----------

arima = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['ARIMA', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'ARIMA') & (datum.Features == 'Yes')
)


prophet = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['Prophet', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'Prophet') & (datum.Features == 'Yes')
)



lstm = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Location', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round:Q', 
            axis=alt.Axis(title = ['RMSE'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['Model:N','city:N', 'station:N', 'MSE_Raw:Q', 'MSE_round:Q', 'RMSE_Raw:Q', 'RMSE_round:Q']
).properties(
    title=['LSTM', 
           'Rounded Predictions'],
    width=600,
    height=400).transform_filter(
    (datum.Model == 'LSTM 1-step') & (datum.Features == 'Yes')
)



alt.hconcat(
    arima,
    prophet,
    lstm,
    data=BatchModels_Evals
).resolve_scale(
    y='shared'
)

# COMMAND ----------

summary_res = BatchModels_Evals[(BatchModels_Evals['Features'] == 'Yes') & (BatchModels_Evals['Model'] != 'LSTM 6-step')].groupby(['city', 'Model']).agg(MeanRMSE=('RMSE_round', np.mean), MedianRMSE= ('RMSE_round',np.median)).reset_index()
summary_res.sort_values(by = ['city', 'MedianRMSE'], ascending=False).reset_index(drop=True)

# COMMAND ----------


lstm6nf = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Locations', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round', 
            axis=alt.Axis(title = ['RMSE Rounded'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'City'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = [ 'city:N', 'station:N', 'Features:N', 'MSE_Raw', 'MSE_round','RMSE_Raw', 'RMSE_round']
).properties(
    title=['LSTM Multistep: 6 steps out', 
           'Rounded Predictions', 'without Features'],
    width=800,
    height=400
).transform_filter(
    (datum.Model == 'LSTM 6-step') & (datum.Features == 'No')
)


lstm6f = alt.Chart(BatchModels_Evals).mark_bar(filled = True).encode(
    x=alt.X('station:N', 
            axis=alt.Axis(title='Station Locations', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('RMSE_round', 
            axis=alt.Axis(title = ['RMSE Rounded'],titleFontSize=14,labelFontSize=12, grid=False)),
    color = alt.Color('city:N', 
                      legend = alt.Legend(
                          title = 'Station Location'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = ['city:N', 'station:N','Features:N', 'MSE_Raw', 'MSE_round','RMSE_Raw', 'RMSE_round']
).properties(
    title=['LSTM Multistep: 6 steps out', 
           'Rounded Predictions', 'with Features'],
    width=800,
    height=400
).transform_filter(
    (datum.Model == 'LSTM 6-step') & (datum.Features == 'Yes')
)


alt.hconcat(
    lstm6nf,
    lstm6f,
    data=BatchModels_Evals
).resolve_scale(
    y='shared'
)


# .configure_title(
#     fontSize=20,
#     color='black'
# ).configure_legend(
#     strokeColor='gray',
#     fillColor='#F9F9F9',
#     padding=10,
#     titleFontSize=12,
#     cornerRadius=10,
#     orient='top-left')

# COMMAND ----------

summary_res = BatchModels_Evals[(BatchModels_Evals['Model'] == 'LSTM 6-step')].groupby(['city', 'Model', 'Features']).agg(MeanRMSE=('RMSE_round', np.mean), MedianRMSE= ('RMSE_round',np.median)).reset_index()
summary_res.sort_values(by = ['city', 'MedianRMSE'], ascending=False).reset_index(drop=True)

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

# ## TESTING THE PROPHET FUNCTION

# start = dt.datetime(2022, 1, 1)
# end = dt.datetime(2022, 5, 5)
# date_col = 'DateTime'
# actualcol= 'Ports Available'
# output_col = 'Ports Available'
# station = 'Slrp'

# ## filter data set
# slrp_p = arima_filter(slrp_transformed, start, end, date_col)

# slrp_prophet_train, slrp_prophet_test = split2_TrainTest(slrp_p, 0.7)

# traindf = slrp_prophet_train
# testdf = slrp_prophet_test

# #prophet_model, prophet_testdf, prophet_Evals = run_Nprophet(slrp_prophet_train, slrp_prophet_test, 'DateTime', 'Ports Available', station)


# if date_col != 'ds':
#     traindf = traindf.rename(columns={date_col: 'ds'})
#     testdf = testdf.rename(columns={date_col: 'ds'})

# if output_col != 'y':
#     traindf = traindf.rename(columns={output_col: "y"})
#     testdf = testdf.rename(columns={output_col: "y"})
        
# print(traindf.columns)
# traindf = traindf[['ds', 'y']]
# testdf = testdf[['ds', 'y']]
    
# # create model
# m = NeuralProphet()
# m.fit(traindf, freq = '10min')#, epochs = 10)
    
# # make predictions
# future = m.make_future_dataframe(traindf, periods = testdf.shape[0])#, freq = '10min')
# forecast = m.predict(future)

# print(forecast)
# print(m.test(testdf))
    
# preds = forecast[(forecast['ds'] <= testdf['ds'].max()) & (forecast['ds'] >= testdf['ds'].min())]

# print(preds)
# # # rounding predictions
# # ## need to change how we are rounding if there is more than 1 station being predicted for
# # ## this assumes that there is at least one point in time when all stations are available (probably a fair assumption)
# # preds['rounded'] = np.around(preds['yhat']).clip(upper = traindf['y'].max())
# # preds = preds[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'rounded']]

# # #create dataframe to output
# # testdf = testdf.rename(columns = {output_col: "Actuals", date_col: 'DateTime'})
# # testdf = testdf[['DateTime', 'Actuals']].merge(preds, left_on = 'DateTime', right_on = 'ds')

# # pred_col = 'yhat'

# # ## Evaluation Metrics ###
# # MSE_raw = mse(testdf['Actuals'], testdf[pred_col])
# # MSE_rounded = mse(testdf['Actuals'], testdf['rounded'])
# # RMSE_raw = math.sqrt(MSE_raw)
# # RMSE_rounded = math.sqrt(MSE_rounded)

# # Evals = dict({station: 
# #              dict({'MSE_raw': MSE_raw,
# #                   'MSE_round': MSE_rounded,
# #                   'RMSE_raw': RMSE_raw,
# #                   'RMSE_round': RMSE_rounded}) 
# #              })

# # #     Evals = dict({'MSE_raw': MSE_raw,
# # #                       'MSE_round': MSE_rounded,
# # #                       'RMSE_raw': RMSE_raw,
# # #                       'RMSE_round': RMSE_rounded})

# # print(Evals)
# # print(m.test(testdf))

    



# COMMAND ----------

# MAGIC %md
# MAGIC #Tables for Tableau

# COMMAND ----------

BatchModels_Evals #[BatchModels_Evals['city'] == 'Boulder'][['city', 'RMSE_round', 'Model', 'Features']].groupby(['city', 'Model', 'Features']).mean().reset_index().sort_values('RMSE_round')

# COMMAND ----------

## general RMSEs
# BatchModels_Evals['PDQ_ARIMAOrder'] = BatchModels_Evals['PDQ_ARIMAOrder'].astype(str)
# sparkBatchModel_Evals = spark.createDataFrame(BatchModels_Evals)


# sparkBatchModel_Evals.write.mode("overwrite").saveAsTable('BatchModel_Evals')

# COMMAND ----------

def readsparktodf(pth, modelname, features = 'No'):
    df = spark.read.parquet(f"/mnt/{mount_name}/data/batch/"+str(pth)).toPandas()
    df.loc[:,'Model'] = str(modelname)
    df.loc[:,'Features'] = str(features)
    return df

# COMMAND ----------

####### Read in Results ##########3
Berkeley_LSTM_1StepFeatures = readsparktodf('Berkeley_LSTM_1StepFeatures', 'LSTM 1-step', 'Yes')
Berkeley_LSTM_1StepNoFeatures = readsparktodf('Berkeley_LSTM_1StepNoFeatures', 'LSTM 1-step', 'No')
Berkeley_LSTM_6StepFeatures = readsparktodf('Berkeley_LSTM_6StepFeatures', 'LSTM 6-step', 'Yes')
Berkeley_LSTM_6StepNoFeatures = readsparktodf('Berkeley_LSTM_6StepNoFeatures', 'LSTM 6-step', 'No')
BerkeleySlrp_ARIMA_Features	= readsparktodf('BerkeleySlrp_ARIMA_Features', 'ARIMA', 'Yes')
BerkeleySlrp_ARIMA = readsparktodf('BerkeleySlrp_ARIMA', 'ARIMA', 'No')
BerkeleySlrp_Prophet_Features = readsparktodf('BerkeleySlrp_Prophet_Features', 'Prophet', 'Yes')
BerkeleySlrp_Prophet = readsparktodf('BerkeleySlrp_Prophet', 'Prophet', 'No')


Boulder_ARIMA_Features = readsparktodf('Boulder_ARIMA_Features', 'ARIMA', 'Yes')
Boulder_ARIMA = readsparktodf('Boulder_ARIMA', 'ARIMA', 'No')
Boulder_LSTM_1StepFeatures = readsparktodf('Boulder_LSTM_1StepFeatures', 'LSTM 1-step', 'Yes')
Boulder_LSTM_1StepNoFeatures = readsparktodf('Boulder_LSTM_1StepNoFeatures', 'LSTM 1-step', 'No')	
Boulder_LSTM_6StepFeatures = readsparktodf('Boulder_LSTM_6StepFeatures', 'LSTM 6-step', 'Yes')
Boulder_LSTM_6StepNoFeatures = readsparktodf('Boulder_LSTM_6StepNoFeatures', 'LSTM 6-step', 'No')
Boulder_Prophet_Features = readsparktodf('Boulder_Prophet_Features', 'Prophet', 'Yes')
Boulder_Prophet = readsparktodf('Boulder_Prophet', 'Prophet', 'No')


PaloAlto_ARIMA_features = readsparktodf('PaloAlto_ARIMA_features', 'ARIMA', 'Yes')
PaloAlto_ARIMA = readsparktodf('PaloAlto_ARIMA', 'ARIMA', 'No')
PaloAlto_LSTM_1StepFeatures = readsparktodf('PaloAlto_LSTM_1StepFeatures', 'LSTM 1-step', 'Yes')
PaloAlto_LSTM_1StepNoFeatures = readsparktodf('PaloAlto_LSTM_1StepNoFeatures', 'LSTM 1-step', 'No')
PaloAlto_LSTM_6StepFeatures = readsparktodf('PaloAlto_LSTM_6StepFeatures', 'LSTM 6-step', 'Yes')
PaloAlto_LSTM_6StepNoFeatures = readsparktodf('PaloAlto_LSTM_6StepNoFeatures', 'LSTM 6-step', 'No')
PaloAlto_Prophet_Features = readsparktodf('PaloAlto_Prophet_Features', 'Prophet', 'Yes')
PaloAlto_Prophet = readsparktodf('PaloAlto_Prophet', 'Prophet', 'No')


# COMMAND ----------

######### berkeley cleanup ########
## add site name for slrp
BerkeleySlrp_ARIMA_Features.loc[:, 'SiteName'] = 'slrp'
BerkeleySlrp_ARIMA.loc[:, 'SiteName'] = 'slrp'
BerkeleySlrp_Prophet_Features.loc[:, 'SiteName'] = 'slrp'
BerkeleySlrp_Prophet.loc[:, 'SiteName'] = 'slrp'

# add location
BerkeleySlrp_ARIMA_Features.loc[:, 'Location'] = 'Berkeley'
BerkeleySlrp_ARIMA.loc[:, 'Location'] = 'Berkeley'
BerkeleySlrp_Prophet_Features.loc[:, 'Location'] = 'Berkeley'
BerkeleySlrp_Prophet.loc[:, 'Location'] = 'Berkeley'


# drop unneeded colums
BerkeleySlrp_ARIMA_Features = BerkeleySlrp_ARIMA_Features.drop([
    'IsWeekend','Hour','holiday','note_Break','note_Holiday','note_None','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)
BerkeleySlrp_ARIMA = BerkeleySlrp_ARIMA.drop([
    'IsWeekend','Hour','holiday','note_Break','note_Holiday','note_None','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)


BerkeleySlrp_Prophet_Features = BerkeleySlrp_Prophet_Features.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)
BerkeleySlrp_Prophet = BerkeleySlrp_Prophet.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)

BerkeleySlrp_ARIMA_Features = BerkeleySlrp_ARIMA_Features.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})
BerkeleySlrp_ARIMA = BerkeleySlrp_ARIMA.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})

BerkeleySlrp_Prophet_Features = BerkeleySlrp_Prophet_Features.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})
BerkeleySlrp_Prophet = BerkeleySlrp_Prophet.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})


berkeley= pd.concat([ BerkeleySlrp_ARIMA_Features, BerkeleySlrp_ARIMA,
    Berkeley_LSTM_1StepFeatures, Berkeley_LSTM_1StepNoFeatures,
    Berkeley_LSTM_6StepFeatures, Berkeley_LSTM_6StepNoFeatures,
    BerkeleySlrp_Prophet_Features, BerkeleySlrp_Prophet
], sort=False)

berkeley

# COMMAND ----------

######### Palo Alto ########
# drop unneeded colums
PaloAlto_ARIMA_features = PaloAlto_ARIMA_features.drop([
    'IsWeekend','Hour','holiday','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)
# PaloAlto_ARIMA = PaloAlto_ARIMA.drop([
#     'IsWeekend','Hour','holiday','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)


PaloAlto_Prophet_Features = PaloAlto_Prophet_Features.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)
PaloAlto_Prophet = PaloAlto_Prophet.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)

PaloAlto_ARIMA_features = PaloAlto_ARIMA_features.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})
PaloAlto_ARIMA = PaloAlto_ARIMA.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})

PaloAlto_Prophet_Features = PaloAlto_Prophet_Features.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})
PaloAlto_Prophet = PaloAlto_Prophet.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})


paloalto= pd.concat([ PaloAlto_ARIMA_features,
                     PaloAlto_ARIMA,
                     PaloAlto_LSTM_1StepFeatures,
                     PaloAlto_LSTM_1StepNoFeatures,
                     PaloAlto_LSTM_6StepFeatures,
                     PaloAlto_LSTM_6StepNoFeatures,
                     PaloAlto_Prophet_Features,
                     PaloAlto_Prophet
], sort=False)

paloalto

# COMMAND ----------

######### Boulder ########
# drop unneeded colums
Boulder_ARIMA_Features = Boulder_ARIMA_Features.drop([
    'IsWeekend','Hour','holiday','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)
# Boulder_ARIMA = Boulder_ARIMA.drop([
#     'IsWeekend','Hour','holiday','month_cosine','month_sine','hour_cosine','hour_sine','dayofweek_cosine','dayofweek_sine'], axis = 1)


Boulder_Prophet_Features = Boulder_Prophet_Features.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)
Boulder_Prophet = Boulder_Prophet.drop(['yhat_lower', 'yhat_upper', 'ds'], axis = 1)

Boulder_ARIMA_features = Boulder_ARIMA_Features.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})
Boulder_ARIMA = Boulder_ARIMA.rename(columns={'predictions':'Predictions', 'predictions_(rounded)':'Predictions_(rounded)'})

Boulder_Prophet_Features = Boulder_Prophet_Features.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})
Boulder_Prophet = Boulder_Prophet.rename(columns={'yhat': 'Predictions', 'rounded':'Predictions_(rounded)'})


boulder= pd.concat([ Boulder_ARIMA_features,
                     Boulder_ARIMA,
                     Boulder_LSTM_1StepFeatures,
                     Boulder_LSTM_1StepNoFeatures,
                     Boulder_LSTM_6StepFeatures,
                     Boulder_LSTM_6StepNoFeatures,
                     Boulder_Prophet_Features,
                     Boulder_Prophet
], sort=False)

boulder

# COMMAND ----------

# combine all
batchmodel_results = pd.concat([
    berkeley,
    paloalto,
    boulder
], ignore_index=True, sort=False)

batchmodel_results = batchmodel_results.rename(columns={'Predictions_(rounded)': 'Predictions_Rounded'})
batchmodel_results = batchmodel_results.sort_values(['DateTime', 'Location', 'SiteName', 'Model', 'Features'])

# COMMAND ----------

## append benchmark models
Berkeley_simpleavg = spark.read.parquet(f"/mnt/{mount_name}/data/batch/BerkeleySlrp_BenchmarkSimpleAvg").toPandas()
PaloAlto_simpleavg = spark.read.parquet(f"/mnt/{mount_name}/data/batch/PaloAlto_BenchmarkSimpleAvg").toPandas()
Boulder_simpleavg = spark.read.parquet(f"/mnt/{mount_name}/data/batch/Boulder_BenchmarkSimpleAvg").toPandas()

PaloAlto_simpleavg = PaloAlto_simpleavg.rename(columns={"datetime_pac": "DateTime"})
Boulder_simpleavg = Boulder_simpleavg.rename(columns={'Date_Time': "DateTime"})


Berkeley_avgDayHour = spark.read.parquet(f"/mnt/{mount_name}/data/batch/BerkeleySlrp_BenchmarkDayHourAvg").toPandas()
PaloAlto_avgDayHour = spark.read.parquet(f"/mnt/{mount_name}/data/batch/PaloAlto_BenchmarkDayHourAvg").toPandas()
Boulder_avgDayHour = spark.read.parquet(f"/mnt/{mount_name}/data/batch/Boulder_BenchmarkDayHourAvg").toPandas()


Berkeley_avglast12 = spark.read.parquet(f"/mnt/{mount_name}/data/batch/BerkeleySlrp_BenchmarkAvgLast12").toPandas()
PaloAlto_avglast12 = spark.read.parquet(f"/mnt/{mount_name}/data/batch/PaloAlto_BenchmarkAvgLast12").toPandas()
Boulder_avglast12 = spark.read.parquet(f"/mnt/{mount_name}/data/batch/Boulder_BenchmarkAvgLast12").toPandas()


batchmodel_results = pd.concat([
    batchmodel_results,
    Berkeley_simpleavg,
    PaloAlto_simpleavg,
    Boulder_simpleavg,
    Berkeley_avgDayHour,
    PaloAlto_avgDayHour,
    Boulder_avgDayHour,
    Berkeley_avglast12,
    PaloAlto_avglast12,
    Boulder_avglast12
], ignore_index=True, sort=False)

# COMMAND ----------

test = batchmodel_results[batchmodel_results['Model'] == 'LSTM 6-step'][['DateTime', 'Actuals', 'Predictions_Rounded', 'Model', 'Features', 'Location', 'StepsOut']]
test.loc[:, 'Error'] = test['Predictions_Rounded'] - test['Actuals']
test.loc[:, 'Error'] = test['Error'].abs()
MAE_timestep = test[['Model',	'Features',	'Location',	'StepsOut',	'Error']].groupby(['Model', 'Features', 'Location', 'StepsOut']).mean()
MAE_timestep = MAE_timestep.reset_index()
MAE_timestep

# COMMAND ----------

Berk_MAE = alt.Chart(MAE_timestep).mark_bar(filled = True).encode(
    x=alt.X('StepsOut:O', 
            axis=alt.Axis(title='LSTM Steps out Predicted', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('Error', 
            axis=alt.Axis(title = ['MAE'],titleFontSize=14,labelFontSize=12, grid=False)),
    column = 'Features:N',
    color = alt.Color('Features:N', 
                      legend = alt.Legend(
                          title = 'Features'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = [ 'Model:N', 'Location:N', 'Features:N','StepsOut', 'Error', ]
).properties(
    title=['Berkeley: SlrpEV', 
           'Mean Absolute Error for LSTM per Step out'],
    width=200,
    height=300
).transform_filter(
    (datum.Location == 'Berkeley')
)


PA_MAE = alt.Chart(MAE_timestep).mark_bar(filled = True).encode(
    x=alt.X('StepsOut:O', 
            axis=alt.Axis(title='LSTM Steps out Predicted', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('Error', 
            axis=alt.Axis(title = ['MAE'],titleFontSize=14,labelFontSize=12, grid=False)),
    column = 'Features:N',
    color = alt.Color('Features:N', 
                      legend = alt.Legend(
                          title = 'Features'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = [ 'Model:N', 'Location:N', 'Features:N','StepsOut', 'Error', ]
).properties(
    title=['Palo Alto', 
           'Mean Absolute Error for LSTM per Step out'],
    width=200,
    height=300
).transform_filter(
    (datum.Location == 'PaloAlto')
)

Bo_MAE = alt.Chart(MAE_timestep).mark_bar(filled = True).encode(
    x=alt.X('StepsOut:O', 
            axis=alt.Axis(title='LSTM Steps out Predicted', titleFontSize=14, labelFontSize=12, grid=False)),
    y=alt.Y('Error', 
            axis=alt.Axis(title = ['MAE'],titleFontSize=14,labelFontSize=12, grid=False)),
    column = 'Features:N',
    color = alt.Color('Features:N', 
                      legend = alt.Legend(
                          title = 'Features'), 
                      scale=alt.Scale(scheme='tableau10')),
    tooltip = [ 'Model:N', 'Location:N', 'Features:N','StepsOut', 'Error', ]
).properties(
    title=['Boulder', 
           'Mean Absolute Error for LSTM per Step out'],
    width=200,
    height=300
).transform_filter(
    (datum.Location == 'Boulder')
)


alt.hconcat(
    Berk_MAE,
    PA_MAE,
    Bo_MAE,
    data=MAE_timestep
).resolve_scale(
    y='shared'
).configure_title(anchor='middle')


# COMMAND ----------

batchmodel_results.columns

# COMMAND ----------

batchmodel_results = batchmodel_results.melt(id_vars=['DateTime', 'Model', 'Features', 'SiteName', 'Location', 'StepsOut'], var_name='Category', value_name='Ports')
batchmodel_results

# COMMAND ----------

batchmodel_results = batchmodel_results.replace('slrp', 'Slrp')
batchmodel_results

# COMMAND ----------

# write to table
# ## general RMSEs
# BatchModels_Evals['PDQ_ARIMAOrder'] = BatchModels_Evals['PDQ_ARIMAOrder'].astype(str)

sparkBatchModel_Res = spark.createDataFrame(batchmodel_results)
sparkBatchModel_Res.write.mode("overwrite").saveAsTable('BatchModel_Results')