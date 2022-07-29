# Databricks notebook source
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

display(dbutils.fs.ls(f"/mnt/{mount_name}/"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Steps
# MAGIC 
# MAGIC 1. install kafka (cell A1)
# MAGIC 2. start zookeeper (cell B1)
# MAGIC 3. start kafka broker (cell C1 -> C3)
# MAGIC 4. create topics (cell A2)
# MAGIC 5. start all 3 consumers   
# MAGIC     - Berkeley (cell E1 -> E2)
# MAGIC     - Palo Alto (cell F1 -> F2)
# MAGIC     - Boulder (cell G1 -> G2)
# MAGIC 6. start producer (cell D1 -> D7)
# MAGIC 7. Watch the streaming models!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Kafka

# COMMAND ----------

# DBTITLE 1,A1
# MAGIC %sh
# MAGIC wget https://archive.apache.org/dist/kafka/2.6.2/kafka_2.12-2.6.2.tgz
# MAGIC tar -xzf kafka_2.12-2.6.2.tgz
# MAGIC ls

# COMMAND ----------

# DBTITLE 1,A2
# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC bin/kafka-topics.sh --create --topic berkeley --bootstrap-server localhost:9092 --partitions 1 
# MAGIC bin/kafka-topics.sh --create --topic palo-alto --bootstrap-server localhost:9092 --partitions 1
# MAGIC bin/kafka-topics.sh --create --topic boulder --bootstrap-server localhost:9092 --partitions 1
# MAGIC bin/kafka-topics.sh --create --topic test --bootstrap-server localhost:9092 --partitions 1

# COMMAND ----------

# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC bin/kafka-topics.sh --list --zookeeper localhost:2181

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configurations
# MAGIC #### Java

# COMMAND ----------

# MAGIC %sh
# MAGIC echo """KafkaClient {
# MAGIC    org.apache.kafka.common.security.scram.ScramLoginModule required
# MAGIC    username="w210"
# MAGIC    password="Sunshine123";
# MAGIC    };""" >> users_jaas.conf

# COMMAND ----------

# MAGIC %sh
# MAGIC export KAFKA_OPTS=-Djava.security.auth.login.config=/databricks/driver/users_jaas.conf

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir tmp
# MAGIC touch tmp/kafka.client.truststore.jks

# COMMAND ----------

# MAGIC %sh ls /databricks/driver/tmp/

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo chmod +777 /usr/lib/jvm/zulu8-ca-amd64/jre/lib/security/cacerts
# MAGIC cp -v /usr/lib/jvm/zulu8-ca-amd64/jre/lib/security/cacerts /databricks/driver/tmp/kafka.client.truststore.jks 

# COMMAND ----------

# MAGIC %sh 
# MAGIC echo """security.protocol=SASL_SSL
# MAGIC sasl.mechanism=SCRAM-SHA-512
# MAGIC ssl.truststore.location=/databricks/driver/tmp/kafka.client.truststore.jks""" >> kafka_2.12-2.6.2/bin/client_sasl.properties

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls kafka_2.12-2.6.2/bin/

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Reference Data

# COMMAND ----------

# def make_predictions_old(model_name, location_df, streamed_df, location_df_6m, results_df):
#     """
#     model_name: name of model you want to load from S3
#     location_df: training data
#     location_df_6m: all data (training + streaming)
    
#     Make predictions from an LSTM model by getting the station information from the last `n_inputs` timestamps
#     and running it through the model for predictions
    
#     Use the location_df_6m dataframe to calculate the errors (MSE, RMSE) of the predictions
    
#     This function assumes that we are only predicting using a single feature (no time-based features)
#     """
    
#     #     model = get_model_from_s3(model_name)
    
#     # For testing, concatenate the original dataframe (seed) with the sreamed data
#     full_historical = pd.concat([location_df, streamed_df]).sort_values(by = 'datetime')
    
#     # Setup for saving figures
#     bucket='w210v2'
#     client = boto3.client("s3",
#         aws_access_key_id="AKIA3CS2H32VF7XY33S2",
#         aws_secret_access_key="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",
#     )
    
    
#     # Loop through each unique station
#     for station in location_df['station'].unique():
        
#         # Filter to that station's values
#         station_historical = full_historical[full_historical['station'] == station]
        
#         # Develop a testing point to predict out n_output time steps
#         test = np.array(station_historical.tail(n_inputs)["ports_available"])
#         test.resize(1, n_inputs, 1)
        
#         # Make predictions from the model
#         predictions = model.predict(test)[0]
#         predictions_rounded = np.round(predictions)

#         # Get the actual values from the 6 month data to calculate the MSE
#         actuals = location_df_6m[(location_df_6m['datetime'] > station_historical['datetime'].max()) &
#                                  (location_df_6m['datetime'] <= station_historical['datetime'].max() + 
#                                   dt.timedelta(minutes = 10*n_outputs))]
        
#         # Add the predictions to this data frame
#         actuals['predicted'] = predictions
#         actuals['predicted_rounded'] = predictions_rounded
        
#         # For sanity checking, print the maximum timestamp from the historical data and the minimum value from the predictions
#         print(full_historical['datetime'].max())
#         print(actuals['datetime'].min())
#         actuals = actuals.reset_index(drop = True)

#         ## Evaluation Metrics ###
#         MSE_raw = mse(actuals['ports_available'], actuals['predicted'])
#         MSE_rounded = mse(actuals['ports_available'], actuals['predicted_rounded'])
#         RMSE_raw = math.sqrt(MSE_raw)
#         RMSE_rounded = math.sqrt(MSE_rounded)
        
#         # Calcualte RMSEs for each hour
#         RMSE_1 = np.sqrt(mse(actuals.iloc[0:6, :]['ports_available'], actuals.iloc[0:6,:]['predicted_rounded']))
#         RMSE_2 = np.sqrt(mse(actuals.iloc[6:12, :]['ports_available'], actuals.iloc[6:12,:]['predicted_rounded']))
#         RMSE_3 = np.sqrt(mse(actuals.iloc[12:18, :]['ports_available'], actuals.iloc[12:18,:]['predicted_rounded']))
#         RMSE_4 = np.sqrt(mse(actuals.iloc[18:24, :]['ports_available'], actuals.iloc[18:24,:]['predicted_rounded']))
#         RMSE_5 = np.sqrt(mse(actuals.iloc[24:30, :]['ports_available'], actuals.iloc[24:30,:]['predicted_rounded']))
#         RMSE_6 = np.sqrt(mse(actuals.iloc[30:36, :]['ports_available'], actuals.iloc[30:36,:]['predicted_rounded']))
        
#         new_results = pd.DataFrame(data = {'Overall RMSE': [RMSE_rounded],
#                                            'RMSE for Hour 1': [RMSE_1],
#                                            'RMSE for Hour 2': [RMSE_2],
#                                            'RMSE for Hour 3': [RMSE_3],
#                                            'RMSE for Hour 4': [RMSE_4],
#                                            'RMSE for Hour 5': [RMSE_5],
#                                            'RMSE for Hour 6': [RMSE_6]},
#                                    index = [actuals['datetime'].min()])

#         print(RMSE_raw)
#         print(RMSE_rounded)
        
#         return pd.concat([results, new_results])

# COMMAND ----------

# dictionary of timestamps of the end of each training set for each location


# update dictionary with the next training end timestamp (user-defined period)
# if the next timestamp from the streaming data is equal to the dictionary, update the dictionary end-datetime again
    # update the LSTM models
# else: make another prediction

# need to store predictions
# need to associate the predictions with actuals
# store datetime + predictions --> map these back with the actuals

# as we're reading in streaming data, update S3 file

# COMMAND ----------

# from kafka import KafkaConsumer

# ## Set the topic for this
# topic = "test"

# ## Delete streaming data for 
# try:
#     dbutils.fs.rm(f"/mnt/{mount_name}/streaming/{topic}", True)
# except Exception as e:
#     print(e)

# consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')

# last_timestamp = None

# received_data = pd.DataFrame(data = {'datetime': [],
#                                      'station': [],
#                                      'ports_available': []})
# for message in consumer:
#     print(message)