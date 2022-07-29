# Databricks notebook source
# DBTITLE 1,D1
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

# COMMAND ----------

# DBTITLE 1,D2
topic_name = "berkeley"
slrp_df = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_stream")
slrp_df = slrp_df.withColumn("topic", F.lit(topic_name))
slrp_df = slrp_df.toPandas()
slrp_df["datetime_adj"] = slrp_df["datetime"]
slrp_df

# COMMAND ----------

(5) / len(set(boulder_df["datetime"].dt.date)) * 100

# COMMAND ----------

# DBTITLE 1,D3
topic_name = "palo-alto"
palo_alto_df = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_stream")
palo_alto_df = palo_alto_df.withColumn("topic", F.lit(topic_name))
palo_alto_df = palo_alto_df.toPandas()
palo_alto_df["datetime_adj"] = palo_alto_df["datetime"] + (slrp_df["datetime"].min() - palo_alto_df["datetime"].min())
palo_alto_df

# COMMAND ----------

# DBTITLE 1,D4
topic_name = "boulder"
boulder_df = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_stream")
boulder_df = boulder_df.withColumn("topic", F.lit(topic_name))
boulder_df = boulder_df.toPandas()
boulder_df["datetime_adj"] = boulder_df["datetime"]
boulder_df

# COMMAND ----------

# DBTITLE 1,D5
## Append and sort the dataframes for each location together
stream_df = slrp_df.append(palo_alto_df)
stream_df = stream_df.append(boulder_df)
stream_df = stream_df.sort_values(["datetime_adj", "topic"])
stream_df = stream_df.drop(["datetime_adj"], axis=1)
stream_df = stream_df.reset_index(drop=True)
stream_df.head(30)

# COMMAND ----------

# DBTITLE 1,D6
stream_df["key"] = stream_df['station'] + "," + stream_df['datetime'].astype(str)
stream_df["value"] = stream_df['ports_available'].astype(str)
stream_df

# COMMAND ----------

(350254 * 0.0226) / 60 / 24

# COMMAND ----------

## Loop through all filesystem objects within the data directory


## !!CAREFUL!! 
## ONLY RUN THIS IF YOU ARE RERUNNING ALL OF THE STREAMING.
##             CONSIDER MOVING THE FILES INSTEAD OF DELETING THEM.
if False:
    count = 0
    data_dir_entries = dbutils.fs.ls(f"/mnt/{mount_name}/data/streaming_predictions/")
    for data_dir_entry in data_dir_entries:
        if "_pred" in data_dir_entry.name:
            print(data_dir_entry.name)
            try:
                dbutils.fs.rm(data_dir_entry.path, True)
                pass
            except:
                print(f"Couldn't delete: {data_dir_entry.path}")


# COMMAND ----------

# DBTITLE 1,D7
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# producer_df = stream_df[(stream_df['station'] == 'Slrp') & (stream_df['datetime'].dt.month == 2)]
producer_df = stream_df

for index, row in producer_df.iterrows():
    topic, key, value = row['topic'], row['key'], row['value']
    print(topic, key, value)
    producer.send(topic=topic,
                  key=str.encode(key),
                  value=str.encode(value))
    
#     if (row['datetime'].day == 15) & (row['datetime'].hour == 0) and (row['datetime'].minute == 0):
#         time.sleep(0)
#     elif row['datetime'].minute == 0:
#         time.sleep(0)
#     else:
#         time.sleep(0)