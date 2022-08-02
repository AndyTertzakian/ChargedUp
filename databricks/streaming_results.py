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
import holidays
import boto3
import pyspark.sql.functions as F
import pickle

import warnings
warnings.filterwarnings("ignore")

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

slrp_stream_results = spark.read.parquet(f"/mnt/{mount_name}/data/streaming_predictions/slrp_pred_errors")
display(slrp_stream_results)

# COMMAND ----------

(slrp_stream_results.repartition(1)
    .write
    .format("csv")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/streaming_predictions/slrp_pred_errors_csv"))

# COMMAND ----------

palo_alto_seed = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_seed")
palo_alto_seed = palo_alto_seed.toPandas()
print(palo_alto_seed['station'].unique())

# COMMAND ----------

boulder_seed = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_seed")
boulder_seed = boulder_seed.toPandas()
print(boulder_seed['station'].unique())

# COMMAND ----------

print(palo_alto_seed['station'].unique()[3:])

# COMMAND ----------

# palo_alto_seed = spark.read.parquet(f"/mnt/{mount_name}/data/palo_alto_seed")
# palo_alto_seed = palo_alto_seed.toPandas()
# palo_alto_stream_results = {}

for station in palo_alto_seed['station'].unique()[3:]:
    palo_alto_stream_station_results = spark.read.parquet(f"/mnt/{mount_name}/data/streaming_predictions/" + station.lower() + "_pred_errors")
    print(f"Loaded {station} predictions")
#     palo_alto_stream_results[station] = palo_alto_stream_station_results
    (palo_alto_stream_station_results.repartition(1)
    .write
    .format("csv")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/streaming_predictions/" + station.lower() + "_pred_errors_csv"))

# COMMAND ----------

for station in boulder_seed['station'].unique():
    boulder_stream_station_results = spark.read.parquet(f"/mnt/{mount_name}/data/streaming_predictions/" + station.lower() + "_pred_errors")
    print(f"Loaded {station} predictions")
    (boulder_stream_station_results.repartition(1)
    .write
    .format("csv")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/streaming_predictions/" + station.lower() + "_pred_errors_csv"))

# COMMAND ----------

