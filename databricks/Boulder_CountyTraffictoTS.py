# Databricks notebook source
# MAGIC %md
# MAGIC # Boulder County Continous Traffic Data
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   [Data Set](https://dtdapps.coloradodot.info/otis/TrafficData#ui/0/0/0/criteria//5/true/true/) 

# COMMAND ----------

# DBTITLE 1,Import Libraries and Set up S3
# import libraries
import pandas as pd
import datetime as dt
import time
from datetime import timedelta
import itertools

# # import libraries
# import pandas as pd
# # import altair as alt
# import datetime as dt
# import numpy as np
# #import json
# import time
# #import urllib
# #import requests
# import seaborn as sns
# #from vega_datasets import data
# #from pandas.io.json import json_normalize
# from datetime import timedelta
# #import plotly.graph_objects as go
# import pytz
# import warnings
# import matplotlib.pyplot as plt
# import math
# from keras.models import Model, Sequential
# import folium
# import holidays




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
    

display(dbutils.fs.ls(f"/mnt/{mount_name}/data/"))


# COMMAND ----------

traffic = spark.read.option("header", True).csv(f"/mnt/{mount_name}/data/Boulder2ContinuousTraffic.csv")

# #convert to pandas dataframe
traffic = traffic.toPandas()
traffic

# COMMAND ----------

traffic.columns

# COMMAND ----------

traffic.dtypes

# COMMAND ----------

# DBTITLE 1,Filtering and Clean Data
# filter data to desired fields
clean_traffic = traffic[traffic['City'] == "Longmont"].drop(['Route', 'Start', 'End', 'StationID', 'Description', 'Dir'], axis=1)

# sum primary and secondary counts
clean_traffic = clean_traffic.groupby(['County', 'City', 'Count Date']).sum().reset_index()


# rename columns
tcols = clean_traffic.columns
hours = tcols[3:]
newcols = list(itertools.chain(tcols[0:3], [int(h.replace('h', '')) for h in hours]))
res = {tcols[i]: newcols[i] for i in range(len(tcols))}
clean_traffic.rename(columns=res, inplace= True)


# get hour as col and counts as col variable
clean_traffic = clean_traffic.melt(id_vars=['County', 'City', 'Count Date'], 
                                   var_name='Hour', value_name='TrafficCount')
clean_traffic.head()

# COMMAND ----------

# DBTITLE 1,Fix Dtypes
clean_traffic['Count Date'] = pd.to_datetime(clean_traffic['Count Date'], format = '%m/%d/%Y')
clean_traffic['Hour'] = pd.to_numeric(clean_traffic['Hour']) 
clean_traffic['TrafficCount'] = pd.to_numeric(clean_traffic['TrafficCount']) 

clean_traffic.dtypes

# COMMAND ----------

# DBTITLE 1,Convert To Timeseries
timeseries = clean_traffic

# get datetime column
timeseries['DateTime'] = timeseries[['Count Date', 'Hour']].apply(
    lambda x: dt.datetime(
        year=x[0].year,
        month=x[0].month,
        day=x[0].day,
        hour=x[1],
        minute=0,
        second = 0
    ),
    axis = 1
)


timeseries = timeseries.drop(['Count Date', 'Hour'], axis = 1)

timeseries['DateTime'] = timeseries['DateTime'].apply(
    lambda x: list(pd.date_range(x, periods = 6, freq='10T'))
)


timeseries = timeseries.explode('DateTime')

timeseries = timeseries[['County', 'City', 'DateTime', 'TrafficCount']].reset_index()
timeseries

# COMMAND ----------

# DBTITLE 1,Write to s3
# create spark data frame
traffic_ts = spark.createDataFrame(timeseries) 


display(traffic_ts)

## Write to AWS S3
(traffic_ts
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/Boulder2ContinuousTraffic_TS"))