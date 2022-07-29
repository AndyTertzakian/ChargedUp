# Databricks notebook source
# MAGIC %md
# MAGIC CalTech Historical Data
# MAGIC *EDA, cleanup, etc.*
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   [Data Set](https://google.com) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Data

# COMMAND ----------

# import libraries
import pandas as pd
import altair as alt
import datetime as dt
import numpy as np
import json
import time
import urllib
import requests
import boto3
import seaborn as sns
from vega_datasets import data
from pandas.io.json import json_normalize
from datetime import timedelta
import plotly.graph_objects as go
import pytz
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
import folium

## Spark imports
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

## Spark model imports 
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors 
from pyspark.ml.feature import VectorAssembler, Imputer

## Databricks imports
import databricks.koalas as ks

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

raw_caltech_df = spark.read.option("multiline","true").json(f"/mnt/{mount_name}/data/Caltech_stations_cleaned.json")

cols_to_drop = ["_id", "userID", "userInputs", "clusterID", "siteID"]
for col_to_drop in cols_to_drop:
    raw_caltech_df = raw_caltech_df.drop(col_to_drop)

## Clean datetime columns
dt_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
for dt_col in dt_cols:
    raw_caltech_df = raw_caltech_df.withColumn(dt_col, F.regexp_replace(F.col("connectionTime"), "[A-Za-z]{3}[,]", ""))
    raw_caltech_df = raw_caltech_df.withColumn(dt_col, F.regexp_replace(F.col("connectionTime"), "GMT", ""))
    raw_caltech_df = raw_caltech_df.withColumn(dt_col, F.to_timestamp(F.col("connectionTime"), " dd MMM yyyy HH:mm:ss "))
    raw_caltech_df = raw_caltech_df.withColumn(dt_col, F.from_utc_timestamp(F.col("connectionTime"), F.col("timezone")))
    
raw_caltech_kdf = ks.DataFrame(raw_caltech_df)
raw_caltech_kdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## General EDA - Data understanding

# COMMAND ----------

raw_caltech_kdf.info()

# COMMAND ----------

raw_caltech_kdf.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Static Station Data

# COMMAND ----------

raw_caltech_kdf['stationID'].unique()

# COMMAND ----------

raw_caltech_kdf['spaceID'].unique()

# COMMAND ----------

# City-Operated Charging Stations
# https://bouldercolorado.gov/services/electric-vehicle-charging-stations

header = ['Site Name',
			'Address',
			'Plugs',
			'Fee',
			'Notes']
stations = [
            ['Boulder Airport', '3335 Airport Rd','2','Free','Level 2'],
            ['Alpine-Balsam Parking Garage', '2667 Broadway','2','Free','Level 2'],
            ['OSMP Annex', '7315 Red Deer Dr','2','Free','Level 2'],
            ['Atrium','1770 13th St','2','Free','Level 2'],
            ['Chautauqua','600 Baseline Rd','2','Free','Level 2'],
            ['Scott Carpenter Park','1505 30th St','4','Free','Level 2'],
            ['East Boulder Community Center','5660 Sioux Dr','2','Free','Level 2'],
            ['North Boulder Recreation Center','3172 Broadway','2','Free','Level 2'],
            ['South Boulder Recreation Center','1360 Gillaspie Dr','2','Free','Level 2'],
            ['Boulder Reservoir','5565 51st St','2','Free','Level 2'],
            ['Valmont Dog Park','5333 Valmont Rd','4','Free','Level 2'],
            ['10th & Walnut Parking Garage','900 Walnut St','4',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['11th & Spruce Parking Garage','1104 Spruce','2',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['11th & Walnut Parking Garage','1100 Walnut St','2',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['14th & Walnut Parking Garage','1400 Walnut Parking Garage','2',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['15th & Pearl Parking Garage','1500 Pearl St','4',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['2240 Broadway Parking Garage','2240 Broadway','2',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2'],
            ['Boulder Junction Parking Garage','2052 Junction Pl','2',
             '$1 for the first two hours; $2.50 per hour for each additional hour*',
             'Level 2']
            ]
asterisk = '*Plus city garage parking fees'

site_names = [station_info[0] for station_info in stations]
addresses = [station_info[1] for station_info in stations]
plug_nums = [station_info[2] for station_info in stations]
fees = [station_info[3] for station_info in stations]
notes = [station_info[4] for station_info in stations]

# COMMAND ----------

static_station_data = pd.DataFrame(data = {header[0]: site_names,
                                           header[1]: addresses,
                                           header[2]: plug_nums,
                                           header[3]: fees,
                                           header[4]: notes})
static_station_data['Address'].unique()

# COMMAND ----------

for address in boulder['Address'].unique():
    if address in static_station_data['Address'].unique():
        if len(address) < 16:
            print(address + '\t\tYes')
        else:
            print(address + '\tYes')
    else:
        if len(address) < 16:
            print(address + '\t\tNo')
        else:
            print(address + '\tNo')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Stations

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
limit = 1000
ax[0].hist(boulder[boulder['Total Duration (minutes)'] < limit]['Total Duration (minutes)'], bins = 100);
ax[1].hist(boulder[boulder['Charging Time (minutes)'] < limit]['Charging Time (minutes)'], bins = 100);

# COMMAND ----------

sorted(boulder['Station_Name'].unique())

# COMMAND ----------

# Fill in missing data
boulder['End_Date___Time'] = boulder['End_Date___Time'].fillna(boulder['Start_Date___Time'] +\
                             dt.timedelta(minutes = boulder['Total Duration (minutes)'].mean()))

# COMMAND ----------

boulder_orig = boulder.copy()
boulder_current = boulder[boulder['Charging Time (minutes)'] < 1440].reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Visualizations

# COMMAND ----------

palo_alto['Total Duration (minutes)'].hist(bins = 20)

# COMMAND ----------

duration_Stations = palo_alto[['Station Name', 'Port Type','Total Duration (minutes)' ]].groupby(['Station Name', 'Port Type']).mean()

seaborn_facet_duration = sns.catplot(
    x= 'Port Type',
    y = 'Total Duration (minutes)', 
    row = 'Station Name', 
    kind = 'bar',
    data = duration_Stations.reset_index()
)

seaborn_facet_duration

#duration_Stations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting to Time Series & Other related data prep

# COMMAND ----------

start__ = boulder_current['Start_Date___Time'].min() - dt.timedelta(minutes = boulder_current.loc[0,'Start_Date___Time'].minute % 10)
end__ = boulder_current['End_Date___Time'].max() + dt.timedelta(minutes = 10 - boulder_current.loc[0,'Start_Date___Time'].minute % 10)
all_times = pd.date_range(start = start__, end = end__, freq = '10min')

group_by_var = 'Address'

# dur__ = palo_alto.loc[0]['Total Duration (minutes)']

print(start__)
print(end__)
print(len(all_times))

expanded_df = pd.DataFrame(data = {'Date Time': all_times})
for loc in boulder_current[group_by_var].unique():
    expanded_df[loc] = [0] * len(all_times)
for session in range(boulder_current.shape[0]):
    start_time = boulder_current.loc[session, 'Start_Date___Time']
    end_time = boulder_current.loc[session, 'End_Date___Time']
    loc = boulder_current.loc[session, group_by_var]
    expanded_df.loc[(expanded_df['Date Time'] >= start_time) & (expanded_df['Date Time'] <= end_time), loc] += 1

# COMMAND ----------

expanded_df['Date'] = expanded_df['Date Time'].dt.date
expanded_df['Hour'] = expanded_df['Date Time'].dt.hour
expanded_df['Date Hour'] = expanded_df['Date Time'].dt.floor('h')

# COMMAND ----------

temp = pd.melt(expanded_df, id_vars = ['Date Time', 'Date', 'Hour', 'Date Hour'],
               var_name = 'Station', value_name = 'Ports Occupied', 
               value_vars = boulder_current[group_by_var].unique())
temp

# COMMAND ----------

temp_grouped = temp.groupby(['Date', 'Station']).mean().reset_index()

# COMMAND ----------

len(boulder_current[group_by_var].unique())

# COMMAND ----------

# temp = expanded_df[(expanded_df['Date Time'].dt.year == 2019) & (expanded_df['Date Time'].dt.month == 12) & (expanded_df['Date Time'].dt.day >= 29)]
station_names = boulder_current[group_by_var].unique()
rows = 5
cols = 5
fig, ax = plt.subplots(rows, cols, figsize = (40, 20))
for i in range(rows * cols):
    if i < len(station_names):
        temp_group_filtered = temp_grouped[temp_grouped['Station'] == station_names[i]]
        ax[int(i/cols),i%cols].plot(temp_group_filtered['Date'], temp_group_filtered['Ports Occupied']);
        ax[int(i/cols),i%cols].set_title(station_names[i])
fig.tight_layout()
# plt.savefig(path_to_210_folder + 'Summer 2022 Capstone/data/boulder_stations_time_series.png', dpi = 500);

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

## What % of stations are fully occupied / What % of time are stations fully occupied

# COMMAND ----------

for station in station_names:
  print(expanded_df[station].value_counts())