# Databricks notebook source
# MAGIC %md
# MAGIC Boulder Historical Data
# MAGIC *EDA, cleanup, etc.*
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   [Data Set](https://ev.caltech.edu/dataset) 

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



# COMMAND ----------

# read in as spark data frame
#jpl = spark.read.option("header", True) \
                      .csv(f"/mnt/{mount_name}/data/BoulderCO_data_Jan2018throughApr2022.csv")

#convert to pandas dataframe
#jpl = jpl.toPandas()

#print("Shape:", jpl.shape)

# COMMAND ----------

#jpl.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## General EDA - Data understanding

# COMMAND ----------

# what columns do I have and what type are they?
boulder.info()

# need to convert data and times.
# need to standardize time zones.

# COMMAND ----------

boulder.isnull().sum()

# only 1 null value, can fill this in with imputed data

# COMMAND ----------

boulder.nunique()

# COMMAND ----------

print(boulder['Start_Time_Zone' ].unique())
print(boulder['End_Time_Zone' ].unique())
# only mountain time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking Old Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting DTypes

# COMMAND ----------

#convert to date time
boulder_old['Start_Date___Time'] = pd.to_datetime(boulder_old['Start_Date___Time'])
boulder['Start_Date___Time'] = pd.to_datetime(boulder['Start_Date___Time'])
boulder_old['End_Date___Time'] = pd.to_datetime(boulder_old['End_Date___Time'])
boulder['End_Date___Time'] = pd.to_datetime(boulder['End_Date___Time'])

# COMMAND ----------

# convert duration columns to a standard measurement, and int object
# Convert to minutes.
boulder['Total Duration (minutes)'] = boulder['Total_Duration__hh_mm_ss_'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]) + int(x[2])*(1/60))
boulder['Charging Time (minutes)'] = boulder['Charging_Time__hh_mm_ss_'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]) + int(x[2])*(1/60))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does the new dataset have the same info as the old dataset?

# COMMAND ----------

boulder_old = boulder_old.sort_values(by = 'Start_Date___Time', axis = 0).reset_index()
boulder = boulder.sort_values(by = 'Start_Date___Time', axis = 0).reset_index()

# COMMAND ----------

n = 32635
checking = pd.DataFrame(data = {'matches': boulder_old['Start_Date___Time'].dt.month[:n] == boulder['Start_Date___Time'].dt.month[:n]})
checking[checking['matches'] == False]

# COMMAND ----------

# MAGIC %md
# MAGIC Yes, the new dataset (Jan 2018 through Apr 2022) matches the old dataset (Jan 2018 through Nov 2021) for the months where they overlap.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Static Station Data

# COMMAND ----------

sorted(boulder['Station_Name'].unique())

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