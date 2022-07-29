# Databricks notebook source
# MAGIC %md
# MAGIC # Boulder Historical Data
# MAGIC *EDA, cleanup, etc.*
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   [Data Set](https://open-data.bouldercolorado.gov/datasets/39288b03f8d54b39848a2df9f1c5fca2_0/about) April 2022
# MAGIC *   [Static Info on Boulder Charging Stations](https://bouldercolorado.gov/services/electric-vehicle-charging-stations)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Data

# COMMAND ----------

# import libraries
import pandas as pd
# import altair as alt
import datetime as dt
import numpy as np
import json
import time
import urllib
import requests
import seaborn as sns
from pandas.io.json import json_normalize
from datetime import timedelta
import plotly.graph_objects as go
import pytz
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
import folium
import holidays



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

# read in as spark data frame
boulder = spark.read.option("header", True) \
                      .csv(f"/mnt/{mount_name}/data/BoulderCO_data_Jan2018throughApr2022.csv")

boulder_old = spark.read.option("header", True) \
                      .csv(f"/mnt/{mount_name}/data/BoulderCO_data_Jan2018throughNov2021.csv")

#convert to pandas dataframe
boulder = boulder.toPandas()
boulder_old = boulder_old.toPandas()

print("Shape:", boulder.shape)
print("Shape:", boulder_old.shape)

boulder.head()

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

print(boulder['Start_Time_Zone'].unique())
print(boulder['End_Time_Zone'].unique())
# only mountain time

# COMMAND ----------

boulder[['Start_Time_Zone', 'End_Time_Zone']].drop_duplicates()

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

static_station_data

# COMMAND ----------

pricing = '$1 for the first two hours; $2.50 per hour for each additional hour*'
static_station_data.loc[len(static_station_data)] = ['Boulder Park', '1739 Broadway', 2, 'Free', 'Level 2']
static_station_data.loc[len(static_station_data)] = ['Boulder Facilities', '1745 14th Street', 2, 'Free', 'Level 2']
static_station_data.loc[len(static_station_data)] = ['1100 Parking Garage', '1100 Spruce St', 2, 'Free', 'Level 2']
static_station_data.loc[len(static_station_data)] = ['5050 Parking Garage', '5050 Pearl St', 2, 'Free', 'Level 2']
static_station_data.loc[len(static_station_data)] = ['1500 Parking Garage', '1500 Pearl St', 4, 'Free', 'Level 2']
static_station_data.loc[14, 'Address'] = '1400 Walnut St'
static_station_data

# COMMAND ----------

def standardize_address(data):
    return [i[0] + ' ' + i[1].lower() for i in data.str.split(' ')]

# COMMAND ----------

static_names = standardize_address(static_station_data['Address'])
static_station_data['Standardized_Location'] = static_names

# COMMAND ----------

for address in np.sort(boulder['Address'].unique()):
    address = address.split(' ')
    address = (address[0] + ' ' + address[1]).lower()
    if address in static_names:
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
# MAGIC ## Charging Sessions: EDA
# MAGIC 
# MAGIC * how long are charging sessions?
# MAGIC * Is the duration longer on weekend or weekday?
# MAGIC * does the duration change during holidays?

# COMMAND ----------

# copy for EDA only
cseda = boulder.copy()

# add date time parts
cseda['Date'] =  cseda['Start_Date___Time'].dt.date
cseda['Month'] = cseda['Start_Date___Time'].dt.month
cseda['Year'] = cseda['Start_Date___Time'].dt.year
cseda['Year-Month'] = cseda['Year'].astype(str) + '-' + cseda['Month'].astype(str)
cseda['DayofWeek'] = cseda['Start_Date___Time'].dt.weekday
cseda['IsWeekend'] = cseda['DayofWeek'] > 4
cseda['Hour'] = cseda['Start_Date___Time'].dt.hour

cseda.columns

# COMMAND ----------

holiday_list = []
for ptr in holidays.US(years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).items():
    holiday_list.append(ptr)
us_holidays = pd.DataFrame(holiday_list).rename(columns = {0: 'date', 1: 'holiday_name'})
us_holidays['holiday'] = 1
us_holidays.info()


# COMMAND ----------

cseda = cseda.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (15,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.boxplot(ax = ax[0], x = 'holiday', y= 'Total Duration (minutes)', data = cseda);
ax[0].set_title('Total Duration')
ax[0].set(ylim=(0, 1000))

sns.boxplot(ax = ax[1], x = 'holiday', y= 'Charging Time (minutes)', data = cseda);
ax[1].set_title('Charging Time')
ax[1].set(ylim=(0, 1000))

# COMMAND ----------

siteName = sorted(cseda['Station_Name'].unique())
rows = 6
cols = 5

fig, ax = plt.subplots(rows, cols, figsize = (30,20))
fig.subplots_adjust(hspace=0.5, wspace=0.25)
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = cseda[cseda['Station_Name'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['holiday'], y= temp_filter['Total Duration (minutes)']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        
        #ax[int(i/cols), i%cols].set_ylim([0, 1])   
        
fig

# COMMAND ----------

#x_var = 'IsWeekend' # duations slightly less at the end of the week
#x_var = 'DayofWeek'  # charging duration is a little less duration end of week
x_var = 'Hour'   ## charging durations/total durations are longer when started in evening

fig, ax = plt.subplots(1, 2, figsize = (15,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.boxplot(ax = ax[0], x = x_var, y= 'Total Duration (minutes)', data = cseda);
ax[0].set_title('Total Duration')
ax[0].set(ylim=(0, 1000))

sns.boxplot(ax = ax[1], x = x_var, y= 'Charging Time (minutes)', data = cseda);
ax[1].set_title('Charging Time')
ax[1].set(ylim=(0, 1001))

# COMMAND ----------

x_var = 'Hour'   ## charging durations/total durations are longer when started in evening
hue_var = 'IsWeekend'  ## duration tend to be higher when stat in the late evening when its a weekend. durations are shorter during day, 8 - 3pm
#hue_var = 'holiday'

fig, ax = plt.subplots(1, 2, figsize = (20,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.lineplot(ax = ax[0], x = x_var, y= 'Total Duration (minutes)', hue = hue_var, ci='sd', data = cseda);
ax[0].set_title('Total Duration')
#ax[0].set(ylim=(0, 1000))

sns.lineplot(ax = ax[1], x = x_var, y= 'Charging Time (minutes)',hue = hue_var, ci='sd', data = cseda);
ax[1].set_title('Charging Time')
ax[1].set(ylim=(0, 1000))

# COMMAND ----------

x_var = 'Year-Month'
y_var = 'Total Duration (minutes)'
y_var2 = 'Charging Time (minutes)'

siteName = sorted(cseda['Station_Name'].unique())
rows = 10
cols = 3


fig, ax = plt.subplots(2, 1, figsize = (30,20))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.lineplot(ax = ax[0], x = x_var, y= y_var, data = temp_filter);
ax[0].set_title(y_var)

sns.lineplot(ax = ax[1], x = x_var, y= y_var2, data = temp_filter);
ax[1].set_title(y_var2)
          
        
fig

# COMMAND ----------

x_var = 'Year-Month'
y_var = 'Total Duration (minutes)'

siteName = sorted(cseda['Station_Name'].unique())
rows = 10
cols = 3


fig, ax = plt.subplots(rows, cols, figsize = (30,20))
fig.subplots_adjust(hspace=0.5, wspace=0.25)
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = cseda[cseda['Station_Name'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = x_var, y= y_var, data = temp_filter);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        #ax[int(i/cols), i%cols].set(ylim=(0, 1000))
          
        
fig


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Visualizations

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

temp['Month'] = temp['Date Time'].dt.month
temp['Year'] = temp['Date Time'].dt.year
get_invalid_stations = temp.groupby(['Year', 'Month', 'Station']).mean()

# COMMAND ----------

get_invalid_stations = get_invalid_stations.reset_index()
invalid_stations = standardize_address(get_invalid_stations[(get_invalid_stations['Year'] == 2022) & (get_invalid_stations['Month']==4) & (get_invalid_stations['Ports Occupied']<=0.01)]['Station'])
invalid_stations

# COMMAND ----------

temp.columns

# COMMAND ----------

test = pd.DataFrame(temp[(temp['Station'] == '7315 Red Deer Dr') | (temp['Station'] == '5333 Valmont Rd')].sort_values(by = 'Date Time').head(30)['Station'].str.split(' ').to_list())
test['Address'] = test[0] + [' ']*len(test[0]) + test[1]
test

# COMMAND ----------

temp_2 = pd.DataFrame(temp['Station'].str.split(' ').to_list())
temp['Standardized Station'] = temp_2[0] + [' ']*len(temp_2[0]) + temp_2[1]

# COMMAND ----------

print(temp.shape)
temp_filtered = temp[temp['Standardized Station'].str not in invalid_stations]
print(temp_filtered.shape)

# COMMAND ----------

temp.loc[:,'Ports Available'] = temp['Plugs'] - temp['Ports Occupied']

# COMMAND ----------

# MAGIC %md
# MAGIC # WRITE TO S3 BUCKET

# COMMAND ----------

boulder_ts = spark.createDataFrame(temp) 
display(boulder_ts)
    
## Write to AWS S3
(boulder_ts
#         .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/boulder_ts"))

# COMMAND ----------

# MAGIC %md
# MAGIC # READ FROM S3 BUCKET

# COMMAND ----------

boulder_ts = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_ts")

boulder_ts = boulder_ts.toPandas()
print(boulder_ts['Station'].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleaning Time Series Data

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Remove Invalid Stations
# MAGIC 2. Cap number of occupied chargers to the total number of chargers at the location

# COMMAND ----------

get_invalid_stations = boulder_ts.groupby(['Year', 'Month', 'Station']).mean()
get_invalid_stations = get_invalid_stations.reset_index()
invalid_stations = get_invalid_stations[(get_invalid_stations['Year'] == 2022) & (get_invalid_stations['Month']==4) & (get_invalid_stations['Ports Occupied']<=0.01)]['Station']
invalid_stations

# COMMAND ----------

boulder_ts_filtered = boulder_ts[~boulder_ts['Station'].isin(invalid_stations)]

print(boulder_ts_filtered['Station'].unique())

# COMMAND ----------

stations = pd.DataFrame(data = {'proper_name': boulder_ts_filtered['Station'].unique(),
                                'standardized_name': standardize_address(pd.Series(boulder_ts_filtered['Station'].unique()))})
all_static_data = stations.merge(static_station_data, left_on = 'standardized_name', right_on = 'Standardized_Location')
all_static_data['Plugs'] = all_static_data['Plugs'].astype(int)
all_static_data.head()

# COMMAND ----------

all_static_data.info()

# COMMAND ----------

boulder_ts_filtered_merged = boulder_ts_filtered.merge(all_static_data[['proper_name', 'Plugs', 'Fee', 'Notes']],
                         left_on = 'Station', right_on = 'proper_name')

boulder_ts_filtered_merged.loc[boulder_ts_filtered_merged['Plugs'] < boulder_ts_filtered_merged['Ports Occupied'], 'Ports Occupied'] = boulder_ts_filtered_merged.loc[boulder_ts_filtered_merged['Plugs'] < boulder_ts_filtered_merged['Ports Occupied'], 'Plugs']

boulder_ts_filtered_merged[boulder_ts_filtered_merged['Plugs'] < boulder_ts_filtered_merged['Ports Occupied']]

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Cleaned Time Series to S3

# COMMAND ----------

temp = spark.createDataFrame(boulder_ts_filtered_merged) 
display(temp)
    
## Write to AWS S3
(temp
#         .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/boulder_ts_clean"))

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA On cleaned Data

# COMMAND ----------

# temp = expanded_df[(expanded_df['Date Time'].dt.year == 2019) & (expanded_df['Date Time'].dt.month == 12) & (expanded_df['Date Time'].dt.day >= 29)]
temp = boulder_ts_filtered_merged[(boulder_ts_filtered_merged['Date Time'].dt.year == 2022) &
                                 (boulder_ts_filtered_merged['Date Time'].dt.month == 4) &
                                 (boulder_ts_filtered_merged['Date Time'].dt.day < 20)]
station_names = all_static_data['proper_name'].unique()
rows = 5
cols = 4
fig, ax = plt.subplots(rows, cols, figsize = (40, 20))
for i in range(rows * cols):
    if i < len(station_names):
        temp_filtered = temp[temp['Station'] == station_names[i]]
        ax[int(i/cols),i%cols].plot(temp_filtered['Date Time'], temp_filtered['Ports Occupied']);
        ax[int(i/cols),i%cols].set_title(station_names[i])
fig.tight_layout()
# plt.savefig(path_to_210_folder + 'Summer 2022 Capstone/data/boulder_stations_time_series.png', dpi = 500);

# COMMAND ----------

boulder_ts_filtered_merged['fully_occupied'] = (boulder_ts_filtered_merged['Ports Occupied'] == boulder_ts_filtered_merged['Plugs'])
boulder_ts_filtered_merged['weekday'] = boulder_ts_filtered_merged['Date Time'].dt.weekday

# COMMAND ----------

pd.set_option("display.max_rows", None, "display.max_columns", None)
avg_occupancy = boulder_ts_filtered_merged.groupby(['Station', 'weekday', 'Hour']).mean().reset_index()

# COMMAND ----------

# temp = expanded_df[(expanded_df['Date Time'].dt.year == 2019) & (expanded_df['Date Time'].dt.month == 12) & (expanded_df['Date Time'].dt.day >= 29)]
station_names = all_static_data['proper_name'].unique()
rows = 5
cols = 4
fig, ax = plt.subplots(rows, cols, figsize = (40, 20))
for i in range(rows * cols):
    if i < len(station_names):
        temp_filtered = avg_occupancy[avg_occupancy['Station'] == station_names[i]]
        sns.lineplot(ax = ax[int(i/cols),i%cols], x = temp_filtered['Hour'], y = temp_filtered['fully_occupied'],
                hue = temp_filtered['weekday']);
        ax[int(i/cols),i%cols].set_title(station_names[i])
fig.tight_layout()
# plt.savefig(path_to_210_folder + 'Summer 2022 Capstone/data/boulder_stations_time_series.png', dpi = 500);

# COMMAND ----------



# COMMAND ----------

boulder_ts = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_ts_clean")
boulder_ts = boulder_ts.toPandas()

print("Columns: ", boulder_ts.columns)
print("Num Unique Stations: ", len(boulder_ts['Station'].unique()))

print("Number of Ports: ", boulder_ts.groupby(['Station', 'Plugs']).sum().reset_index()['Plugs'].sum())

# COMMAND ----------

invalid_stations = ['275 Alpine Ave', '2150 13th St', '2240 Broadway', '2280 Junction Pl', 
                    '5050 Pearl St', '7315 Red Deer Dr', '900 Baseline Rd']
boulder[~boulder['Address'].isin(invalid_stations)].shape

# COMMAND ----------

boulder_ts.head()

# COMMAND ----------

### create dataframe of holidays ####
 
holiday_list = []
for ptr in holidays.US(years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).items():
    holiday_list.append(ptr)
us_holidays = pd.DataFrame(holiday_list).rename(columns = {0: 'date', 1: 'holiday_name'})
us_holidays['holiday'] = 1


us_holidays

# COMMAND ----------

### add date time parts ###
boulder_ts['Year-Month'] = boulder_ts['Year'].astype(str) + '-' + boulder_ts['Month'].astype(str)
boulder_ts['DayofWeek'] = boulder_ts['Date Time'].dt.weekday
boulder_ts['IsWeekend'] = boulder_ts['DayofWeek'] > 4

### Merge with holidays
boulder_ts = boulder_ts.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)


### View new dataframe
boulder_ts.head()

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (20,10))
sns.lineplot(ax = ax[0], x = 'Hour', y= 'Ports Occupied', hue = 'IsWeekend', ci='sd', data = boulder_ts);
ax[0].set_title('Ports Occupied by Hour and Weekend')

sns.lineplot(ax = ax[1], x = 'Hour', y= 'Ports Occupied', hue = 'holiday', ci='sd', data = boulder_ts);
ax[1].set_title('Ports Occupied by Hour and holiday')

fig.tight_layout()

# COMMAND ----------

temp = boulder_ts#[['Station', 'IsWeekend', 'Hour', 'Ports Occupied']].groupby(['Station', 'IsWeekend', 'Hour'], as_index = False).mean()
#temp.head()


# #line charts like jennys
siteName = sorted(boulder_ts['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Hour'], y= temp_filter['Ports Occupied'], hue = temp_filter['IsWeekend'], ci='sd');
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, 2]) 

fig.tight_layout()

# COMMAND ----------

x_var = 'Hour'
y_var = 'Ports Occupied'


fig, ax = plt.subplots(figsize = (10,10))
sns.lineplot(x = x_var, y= y_var, data = boulder_ts);
ax.set_title('') 

plt.show()

# COMMAND ----------

temp = boulder_ts[['Station', 'Hour', 'holiday', 'Ports Occupied']]
#temp.head()


# #line charts like jennys
siteName = sorted(boulder_ts['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Hour'], y= temp_filter['Ports Occupied'], hue = temp_filter['holiday']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        #ax[int(i/cols), i%cols].set_ylim([0, 1])   

fig.tight_layout()

# COMMAND ----------

siteName = sorted(boulder_ts['Station'].unique())
rows = 8
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = boulder_ts[boulder_ts['Station'] == siteName[i]]
        sns.scatterplot(ax = ax[int(i/cols), i%cols], 
                        x = 'Date', 
                        y= 'Ports Occupied',  
                        palette="deep",
                       linewidth=0,
                       data = temp_filter);
        
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, boulder_ts[boulder_ts['Station'] == siteName[i] ]['Plugs'].max()])  

fig.tight_layout()

# COMMAND ----------

temp = boulder_ts[['Station', 'Month','DayofWeek', 'Ports Occupied', 'Plugs']]
#temp.head()


siteName = sorted(boulder_ts['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Month'], y= temp_filter['Ports Occupied'], hue = temp_filter['DayofWeek']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station'] == siteName[i] ]['Plugs'].max()])    

fig

# COMMAND ----------

temp = boulder_ts#[['Station', 'DayofWeek', 'holiday', 'Ports Occupied', 'Plugs']]
#temp.head()


# #line charts like jennys
siteName = sorted(boulder_ts['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = temp_filter['DayofWeek'], y= temp_filter['Ports Occupied'], hue = temp_filter['holiday']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station'] == siteName[i] ]['Plugs'].max()]) 

fig.tight_layout()

# COMMAND ----------

siteName = sorted(boulder_ts['Station'].unique())
rows = 18
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (60,30))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = boulder_ts[boulder_ts['Station'] == siteName[i]]
        sns.boxplot(ax = ax[i], 
                        x = 'Year-Month', 
                        y= 'Ports Occupied',  
                        palette="deep",
                       data = temp_filter);
        
        ax[i].set_title(siteName[i])
        ax[i].set_ylim([0, boulder_ts[boulder_ts['Station'] == siteName[i] ]['Plugs'].max()])  

fig

# COMMAND ----------

fig, ax = plt.subplots(figsize = (10,10))
sns.lineplot(x = 'DayofWeek', y= 'Ports Occupied', data = boulder_ts)
ax.set_title('')
ax.set_ylim([0, 2])

plt.show()

# COMMAND ----------

temp = boulder_ts[['Station', 'DayofWeek', 'holiday', 'Ports Occupied', 'Plugs']]
#temp.head()


# #line charts like jennys
siteName = sorted(boulder_ts['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = temp_filter['DayofWeek'], y= temp_filter['Ports Occupied']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station'] == siteName[i] ]['Plugs'].max()]) 

fig.tight_layout()

# COMMAND ----------

print(boulder_ts.shape)

fully_occupied = boulder_ts[boulder_ts['Ports Occupied'] == boulder_ts['Plugs']]
print(fully_occupied.shape)

fully_occupied

# COMMAND ----------


fig, ax = plt.subplots( figsize = (20,10))
sns.histplot(data=fully_occupied, x="proper_name")
plt.xticks(rotation=70)


fig.tight_layout()

# COMMAND ----------

print('Percent Boulder records are fully occupied: {:.2%}'.format(fully_occupied[fully_occupied['proper_name'] == '1739 Broadway'].shape[0] / boulder_ts[boulder_ts['proper_name'] == '1739 Broadway'].shape[0]))

names = sorted(fully_occupied['proper_name'].unique())

percent_fullyOccupied = list()
percent_Weekend = list()
percent_holiday=list()

for name in names:
    percent_fullyOccupied.append(
        str(
            round(100*(
                fully_occupied[fully_occupied['proper_name'] == name].shape[0] / boulder_ts[boulder_ts['proper_name'] == name].shape[0])
        , 2))
        + '%')


for name in names:
    percent_Weekend.append(
        str(
            round(100*(
                fully_occupied[(fully_occupied['proper_name'] == name) & (fully_occupied['IsWeekend'] == 1)].shape[0] / boulder_ts[boulder_ts['proper_name'] == name].shape[0])
        , 2))
        + '%')

    
for name in names:
    percent_holiday.append(
        str(
            round(100*(
                fully_occupied[(fully_occupied['proper_name'] == name) & (fully_occupied['holiday'] == 1)].shape[0] / boulder_ts[boulder_ts['proper_name'] == name].shape[0])
        , 2))
        + '%')
    


df = pd.DataFrame()
data = {'Station':names, 
        'Percent Fully Occupied':percent_fullyOccupied, 
        'Percent Fully Occupied & Weekend':percent_Weekend, 
        'Percent Fully Occupied & Holiday': percent_holiday}


df = pd.DataFrame(data)
df = df.set_index('Station', drop=True)
df = df.sort_values(by=['Percent Fully Occupied'], ascending=False)
df

# COMMAND ----------

result = df.to_html()
print(result)

# COMMAND ----------

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# # Set your custom color palette
# colors = ["#7b7b7b", "#4f8b37"]
# sns.set_palette(sns.color_palette(colors))

# COMMAND ----------

fully_occupied.columns

# COMMAND ----------

test = fully_occupied[['Station', 'Year', 'proper_name']].groupby(['Station','Year']).count().rename(columns={'proper_name':'Count'}).reset_index()

# Set your custom color palette
colors = ["#7BD159", "#59B7D1", "#AF59D1", "#D17359", "#F8E67C", "#378B49", "#7C8EF8", "#F87CCC"]
sns.set_palette(sns.color_palette(colors))



x_var = 'Year'  
hue_var = 'Station'  
y_var = 'Count'


rows = 2
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (15,15))
fig.subplots_adjust(hspace=0.18, wspace=0.15)

sns.lineplot( ax = ax[0], x = x_var, y= y_var, data = test, ci='sd', linewidth = 2);
ax[0].set_title('Count of Records with Fully Occupied Charging Ports', fontsize = 16);
ax[0].set( ylabel = 'Count of Records', xlabel = 'Year', ylim= (0, 13000))
ax[0].yaxis.set_ticks(np.arange(0, 13000, 2500))
ax[0].xaxis.set_ticks(np.arange(2018, 2023, 1))

sns.lineplot( ax = ax[1], x = x_var, y= y_var, hue = hue_var, data = test, ci='sd', linewidth = 2);
ax[1].set_title('Count of Records with Fully Occupied Charging Ports By Charging Station Location', fontsize = 16);
ax[1].legend(loc="upper right", frameon=True, title = 'Charging Station Location', bbox_to_anchor=(1.21, 1.02))
ax[1].set( ylabel = 'Count of Records', xlabel = 'Year', ylim= (0, 13000))
ax[1].yaxis.set_ticks(np.arange(0, 13000, 2500))
ax[1].xaxis.set_ticks(np.arange(2018, 2023, 1))

#fig.suptitle('Count of Records with Fully Occupied Charging Ports', fontsize=18)

plt.show()

# COMMAND ----------

rows = 1
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot( ax = ax[0], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=fully_occupied);
ax[0].set_title('Total Records Fully Occupied by Hour and Part of Week', fontsize = 16);
ax[0].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
ax[0].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 5500))
ax[0].yaxis.set_ticks(np.arange(0, 5500, 1000))


sns.histplot( ax = ax[1], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=fully_occupied);
ax[1].set_title('Total Records Fully Occupied by Hour and Holiday', fontsize = 16);
ax[1].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
ax[1].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 5500))
ax[1].yaxis.set_ticks(np.arange(0, 5500, 1000))



fig.tight_layout()

# COMMAND ----------

df = boulder_ts[boulder_ts['Ports Occupied'] != 0]

rows = 1
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot( ax = ax[0], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=df);
ax[0].set_title('Records with a minimum of 1 Port in Use by\n Hour and Part of Week', fontsize = 16);
ax[0].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
ax[0].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 40000))
#ax[0].yaxis.set_ticks(np.arange(0, 11000, 2000))


sns.histplot( ax = ax[1], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=df);
ax[1].set_title('Records with a minimum of 1 Port in Use by\n Hour and Holiday', fontsize = 16);
ax[1].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
ax[1].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 40000))
#ax[1].yaxis.set_ticks(np.arange(0, 11000, 2000))



fig.tight_layout()

# COMMAND ----------

### minimum of 1 port occupied stations by weekend and hour
df = boulder_ts[boulder_ts['Ports Occupied'] != 0]
sitename = sorted(df['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = df[df['Station'] == siteName[i]]
        sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=temp_filter);
        ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Part of Week for ' + str(siteName[i]), fontsize = 16);
        ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
        ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
        ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


#         sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=temp_filter);
#         ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Holiday', fontsize = 16);
#         ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
#         ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
#         ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


fig.tight_layout()

# COMMAND ----------

### fully occupied stations by weekend and hour
df = fully_occupied
sitename = sorted(df['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = df[df['Station'] == siteName[i]]
        sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=temp_filter);
        ax[int(i/cols), i%cols].set_title('Records with Ports Fully Occupied by\n Hour and Part of Week for ' + str(siteName[i]), fontsize = 16);
        ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
        ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')#, ylim= (0, 40000))
        ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))
        #ax[int(i/cols), i%cols].yaxis.set_ticks(np.arange(0, 11000, 2000))


#         sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=df);
#         ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Holiday', fontsize = 16);
#         ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
#         ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 40000))
#         #ax[int(i/cols), i%cols].yaxis.set_ticks(np.arange(0, 11000, 2000))

fig.tight_layout()

# COMMAND ----------

### fully occupied stations by holiday and hour
df = fully_occupied
sitename = sorted(df['Station'].unique())
rows = 6
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = df[df['Station'] == siteName[i]]
#         sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=temp_filter);
#         ax[int(i/cols), i%cols].set_title('Records with Ports Fully Occupied by\n Hour and Part of Week for ' + str(siteName[i]), fontsize = 16);
#         ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
#         ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
#         ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


        sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=temp_filter);
        ax[int(i/cols), i%cols].set_title('Records with fully Occupoed ports in Use by\n Hour and Holiday', fontsize = 16);
        ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
        ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
        ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))

fig.tight_layout()