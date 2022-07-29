# Databricks notebook source
# MAGIC %md
# MAGIC SlrpEV Historical Data
# MAGIC *EDA, cleanup, etc.*
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   From Prof Scott Moura

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
from decimal import Decimal



## Connect to AWS
access_key = "AKIA3CS2H32VF7XY33S2"
secret_key = "0ZgHc4WyyfQn7uylzrSSdjPwIgJpvukdQZysZWWI"
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
slrp = spark.read.option("header", True) \
                      .csv(f"/mnt/{mount_name}/data/SlrpEV.csv")

#convert to pandas dataframe
slrp = slrp.toPandas()

print("Shape:", slrp.shape)

# COMMAND ----------

slrp.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## General EDA - Data understanding

# COMMAND ----------

# what columns do I have and what type are they?
slrp.info()

# need to convert data and times.
# need to standardize time zones.

# COMMAND ----------

slrp.isnull().sum()

# only 1 null value, can fill this in with imputed data

# COMMAND ----------

slrp.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting DTypes

# COMMAND ----------

#convert to date time
slrp['connectTime_dt'] = pd.to_datetime(slrp['connectTime']).dt.floor('T').dt.round(freq = '10T')
slrp['startChargeTime_dt'] = pd.to_datetime(slrp['startChargeTime']).dt.floor('T').dt.round(freq = '10T')
slrp['Deadline_dt'] = pd.to_datetime(slrp['Deadline']).dt.floor('T').dt.round(freq = '10T')

# COMMAND ----------

print(slrp['connectTime_dt'].min())
print(slrp['connectTime_dt'].max())

# COMMAND ----------

# Convert to ints/floats
slrp['siteId'] = slrp['siteId'].astype(int)
slrp['vehicle_maxChgRate_W'] = slrp['vehicle_maxChgRate_W'].astype(float)
slrp['energyReq_Wh'] = slrp['energyReq_Wh'].astype(float)
slrp['estCost'] = slrp['estCost'].astype(float)
slrp['DurationHrs'] = slrp['DurationHrs'].astype(float)
slrp['reg_centsPerHr'] = slrp['reg_centsPerHr'].astype(float)
slrp['sch_centsPerHr'] = slrp['sch_centsPerHr'].astype(float)
slrp['sch_centsPerKwh'] = slrp['sch_centsPerKwh'].astype(float)
slrp['sch_centsPerOverstayHr'] = slrp['sch_centsPerOverstayHr'].astype(float)
slrp['cumEnergy_Wh'] = slrp['cumEnergy_Wh'].astype(float)
slrp['peakPower_W'] = slrp['peakPower_W'].astype(float)

# COMMAND ----------

# Convert duration to minutes.
slrp['DurationMin'] = slrp['DurationHrs'] * 60

# Convert from duration charging to duration at station
slrp['DurationMin'] = slrp['DurationMin'] + ((slrp['startChargeTime_dt'] - slrp['connectTime_dt']).dt.total_seconds()/60)

# COMMAND ----------

slrp['Station'] = ['Slrp'] * slrp.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Stations

# COMMAND ----------

# Filter to correct site ID
slrp = slrp[slrp['siteId'] == 25]

# COMMAND ----------

fig, ax = plt.subplots(1, 1, figsize = (10, 5))
plt.hist(slrp['DurationMin'], bins = 50);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Visualizations

# COMMAND ----------

slrp.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Time Series Conversion Helper Functions

# COMMAND ----------

###### function to fill in times 
def stationTS_func(df, dateTimeCol, locationCol):
    '''fill in missing time stamps, assumes already in 10 minute intervals 
    must have a column with Station Location '''
    
    # station installed at different starts
    start_ts = df[str(dateTimeCol)].min()
    ### all end at same time
    end_ts = df[str(dateTimeCol)].max()
    all_times = pd.date_range(start = start_ts, end = end_ts, freq = '10min')
    
    # printing to confirm
    print(start_ts)
    print(end_ts)
    print(len(all_times))
    
    # Create df with only times, one column
    expanded_ts = pd.DataFrame(data = {'DateTime': all_times})
    
    #merge df with expanded time
    station_ts =  pd.merge(left = expanded_ts, right=df, how='left', left_on='DateTime', right_on=str(dateTimeCol))
    #replace empty station name col
    station_ts[locationCol].fillna(str(df[locationCol].unique()[0]), inplace = True)
    
    return station_ts

# COMMAND ----------

# MAGIC %md
# MAGIC # OLD Time Series Conversion -- DO NOT RUN

# COMMAND ----------

# copy for EDA only
df = slrp.copy()

# add date time parts
df['Date'] =  df['connectTime_dt'].dt.date
df['Month'] = df['connectTime_dt'].dt.month
df['Year'] = df['connectTime_dt'].dt.year
df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)
df['DayofWeek'] = df['connectTime_dt'].dt.weekday
df['IsWeekend'] = df['DayofWeek'] > 4
df['Hour'] = df['connectTime_dt'].dt.hour

df.columns

# COMMAND ----------

holiday_list = []
for ptr in holidays.US(years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).items():
    holiday_list.append(ptr)
us_holidays = pd.DataFrame(holiday_list).rename(columns = {0: 'date', 1: 'holiday_name'})
us_holidays['holiday'] = 1
us_holidays.info()

df = df.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)

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

# Create an array with the colors you want to use

# Set your custom color palette
colors = ["#7b7b7b", "#4f8b37"]
sns.set_palette(sns.color_palette(colors))

# COMMAND ----------

# Create an array with the colors you want to use

# Set your custom color palette


x_var = 'Hour'   
hue_var = 'IsWeekend' 
hue_var2 = 'holiday'

fig, ax = plt.subplots(1, 2, figsize = (20,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.lineplot(ax = ax[0], x = x_var, y= 'DurationMin', hue = hue_var, data = df, ci="sd");
ax[0].legend(loc="upper right", frameon=True)
ax[0].set(ylim=(0, 2000), ylabel = 'Duration in Minutes', xlabel = 'Hour')
ax[0].set_title("Total Duration by Start Hour and Weekend",fontsize=16)

sns.lineplot(ax = ax[1], x = x_var, y= 'DurationMin', hue = hue_var2, data = df, ci="sd");
ax[1].legend(loc="upper right", frameon=True)
ax[1].set(ylim=(0, 2000), ylabel = 'Duration in Minutes', xlabel = 'Hour')
ax[1].set_title("Total Duration by Start Hour and Holiday",fontsize=16)

#plt.savefig('/dbfs/FileStore/EDA_slrp_TotalDurationWeekendHoliday.png') 
plt.show()

# COMMAND ----------

############### Convert to Time Series #########################

#### create copy to apply changes and only pull necessary columns
slrp_ts = slrp[['stationId', 'connectTime_dt', 'DurationMin']].copy(deep = True)


##### add column with time ranges for start and end
# create copy
slrp_ts['DateTime'] = list(zip(slrp_ts['connectTime_dt'], slrp_ts['DurationMin']))
# apply a function to get time series list, start date, number of 10 minutes periods in duration, and use freq of 10 minutes
slrp_ts['DateTime'] = slrp_ts['DateTime'].apply(lambda x: list(pd.date_range(x[0], 
                                                                             periods = math.ceil(x[1]/10), 
                                                                             freq = '10T')))


#### explode
slrp_ts = slrp_ts.explode('DateTime')
print('Exploded Timeseries completed')

#drop start and end cols, no longer need
slrp_ts = slrp_ts.drop(['connectTime_dt', 'DurationMin'], axis = 1)


##### drop duplicates
# remove charging sessions that may have stopped and start in same 10 minute interval --- no double counting, is there double counting if used different connector?
print('Before duplicate drop:', slrp_ts.shape)
slrp_ts = slrp_ts.drop_duplicates()
slrp_ts.reset_index(inplace= True, drop=True)
print('After duplicate drop:', slrp_ts.shape)


##### Ports Occupied ###
# one for each charging session, by 10 minute interval
slrp_ts.loc[:,'Ports Occupied'] = int(1)
print('Added Ports Occupied')
print('After Ports:', slrp_ts.shape)

# palo_altoTS_prev = palo_altoTS.copy()

### Group by DateTime and station name , sum the port Occupied and concatenate all others
# note all station info columns must be object
slrp_ts = slrp_ts.groupby(['DateTime', 'stationId'], as_index=False).agg(lambda x: x.sum() if x.dtype=='int64' else ';'.join(x))
print('After group:', slrp_ts.shape)
slrp_ts['Ports Occupied'] = slrp_ts['Ports Occupied'].astype(float)

# COMMAND ----------

########## Fill in missing timestamps per station #########

slrp_ts_filled = stationTS_func(slrp_ts, 'DateTime', 'stationId')

## new timeseries
print('final dataframe', slrp_ts_filled.shape)

# fill in 0s for null ports occupied
slrp_ts_filled['Ports Occupied'] = slrp_ts_filled['Ports Occupied'].fillna(0)
slrp_ts_filled

# COMMAND ----------

#### Group by DateTime , sum the port Occupied and concatenate all others
## note all station info columns must be object
slrp_ts_grouped = slrp_ts_filled.groupby(['DateTime'], as_index=False).sum()
print('After group:', slrp_ts_grouped.shape)

# COMMAND ----------

slrp_ts_grouped['Ports Occupied'].value_counts()

# COMMAND ----------

slrp_ts_grouped['Ports Occupied'].mean()

# COMMAND ----------

# Cap the maximum at 8
slrp_ts_grouped['Plugs'] = 8
slrp_ts_grouped.loc[slrp_ts_grouped['Plugs'] < slrp_ts_grouped['Ports Occupied'], 'Ports Occupied'] = slrp_ts_grouped.loc[slrp_ts_grouped['Plugs'] < slrp_ts_grouped['Ports Occupied'], 'Plugs']

# COMMAND ----------

slrp_ts_grouped['Ports Occupied'].value_counts()

# COMMAND ----------

slrp_ts_grouped.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # New Time Series Conversion

# COMMAND ----------

power_data = pd.DataFrame()

for i in slrp.index:
    indiv_session = pd.DataFrame(eval(slrp.loc[i, 'power'])).sort_values(by = 'timestamp')
    indiv_session['power_W'] = indiv_session['power_W'].astype(float)
    indiv_session['power_W'] = indiv_session['power_W'].astype(float)
    indiv_session['timestamp'] = pd.to_datetime(indiv_session['timestamp'].astype(float), unit = 's')
    indiv_session['timestamp_rounded'] = indiv_session['timestamp'].dt.floor('T').dt.round(freq = '10T')
    indiv_session = indiv_session.groupby('timestamp_rounded').mean().reset_index()
    indiv_session['station_occupied'] = 1
    indiv_session['station_charging'] = (indiv_session['power_W'] > 0).astype(int)
    indiv_session['station'] = 'Slrp'
    
    power_data = pd.concat([power_data, indiv_session])

power_data = power_data.groupby(['timestamp_rounded', 'station']).sum().reset_index()
print(power_data.shape)

# COMMAND ----------

power_ts = stationTS_func(power_data, 'timestamp_rounded', 'station').drop(columns = 'timestamp_rounded').fillna(0)
print(power_ts.shape)
power_ts.head()

# COMMAND ----------

power_ts['station_occupied'].value_counts()

# COMMAND ----------

# add date time parts
power_ts['Date'] =  power_ts['DateTime'].dt.date
power_ts['Month'] = power_ts['DateTime'].dt.month
power_ts['Year'] = power_ts['DateTime'].dt.year
power_ts['Year-Month'] = power_ts['Year'].astype(str) + '-' + power_ts['Month'].astype(str)
power_ts['DayofWeek'] = power_ts['DateTime'].dt.weekday
power_ts['IsWeekend'] = power_ts['DayofWeek'] > 4
power_ts['Hour'] = power_ts['DateTime'].dt.hour

# COMMAND ----------

power_ts = power_ts.rename(columns = {'station_occupied': 'Ports Occupied', 'station_charging': 'Ports Charging'})
power_ts['Plugs'] = 8
power_ts.loc[power_ts['Plugs'] < power_ts['Ports Occupied'], 'Ports Occupied'] = power_ts.loc[power_ts['Plugs'] < power_ts['Ports Occupied'], 'Plugs']
power_ts['Ports Available'] = power_ts['Plugs'] - power_ts['Ports Occupied']

# COMMAND ----------

power_ts.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

## What % of stations are fully occupied / What % of time are stations fully occupied

# COMMAND ----------

temp = power_ts.copy()
# temp['Month'] = temp['DateTime'].dt.month
# temp['Year'] = temp['DateTime'].dt.year
# temp['Date'] = temp['DateTime'].dt.date
# temp['Year-Month'] = temp['Year'].astype(str) + '-' + temp['Month'].astype(str)

fig, ax = plt.subplots(2, 1, figsize = (15, 10))
temp1 = temp.groupby('Year-Month').mean().reset_index()
ax[0].plot(temp1['Year-Month'], temp1['Ports Occupied']);
temp2 = temp.groupby('Date').mean().reset_index()
ax[1].plot(temp2['Date'], temp2['Ports Occupied']);

# COMMAND ----------

x_var = 'Hour'   
hue_var = 'IsWeekend' 
#hue_var = 'holiday'

fig, ax = plt.subplots(1, 2, figsize = (15,8))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

temp = power_ts.copy()
# temp = power_ts.groupby(['Hour', 'IsWeekend']).mean()

sns.lineplot(ax = ax[0], x = x_var, y= 'Ports Occupied', hue = hue_var, data = temp, ci = 'sd');
ax[0].set_title('Ports Occupied by Hour', fontsize = 16);
ax[0].legend(loc = 'upper left', labels = ['Weekday', 'Weekend'], title = 'Part of Week')
ax[0].set(xlabel = 'Hour', xlim = (0, 23), ylim = (-0.5,4))
ax[0].xaxis.set_ticks(np.arange(0, 24, 2))
ax[0].yaxis.set_ticks(np.arange(0, 5 , 1))

sns.lineplot(ax = ax[1], x = x_var, y= 'power_W', hue = hue_var, data = temp, ci = 'sd');
ax[1].set_title('Power Consumption by Hour', fontsize = 16);
ax[1].legend(loc = 'upper left', labels = ['Weekday', 'Weekend'], title = 'Part of Week')
ax[1].set( ylabel = 'Power in Watts', xlabel = 'Hour', xlim = (0, 23))
ax[1].xaxis.set_ticks(np.arange(0, 24, 2))

plt.show()

# COMMAND ----------

means = power_ts.groupby(['Hour', 'IsWeekend']).mean()[['power_W', 'Ports Occupied']]
means['power_W_std'] = power_ts.groupby(['Hour', 'IsWeekend']).std()['power_W']
means['Ports Occupied Std Dev'] = power_ts.groupby(['Hour', 'IsWeekend']).std()['Ports Occupied']
means.head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC # Time Features

# COMMAND ----------

ucb_calendar = pd.DataFrame(data = {'start_date': [dt.date(2020, 11, 11),
                                                   dt.date(2020, 11, 25),
                                                   dt.date(2020, 12, 21),
                                                   dt.date(2020, 12, 24),
                                                   dt.date(2020, 12, 31),
                                                   dt.date(2021, 1, 18),
                                                   dt.date(2021, 2, 15),
                                                   dt.date(2021, 3, 22),
                                                   dt.date(2021, 5, 17),
                                                   dt.date(2021, 5, 31),
                                                   dt.date(2021, 9, 6),
                                                   dt.date(2021, 11, 11),
                                                   dt.date(2021, 11, 24),
                                                   dt.date(2021, 12, 20),
                                                   dt.date(2021, 12, 23),
                                                   dt.date(2021, 12, 30),
                                                   dt.date(2022, 1, 17),
                                                   dt.date(2022, 2, 21),
                                                   dt.date(2022, 3, 21),
                                                   dt.date(2022, 5, 16),
                                                   dt.date(2022, 5, 30)],
                                    'end_date': [dt.date(year = 2020, month = 11, day = 11), 
                                                 dt.date(2020, 11, 27),
                                                 dt.date(2021, 1, 12),
                                                 dt.date(2020, 12, 25),
                                                 dt.date(2021, 1, 1),
                                                 dt.date(2021, 1, 18),
                                                 dt.date(2021, 2, 15),
                                                 dt.date(2021, 3, 26),
                                                 dt.date(2021, 8, 18),
                                                 dt.date(2021, 5, 31),
                                                 dt.date(2021, 9, 6),
                                                 dt.date(2021, 11, 11),
                                                 dt.date(2021, 11, 26),
                                                 dt.date(2022, 1, 11),
                                                 dt.date(2021, 12, 24),
                                                 dt.date(2021, 12, 31),
                                                 dt.date(2022, 1, 17),
                                                 dt.date(2022, 2, 21),
                                                 dt.date(2022, 3, 25),
                                                 dt.date(2022, 8, 16),
                                                 dt.date(2022, 5, 30)],
                                   'note': ['Holiday', 
                                            'Holiday', 
                                            'Break',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Break',
                                            'Break',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Break',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Holiday',
                                            'Break',
                                            'Break',
                                            'Holiday'],
                                   'note2': ['Veterans Day', 
                                             'Thanksgiving', 
                                             'Winter Break',
                                             'Christmas',
                                             'New Years',
                                             'MLK Day',
                                             'Presidents Day',
                                             'Spring Break',
                                             'Summer Break',
                                             'Memorial Day',
                                             'Labor Day',
                                             'Veterans Day',
                                             'Thanksgiving',
                                             'Winter Break',
                                             'Christmas',
                                             'New Years',
                                             'MLK Day',
                                             'Presidents Day',
                                             'Spring Break',
                                             'Summer Break',
                                             'Memorial Day']})
ucb_calendar['date'] = [pd.date_range(s, e, freq='d') for s, e in
              zip(pd.to_datetime(ucb_calendar['start_date']),
                  pd.to_datetime(ucb_calendar['end_date']))]
ucb_calendar = ucb_calendar.explode('date').drop(columns = ['start_date', 'end_date'])
ucb_calendar['date'] = pd.to_datetime(ucb_calendar['date']).dt.date
ucb_calendar = ucb_calendar.reset_index(drop = True)
# Drop duplicates (keep the non-break duplicates)
ucb_calendar = ucb_calendar.drop(index = ucb_calendar[(ucb_calendar[['date']].duplicated(keep = False)) & (ucb_calendar['note'] == 'Break')].index)
ucb_calendar[(ucb_calendar['date'] > dt.date(2021, 7, 1)) & (ucb_calendar['date'] < dt.date(2021, 7, 20))]

# COMMAND ----------

holiday_list = []
for ptr in holidays.US(years = [2020, 2021, 2022]).items():
    holiday_list.append(ptr)
us_holidays = pd.DataFrame(holiday_list).rename(columns = {0: 'date', 1: 'holiday_name'})
us_holidays['holiday'] = 1
us_holidays.info()

# COMMAND ----------

power_ts.head()

# COMMAND ----------

power_ts_merged = power_ts.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0).merge(ucb_calendar, how = 'left', left_on = 'Date', right_on = 'date')\
                        .drop(columns = ['date']).fillna('None')

# COMMAND ----------

power_ts_merged.shape

# COMMAND ----------

power_ts_merged[(power_ts_merged['Date'] > dt.date(2021, 7, 2)) & (power_ts_merged['Date'] <= dt.date(2021, 7, 4))]

# COMMAND ----------

# fig, ax = plt.subplots(1, 1, figsize = (15, 5))
temp = power_ts_merged.copy()
temp2 = temp.groupby('Date').mean().reset_index()
temp2 = temp2.merge(ucb_calendar, how = 'left', left_on = 'Date', right_on = 'date').fillna('None')
fig = plt.subplots(figsize = (15, 5))
sns.scatterplot(x = temp2['Date'], y = temp2['Ports Occupied'], hue = temp2['note']);

# COMMAND ----------

power_ts_merged.columns

# COMMAND ----------

x_var = 'Hour' 
hue_var = 'note' 

fig, ax = plt.subplots(1, 2, figsize = (15,8))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

temp = power_ts_merged.copy()
# temp = power_ts.groupby(['Hour', 'IsWeekend']).mean()

sns.lineplot(ax = ax[0], x = x_var, y= 'Ports Occupied', hue = hue_var, data = temp, ci = 'sd');
ax[0].set_title('Ports Occupied by Hour', fontsize = 16);
ax[0].legend(loc="upper left", frameon=True, title = 'School Closures')
ax[0].set( ylabel = 'Port Occupied', xlabel = 'Hour', xlim = (0, 23))
ax[0].xaxis.set_ticks(np.arange(0, 24, 2))
ax[0].yaxis.set_ticks(np.arange(0, 5, 1))

txt="Break: Winter, Spring and Summer breaks in school sessions \nHolidays: U.S. Observed Holidays \nNone: Non-holiday and Non-break day"
plt.figtext(0.08, 0, txt, wrap=True, horizontalalignment='left', fontsize=12)
plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)

sns.lineplot(ax = ax[1], x = x_var, y= 'power_W', hue = hue_var, data = temp, ci = 'sd');
ax[1].set_title('Power Consumption by Hour', fontsize = 16);
ax[1].legend(loc="upper left", frameon=True, title = 'School Closures')
ax[1].set( ylabel = 'Power in Watts', xlabel = 'Hour', xlim = (0, 23), ylim=(-2500, 17500))
ax[1].xaxis.set_ticks(np.arange(0, 24, 2))


plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # WRITE TO S3 BUCKET

# COMMAND ----------

slrp_ts_save = spark.createDataFrame(power_ts_merged) 
display(slrp_ts_save)
    
## Write to AWS S3
(slrp_ts_save
#         .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/slrp_ts"))

# COMMAND ----------

# MAGIC %md
# MAGIC # READ FROM S3 BUCKET

# COMMAND ----------

slrp_ts = spark.read.parquet(f"/mnt/{mount_name}/data/slrp_ts")

# COMMAND ----------

slrp_ts.toPandas().head(50)

# COMMAND ----------

slrp_ts = slrp_ts.toPandas()

### add date time parts ###
slrp_ts['DayofWeek'] = slrp_ts['DateTime'].dt.weekday
slrp_ts['IsWeekend'] = slrp_ts['DayofWeek'] > 4
slrp_ts['Hour'] = slrp_ts['DateTime'].dt.hour

# COMMAND ----------

fig, ax = plt.subplots(figsize = (20,10))
sns.boxplot(x = 'Year-Month', 
            y= 'Ports Occupied',  
            palette="deep",
            data = slrp_ts);
ax.set_title('Station Occupancy by year-month')
ax.set_ylim([0, slrp_ts['Plugs'].max()])  

fig

# COMMAND ----------

fig, ax = plt.subplots(figsize = (20,10))
sns.lineplot(x = 'Month', 
            y= 'Ports Occupied',  
            palette="deep",
            data = slrp_ts);
ax.set_title('Station Occupancy by year-month')
ax.set_ylim([0, slrp_ts['Plugs'].max()])  

fig

# COMMAND ----------

fig, ax = plt.subplots(figsize = (20,10))
sns.lineplot(x = 'DayofWeek', y= 'Ports Occupied', hue ='holiday', data = slrp_ts);
ax.set_title(' ')
ax.set_ylim([0, slrp_ts['Plugs'].max()]) 

fig.tight_layout()

# COMMAND ----------



fig, ax = plt.subplots(figsize = (10,5))

sns.set( rc = {'axes.labelsize' : 14 })
sns.lineplot(x = 'Hour', y= 'Ports Occupied', hue = 'IsWeekend', data = slrp_ts);
ax.set_title('')
plt.legend(loc="upper right", frameon=True, fontsize=14, title = 'IsWeekend', title_fontsize = 14)
#fig.set_xticklabels(fig.get_xmajorticklabels(), fontsize = 18)
#fig.set_yticklabels(fig.get_ymajorticklabels(), fontsize = 18)


ax.set_xlabel( "Hour" , size = 14 )
ax.set_ylabel( "Ports Occupied" , size = 14 )
#ax.set_ylim([0, 8]) 

fig.tight_layout()

# COMMAND ----------

fig, ax = plt.subplots(figsize = (15, 10))
sns.lineplot(x = 'Hour', 
             y = 'Ports Occupied',
             hue = 'DayofWeek',
             data = slrp_ts);
ax.set_title('')
fig.tight_layout()

# COMMAND ----------

print(slrp_ts.shape)

fully_occupied = slrp_ts[slrp_ts['Ports Occupied'] == slrp_ts['Plugs']]
print(fully_occupied.shape)

# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose

# COMMAND ----------

slrp_ts.columns

# COMMAND ----------

df = slrp_ts[['DateTime', 'station', 'Ports Occupied', 'Year-Month']].groupby(['DateTime', 'station', 'Year-Month']).mean()
df.reset_index()

fig, ax = plt.subplots(figsize = (15, 10))
sns.lineplot(x = 'Year-Month', 
             y = 'Ports Occupied',
             data = df, ci='sd');
ax.set_title('')
fig.tight_layout()

# COMMAND ----------

### look at seasonality
df =slrp_ts[slrp_ts['DateTime'] > dt.datetime(2022, 1,1,0,0,0)]
df.set_index('DateTime')

results = seasonal_decompose(df['Ports Occupied'], period = 6*24*7) #period = 1 week, 6*24*7
#results = seasonal_decompose(df['Ports Occupied'], period = 6*24) #period = period = 1 day 6*24

fig = results.plot(observed=True, seasonal=True, trend=True, resid=True)
fig.set_size_inches((20,10))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleaning Time Series Data

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Remove Invalid Stations
# MAGIC 2. Cap number of occupied chargers to the total number of chargers at the location

# COMMAND ----------

