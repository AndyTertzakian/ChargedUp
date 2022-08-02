# Databricks notebook source
# MAGIC %md
# MAGIC # Palo Alto Historical Data
# MAGIC *EDA, cleanup, etc.*
# MAGIC 
# MAGIC ### Data source notes
# MAGIC *   [Data Set](https://data.cityofpaloalto.org/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/) though 2020
# MAGIC *   https://www.cityofpaloalto.org/Departments/Utilities/Sustainability/Electric-Vehicle
# MAGIC *   https://www.cityofpaloalto.org/Departments/Utilities/Sustainability/Electric-Vehicle/EV-FAQs
# MAGIC 
# MAGIC <br>
# MAGIC Public chargers in Palo Alto **cost 23 cents per kilowatt-hour (kWh) with an “overstay” fee for drivers that leave their vehicles parked at a City charging station after their car is fully charged** . In 3 hours and $5, a Nissan Leaf would get about 70 miles of charge and a Tesla Model 3 would get about 100 miles of charge.
# MAGIC 
# MAGIC **Starting August 1, 2021 the City will be removing a $2 per hour overstay fee** at our City-owned charging stations to allow for overnight (5pm - 8am) parking. With this change, the City is attempting to make it easier to drive electric by providing more convenient access to charging for residents and members of the public who may not have access to charging at home. The **current $0.23 per kWh charge** will remain in effect, but **after your car is fully charged, there will no longer be an overstay fee.** 
# MAGIC 
# MAGIC 
# MAGIC <br>
# MAGIC Although CPAU currently does not offer a city-wide TOU rate, we highly recommend EV charging during post peak night hours to help lower the load and alleviate stress on the distribution grid. CPAU also recommends EV charging during the day in the Spring to consume excess solar PV generation.
# MAGIC 
# MAGIC CPAU recommends EV charging during the following time periods:
# MAGIC 
# MAGIC Nighttime: 11 pm to 6 am (All Seasons)
# MAGIC Daytime: 9 am to 3 pm (Spring only)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <b>
# MAGIC Where can I ask EV or EV Charger related questions?
# MAGIC Please contact the following City of Palo Alto offices:
# MAGIC 
# MAGIC **General EV or EV Charger Questions: Utilities Program Services at (650) 329-2241**
# MAGIC Permit Related Questions: Development Services at (650) 329-2496
# MAGIC Utility Service Upgrade Questions: Utilities Electrical Engineering at (650) 566-4500
# MAGIC 
# MAGIC   
# MAGIC   https://www.plugshare.com/location/273038 
# MAGIC   
# MAGIC Contact e-mail for this Dataset: eric.wong@cityofpaloalto.org

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Publicly Accessible City-owned EV Chargers as of August 2021:
# MAGIC 
# MAGIC |Location | Number of Level 2 Ports |
# MAGIC | ------- | ------- |
# MAGIC |Bryant St. Parking Garage 445 Bryant Street | 13 |
# MAGIC | Cambridge Parking Garage 475 Cambridge Ave. | 10|
# MAGIC | City Hall Parking Garage 250 Hamilton Ave. | 12 |
# MAGIC | High St. Parking Garage 528 High Street | 8 |
# MAGIC | Mitchell Park Library 3700 Middlefield Road | 3 |
# MAGIC | Palo Alto Junior Museum| 6 |
# MAGIC | Rinconada Library 1213 Newell Road | 3 |
# MAGIC | Sherman Parking Garage 350 Sherman Ave. | 33 |
# MAGIC | Ted Thompson Parking Garage 275 Cambridge Ave. | 8 |
# MAGIC | Webster St. Parking Garage 520 Webster Street | 20 |
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC - **bryant** #1 has 3 --  level 1 removed in 2019, snf upgraded in 2020 rest of bryant has 2 = total 13 (although 3 of these is a level 1 charger) -- assume level 1s were upgraded to level 2 in 2020 or onward
# MAGIC     - have a total of 12 ports at bryant 
# MAGIC     - 13th must have been added later
# MAGIC - **cambridge** 2 ports each = 10, all level 2
# MAGIC     - total 10 ports
# MAGIC - **hamilton**, have 5 ports, 2 of which are level 1, some stations may have been added later
# MAGIC     - hamiton 1 has 2 port
# MAGIC     - hamilotin 2 has 2 ports ( 1 was upgraded to level 2 at one point)
# MAGIC     - total of 4 ports
# MAGIC - **high street** == 9 ports, 8 level 2, 1 level 1
# MAGIC     - level 1 upgraded to level 2 in 2017
# MAGIC     - total of 8 ports
# MAGIC - **mpl** has 6 in our data set... did some come out???  1, 2, 3 have very little charging session in 2020, may have came out, long break between sessions - could have been broken-- may consider removing those 3... 
# MAGIC     - 4 5 6 installed in Sept 2014 -- > remove data prior to 2015, and remove station 1-3
# MAGIC     - total of 3 ports, 1 each
# MAGIC - **Rinconda** library yes 3 level 2, but also have 3 level 1s
# MAGIC     - very few level 1 sessions, remove level 1 charging sessions 
# MAGIC     - 3 ports total
# MAGIC - **Sherman** we have 19 in our data set, if assume 2 ports for each station we have we have 26 ( sherman, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 17)
# MAGIC       - all of these stations barely added in dec of 2020
# MAGIC - **ted thomas** matches at 8, 2 each
# MAGIC       ## also added late
# MAGIC - **webster**
# MAGIC     - we only have 6 total -- maybe some added later
# MAGIC 
# MAGIC array(['HIGH', 'BRYANT', 'HAMILTON', 'MPL', 'RINCONADA', 'WEBSTER', 'TED',
# MAGIC        'CAMBRIDGE', 'SHERMAN'], dtype=object)
# MAGIC 
# MAGIC Do not have charging sessions for museaum<br>
# MAGIC - Sherman parking garage may be missing charging stations -- do not have 33 <br>
# MAGIC - Hamilton, we only show 5 ports (and 2 are level 1) -- could be missing stations or some added later

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Data

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
import folium
import holidays


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
palo_alto = spark.read.option("header", True) \
                      .csv(f"/mnt/{mount_name}/data/Palo_Alto_EVChargingStationUsage_FIXED.csv")

#convert to pandas dataframe
palo_alto = palo_alto.toPandas()

print("Shape:", palo_alto.shape)


palo_alto.head()

# COMMAND ----------

palo_alto[['Station Name', 'Start Date', 'Start Time Zone', 'Total Duration (hh:mm:ss)', 'Port Type','Port Number', 'Plug Type']]

# COMMAND ----------

# MAGIC %md
# MAGIC ## General EDA - Data understanding

# COMMAND ----------

# what columns do I have and what type are they?
palo_alto.info()

# need to convert data and times.

# COMMAND ----------

palo_alto.isnull().sum()

# most of these nulls are fine 
# transaction date theory is similar to end date

# EVSE ID not needed
# currency -- do i need to correct if we use pricing?
# ended by, not need for forecasting
# rest not needed for forecasting

# COMMAND ----------

palo_alto.nunique()
#47 stations
#3 timezones
# 2 port types, 2 port numbers and 2 plug types
# all in same city

# COMMAND ----------

#palo_alto[palo_alto['Transaction Date (Pacific Time)'].isna()]
# can fill in transaction dates if need but seems similar to end date

# COMMAND ----------

print(palo_alto['Start Time Zone' ].unique())
print(palo_alto['End Time Zone' ].unique())
print(palo_alto['Currency' ].unique())
# may need to standardize time zones
# PST is also during only those months , #utc seems like random times
# may need to standardize dollars to remove non USD currency

# COMMAND ----------

palo_alto[['Start Time Zone', 'End Time Zone']].drop_duplicates()

##should consider switching all to utc to avoid issues, could be adding on one additional hour?
#The switch from pst to pdt was during daylight savings end or start ,so is accurate as is for pst to pdt or pdt to pst

# COMMAND ----------

palo_alto.groupby(['Currency'])['Currency'].count()

# COMMAND ----------

palo_alto[(palo_alto['Currency'] == 'CAD') | (palo_alto['Currency'] == 'EUR') |  (palo_alto['Currency'] == 'MXN')].sort_values('Currency', axis = 0)

#CAD fee was $0 , so no need to worry
#EUR if want to use price have to convert 2 if pricing varies
#MXN if want to use price have to convert 2 if pricing varies

# COMMAND ----------

#pd.set_option('display.max_rows', None)

# COMMAND ----------

palo_alto[['Station Name', 'Port Type','Port Number' ]].groupby(['Station Name']).nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting DTypes

# COMMAND ----------

#convert to date time
palo_alto['Start Date'] = pd.to_datetime(palo_alto['Start Date'], format = '%m/%d/%Y %H:%M')
palo_alto['End Date'] = pd.to_datetime(palo_alto['End Date'], format = '%m/%d/%Y %H:%M')
palo_alto['Transaction Date (Pacific Time)'] = pd.to_datetime(palo_alto['Transaction Date (Pacific Time)'], format = '%m/%d/%Y %H:%M')

# convert duration columns to a standard measurement, and int object
# Convert to minutes.
palo_alto['Total Duration (minutes)'] = palo_alto['Total Duration (hh:mm:ss)'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]) + int(x[2])*(1/60))
palo_alto['Charging Time (minutes)'] = palo_alto['Charging Time (hh:mm:ss)'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]) + int(x[2])*(1/60))


# converting some columns to integers
palo_alto['Energy (kWh)'] = pd.to_numeric(palo_alto['Energy (kWh)'] )
palo_alto['Fee'] = pd.to_numeric(palo_alto['Fee'] )
#palo_alto['Port Number'] = pd.to_numeric(palo_alto['Port Number'] ) -- to concatenate later, easier if string
palo_alto['Latitude'] = pd.to_numeric(palo_alto['Latitude'] )
palo_alto['Longitude'] = pd.to_numeric(palo_alto['Longitude'] )
palo_alto['GHG Savings (kg)'] = pd.to_numeric(palo_alto['GHG Savings (kg)'] )
palo_alto['Gasoline Savings (gallons)'] = pd.to_numeric(palo_alto['Gasoline Savings (gallons)'] )

# COMMAND ----------

palo_alto.describe()

# can convert some of these to objects, EVSE ID, Plug in Event ID, Driver Postal Code, System S/N ... dont need these items

# fee has outliers
#gasoline saving, ghg saving and energy has outliers

# may need to drop some records -- the max 6873 minutes --> almost 5 days.... this is not normal, for a continous charge... 

# COMMAND ----------

# charging session longer than 1 day
print('Number of Charging Sessions, greater than 1 day: ', len(palo_alto[palo_alto['Total Duration (minutes)'] >= 1440]))
palo_alto[palo_alto['Total Duration (minutes)'] >= 1440]

# COMMAND ----------

palo_alto['diffsten'] = palo_alto['End Date'] - palo_alto['Start Date']
palo_alto['diffsten'] = palo_alto['diffsten']/np.timedelta64(1,'m')

palo_alto[round(palo_alto['diffsten']) > 1+ round(palo_alto['Total Duration (minutes)'])]
#have one row that has a start and end that is over 1 month, but the total duration does not match

### start and end time is wrong... does not match  total duration

# COMMAND ----------

##### Charging duration longer than total duration
print(len(palo_alto[palo_alto['Total Duration (minutes)'] < palo_alto['Charging Time (minutes)']]))
palo_alto[palo_alto['Total Duration (minutes)'] < palo_alto['Charging Time (minutes)']]

## these are odd, but do not impact 

# COMMAND ----------

#charging session where total duration < charging stations
print(len(palo_alto[palo_alto['Total Duration (minutes)'] < palo_alto['Charging Time (minutes)']]))
# seems like charging time and duration has been swapped
palo_alto[palo_alto['Total Duration (minutes)'] < palo_alto['Charging Time (minutes)']]

# COMMAND ----------

# MAGIC %md
# MAGIC ##Stations

# COMMAND ----------

sorted(palo_alto['Station Name'].unique())

# 3 as webster
#ted Thomaps 4 
#sherman 17
# Rinconada Lib 3
# MPL 6
#High 4
# hamilton 2
# cambridge #4
# bryant 6

# COMMAND ----------

# unique stations and ports, etc.
stations = palo_alto [['Station Name','Port Type', 'Port Number', 'Plug Type',  'Address 1', 'City', 'State/Province', 'Postal Code', 'Country', 'Latitude', 'Longitude']]

stations.drop_duplicates().reset_index()


# each station has 2 ports
# Bryant # 1 and Bryant#1 appears to have moved at some point based on lat and long or gps system is inaccurate at the 3rd to 4th decimal

# the mpl stations appear to only have 1 port, unless only port 1 was ever used.. unlikely

#Rinconads lib 1 moved at one point, different addresses

#### can clean up addresses if we need to use addresses and latitude and longitude
# the addresses that differe are actually in same locations
 ###i.e. webster #3, 533 cowper and 520 webster is to same building
 ### or in same gerneal vicinity, likely issue with gps coordinate system or satellite issues

# COMMAND ----------

# stations_geo = palo_alto[['Station Name', 'Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
# stations_geo

# COMMAND ----------

# stations_geo = palo_alto[['Station Name', 'Latitude', 'Longitude']].drop_duplicates()

# lats = list(palo_alto['Latitude'])
# lons = list(palo_alto['Longitude'])
# loc_name = list(palo_alto['Station Name'])
# locations = zip(lats, lons, loc_name)

# color_dict = {'PALO ALTO CA / BRYANT # 1': "red",
#               'PALO ALTO CA / BRYANT #1': "red",
#               'PALO ALTO CA / BRYANT #2': "red",
#               'PALO ALTO CA / BRYANT #3': "red",
#               'PALO ALTO CA / BRYANT #4': "red",
#               'PALO ALTO CA / BRYANT #5': "red",
#               'PALO ALTO CA / BRYANT #6': "red",
#               'PALO ALTO CA / CAMBRIDGE #1': 'green',
#               'PALO ALTO CA / CAMBRIDGE #2': 'green',
#               'PALO ALTO CA / CAMBRIDGE #3': 'green',
#               'PALO ALTO CA / CAMBRIDGE #4': 'green',
#               'PALO ALTO CA / CAMBRIDGE #5': 'green',
#               'PALO ALTO CA / HAMILTON #1': 'blue', 
#               'PALO ALTO CA / HAMILTON #2': 'blue',
#               'PALO ALTO CA / HIGH #1': 'darkblue',
#               'PALO ALTO CA / HIGH #2': 'darkblue',
#               'PALO ALTO CA / HIGH #3': 'darkblue',
#               'PALO ALTO CA / HIGH #4': 'darkblue',
#               'PALO ALTO CA / MPL #1': 'black', 
#               'PALO ALTO CA / MPL #2': 'black', 
#               'PALO ALTO CA / MPL #3': 'black', 
#               'PALO ALTO CA / MPL #4': 'black', 
#               'PALO ALTO CA / MPL #5': 'black', 
#               'PALO ALTO CA / MPL #6': 'black', 
#               'PALO ALTO CA / RINCONADA LIB 1': 'purple',
#               'PALO ALTO CA / RINCONADA LIB 2': 'purple',
#               'PALO ALTO CA / RINCONADA LIB 3': 'purple',
#               'PALO ALTO CA / SHERMAN 1': 'orange', 
#               'PALO ALTO CA / SHERMAN 2': 'orange', 
#               'PALO ALTO CA / SHERMAN 3': 'orange', 
#               'PALO ALTO CA / SHERMAN 4': 'orange', 
#               'PALO ALTO CA / SHERMAN 5': 'orange', 
#               'PALO ALTO CA / SHERMAN 6': 'orange', 
#               'PALO ALTO CA / SHERMAN 7': 'orange', 
#               'PALO ALTO CA / SHERMAN 8': 'orange', 
#               'PALO ALTO CA / SHERMAN 9': 'orange', 
#               'PALO ALTO CA / SHERMAN 11': 'orange', 
#               'PALO ALTO CA / SHERMAN 14': 'orange', 
#               'PALO ALTO CA / SHERMAN 15': 'orange', 
#               'PALO ALTO CA / SHERMAN 17': 'orange', 
#               'PALO ALTO CA / TED THOMPSON #1': 'cadetblue',    
#               'PALO ALTO CA / TED THOMPSON #2': 'cadetblue', 
#               'PALO ALTO CA / TED THOMPSON #3': 'cadetblue', 
#               'PALO ALTO CA / TED THOMPSON #4': 'cadetblue', 
#               'PALO ALTO CA / WEBSTER #1': 'pink',          
#              'PALO ALTO CA / WEBSTER #2': 'pink',        
#              'PALO ALTO CA / WEBSTER #3': 'pink'}

#m = folium.Map(location=[37.468319, -122.143936], tiles="Stamen Terrain", zoom_start=5)
#for lat, lon, loc_name in locations:
#    folium.Marker([lat, lon], icon=folium.Icon(color=color_dict[loc_name], icon="bolt", prefix='fa-solid fa')).add_to(m)
#m

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA: Charging Sessions Data

# COMMAND ----------

palo_alto

# COMMAND ----------

palo_alto['Hour'] = palo_alto['Start Date'].dt.hour

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].boxplot(palo_alto['Total Duration (minutes)']);
ax[0].set_title('Total Duration (minutes)');
ax[1].boxplot(palo_alto['Charging Time (minutes)']);
ax[1].set_title('Charging Time (minutes)');

# COMMAND ----------

#x_var = 'IsWeekend' # duations slightly less at the end of the week
#x_var = 'DayofWeek'  # charging duration is a little less duration end of week
x_var = 'Hour'   ## charging durations/total durations are longer when started in evening

fig, ax = plt.subplots(1, 2, figsize = (15,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.boxplot(ax = ax[0], x = x_var, y= 'Total Duration (minutes)', data = palo_alto);
ax[0].set_title('Total Duration')
ax[0].set(ylim=(0, 1000))

sns.boxplot(ax = ax[1], x = x_var, y= 'Charging Time (minutes)', data = palo_alto);
ax[1].set_title('Charging Time')
ax[1].set(ylim=(0, 1001))

plt.show()

# COMMAND ----------

palo_alto[['Total Duration (minutes)','Charging Time (minutes)']].describe()

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
limit = 1440
ax[0].hist(palo_alto[palo_alto['Total Duration (minutes)'] < limit]['Total Duration (minutes)'], bins = 100);
ax[1].hist(palo_alto[palo_alto['Charging Time (minutes)'] < limit]['Charging Time (minutes)'], bins = 100);

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

palo_alto.plot.scatter(x='Energy (kWh)', y='Fee', c='DarkGreen')

# COMMAND ----------

overstay = palo_alto['Total Duration (minutes)'] - palo_alto['Charging Time (minutes)']
overstay.describe()
# should remove station whos total duration 

# COMMAND ----------

palo_alto.columns

# COMMAND ----------

# copy for EDA only
df = palo_alto.copy()
df = df.replace('PALO ALTO CA / BRYANT # 1', 'PALO ALTO CA / BRYANT #1')
df.drop(df[df['Start Date'] < '2015-01-01'].index, inplace = True)
df.drop(df[(df['Station Name'].str.contains('RINCONADA')) & (df['Plug Type'] == 'NEMA 5-20R')].index, inplace = True)
df.drop(df[(df['Station Name'].str.contains('MPL #1')) | (df['Station Name'].str.contains('MPL #2')) | (df['Station Name'].str.contains('MPL #3'))].index, inplace = True)

##### convert to UTC
# PDT +7h = UTC
# PST +8h = UTC
df.loc[df['Start Time Zone'] == 'PDT', 'Start Date'] +=  timedelta(hours = 7)
df.loc[df['Start Time Zone'] == 'PST', 'Start Date'] +=  timedelta(hours = 8)
df.loc[df['End Time Zone'] == 'PDT', 'End Date'] +=  timedelta(hours = 7)
df.loc[df['End Time Zone'] == 'PST', 'End Date'] +=  timedelta(hours = 8)
df.drop(df[df['Total Duration (minutes)'] > 1440].index, inplace = True)

df['Start Date'] = df['Start Date'].dt.tz_localize('UTC')
df['Start Date'] = df['Start Date'].dt.tz_convert('US/Pacific')
df['End Date'] = df['End Date'].dt.tz_localize('UTC')
df['End Date'] = df['End Date'].dt.tz_convert('US/Pacific')


# add date time parts
df['Date'] =  df['Start Date'].dt.date
df['Month'] = df['Start Date'].dt.month
df['Year'] = df['Start Date'].dt.year
df['Year-Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str)
df['DayofWeek'] = df['Start Date'].dt.weekday
df['IsWeekend'] = df['DayofWeek'] > 4
df['Hour'] = df['Start Date'].dt.hour

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

x_var = 'Hour'   ## charging durations/total durations are longer when started in evening
hue_var = 'IsWeekend'  ## duration tend to be higher when stat in the late evening when its a weekend. durations are shorter during day, 8 - 3pm
#hue_var = 'holiday'  # more variability in use during holidays

fig, ax = plt.subplots(1, 2, figsize = (20,10))
fig.subplots_adjust(hspace=0.5, wspace=0.25)

sns.lineplot(ax = ax[0], x = x_var, y= 'Total Duration (minutes)', hue = hue_var, data = df, ci='sd');
ax[0].set_title('Total Duration')
ax[0].set(ylim=(0, 1000))

sns.lineplot(ax = ax[1], x = x_var, y= 'Charging Time (minutes)',hue = hue_var, data = df, ci='sd');
ax[1].set_title('Charging Time')
ax[1].set(ylim=(0, 1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting to Time Series & Other related data prep

# COMMAND ----------

##### Data cleanup  ############

#### clean up issue with station name
pa_clean = palo_alto.replace('PALO ALTO CA / BRYANT # 1', 'PALO ALTO CA / BRYANT #1')
print(pa_clean.shape)

#### drop records prior to 2015 
print('Number of session before 2015: ', len(pa_clean[pa_clean['Start Date'] < '2015-01-01']))
print(pa_clean.shape)
pa_clean.drop(pa_clean[pa_clean['Start Date'] < '2015-01-01'].index, inplace = True)
print('Dropped session before 2015, ', pa_clean.shape)


#### drop sessions that are off
# rinconada level 1 sessions
pa_clean.drop(pa_clean[(pa_clean['Station Name'].str.contains('RINCONADA')) & (pa_clean['Plug Type'] == 'NEMA 5-20R')].index, inplace = True)
# MLB #1 #2 #3 station removed
pa_clean.drop(pa_clean[(pa_clean['Station Name'].str.contains('MPL #1')) | (pa_clean['Station Name'].str.contains('MPL #2')) | (pa_clean['Station Name'].str.contains('MPL #3'))].index, inplace = True)
print('Dropped session identified in notes, ', pa_clean.shape)


##### convert to UTC
# PDT +7h = UTC
# PST +8h = UTC
pa_clean.loc[pa_clean['Start Time Zone'] == 'PDT', 'Start Date'] +=  timedelta(hours = 7)
pa_clean.loc[pa_clean['Start Time Zone'] == 'PST', 'Start Date'] +=  timedelta(hours = 8)
pa_clean.loc[pa_clean['End Time Zone'] == 'PDT', 'End Date'] +=  timedelta(hours = 7)
pa_clean.loc[pa_clean['End Time Zone'] == 'PST', 'End Date'] +=  timedelta(hours = 8)

# timezone changed to UTC
pa_clean['Start Time Zone'] = 'UTC'
pa_clean['End Time Zone'] = 'UTC'

### round to 10 minutes
pa_clean['Start Date'] = pa_clean['Start Date'].dt.round(freq = '10T')
pa_clean['End Date'] = pa_clean['End Date'].dt.round(freq = '10T')

print(pa_clean.shape)
#(259415, 35)

# COMMAND ----------

print(pa_clean.shape)

# COMMAND ----------

###### drop oddly long charging station times ###### 

print('Records with charging sessions > 1 day:' ,len(pa_clean[pa_clean['Total Duration (minutes)'] > 1440]))
#pa_clean[pa_clean['Total Duration (minutes)'] > 1440]
pa_clean.drop(pa_clean[pa_clean['Total Duration (minutes)'] > 1440].index, inplace = True)
print(pa_clean.shape)

# COMMAND ----------

##### Total Duration and Charging Time, where charging time > Total Duration ######
## swap charging and total duration where total duration is longer than charging time
#df of true false to identify records
print('Num Records Total Duration < Charging Time: ', len(pa_clean[pa_clean['Total Duration (minutes)'] < pa_clean['Charging Time (minutes)']]))

idx = pa_clean['Total Duration (minutes)'] < pa_clean['Charging Time (minutes)']

# replace original duration
pa_clean.loc[idx,['Total Duration (hh:mm:ss)','Charging Time (hh:mm:ss)']] = pa_clean.loc[idx,['Charging Time (hh:mm:ss)','Total Duration (hh:mm:ss)']].values
# replace calculated duration
pa_clean.loc[idx,['Total Duration (minutes)','Charging Time (minutes)']] = pa_clean.loc[idx,['Charging Time (minutes)','Total Duration (minutes)']].values

print('Num Records Total Duration < Charging Time: ', len(pa_clean[pa_clean['Total Duration (minutes)'] < pa_clean['Charging Time (minutes)']]))

# COMMAND ----------

overstay = pa_clean['Total Duration (minutes)'] - pa_clean['Charging Time (minutes)']
overstay.describe()

# COMMAND ----------

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[0].boxplot(pa_clean['Total Duration (minutes)']);
ax[0].set_title('Total Duration (minutes)');
ax[1].boxplot(pa_clean['Charging Time (minutes)']);
ax[1].set_title('Charging Time (minutes)');

# COMMAND ----------

stations_locname = ['BRYANT', 'HAMILTON', 'CAMBRIDGE', 'HIGH', 'MPL', 'RINCONADA LIB', 'SHERMAN', 'TED THOMPSON', 'WEBSTER']


fig, ax = plt.subplots(len(stations_locname), 2, figsize = (15, 30))
for x, n in enumerate(stations_locname):
    ax[x,0].boxplot(pa_clean[pa_clean['Station Name'].str.contains(str(n))]['Total Duration (minutes)']);
    ax[x,0].set_title(str(n) +' Total Duration (minutes)');
    ax[x,1].boxplot(pa_clean[pa_clean['Station Name'].str.contains(str(n))]['Charging Time (minutes)']);
    ax[x,1].set_title(str(n)+' Charging Time (minutes)');


# COMMAND ----------

pa_clean[['Station Name', 'Port Type','Port Number' ]].groupby(['Station Name']).nunique()

# COMMAND ----------

pa_clean['Port Type'].isnull().sum()

# COMMAND ----------

######### replacing nulls #########
# Define a dictionary with the unique plug types and their corresponding port types
plug_types = pa_clean[['Plug Type', 'Port Type']].drop_duplicates().dropna().set_index('Plug Type').to_dict()

# Replace cells where the Port Type is missing
# Get the rows with null port type values
null_ports = pa_clean[pa_clean['Port Type'].isnull()]

# Loop through each of them (there's only 9 so this shouldn't take a long time)
# Look up the correct plug type from the pre-defined dictionary
for row in null_ports.index:
    pa_clean.loc[row, 'Port Type'] = plug_types['Port Type'][null_ports.loc[row, 'Plug Type']]

# Check that the values have been replaced correctly --> expected output = 0
pa_clean['Port Type'].isnull().sum()

# COMMAND ----------

## CELL ADDED BY JENNY
## JUST TRYING TO UNDERSTAND WHAT THE CODE DOES A BIT MORE
#palo_altoTS = pa_clean[['Station Name', 
#                           'Start Date', 
#                           'End Date', 
#                           'Port Type', 'Port Number', 
#                           'Plug Type']].copy(deep = True)
#palo_altoTS['DateTime'] = list(zip(palo_altoTS['Start Date'],palo_altoTS['End Date']))
#palo_altoTS['DateTime'] = palo_altoTS['DateTime'].apply(lambda x: list(pd.date_range(x[0], x[1], freq = '10T')))
#palo_altoTS = palo_altoTS.explode('DateTime')
#palo_altoTS.head()

# COMMAND ----------

############### Convert to Time Series #########################

#### create copy to apply changes and only pull necessary columns
palo_altoTS = pa_clean[['Station Name', 
                           'Start Date', 
                           'Total Duration (minutes)', 
                           'Port Type', 'Port Number', 
                           'Plug Type']].copy(deep = True)


##### add column with time ranges for start and end  ##
# create copy
palo_altoTS['DateTime'] = list(zip(palo_altoTS['Start Date'],palo_altoTS['Total Duration (minutes)']))
# apply a function to get time series list, start date, number of 10 minutes periods in duration, and use freq of 10 minutes
palo_altoTS['DateTime'] = palo_altoTS['DateTime'].apply(lambda x: list(pd.date_range(x[0], periods = math.ceil(x[1]/10), freq = '10T')))


#### explode  ##
palo_altoTS = palo_altoTS.explode('DateTime')
print('Exploded Timeseries completed')

#drop start and end cols, no longer need
palo_altoTS = palo_altoTS.drop(['Start Date', 'Total Duration (minutes)'], axis = 1)


##### drop duplicates   ##
#remove charging sessions that may have stopped and start in same 10 minute interval --- no double counting, is there double counting if used different connector?
print('Before duplicate drop:', palo_altoTS.shape)
palo_altoTS = palo_altoTS.drop_duplicates()
palo_altoTS.reset_index(inplace= True, drop=True)
print('After duplicate drop:', palo_altoTS.shape)


##### Ports Occupied ##
# one for each charging session, by 10 minute interval
palo_altoTS.loc[:,'Ports Occupied'] = int(1)
print('Added Ports Occupied')
print('After Ports:', palo_altoTS.shape)

palo_altoTS_prev = palo_altoTS.copy()



# COMMAND ----------

#### Add Station location ###
palo_altoTS.loc[:,'Station Location'] = palo_altoTS['Station Name'].str.replace('PALO ALTO CA / ', '').str.replace('TED THOMPSON', 'TED_THOMPSON').str.replace('RINCONADA LIB','RINCONADA_LIB').str.rsplit(' ').str[0]

palo_altoTS.head()
print(palo_altoTS['Station Location'].unique())

# COMMAND ----------

#### Group by DateTime and station name , sum the port Occupied and concatenate all others
## note all station info columns must be object
palo_altoTS = palo_altoTS.groupby([ 'DateTime', 'Station Location'], as_index=False).agg(lambda x: x.sum() if x.dtype=='int64' else ';'.join(x))
print('After group:', palo_altoTS.shape)



# COMMAND ----------

palo_altoTS.head()

# COMMAND ----------

########## Fill in missing timestamps per station #########

###### function to fill in times 
def stationTS_func(df, dateTimeCol):
    '''fill in missing time stamps, assumes already in 10 minute intervals 
    must have a column with Station Location '''
    
    # station installed at different starts
    start_ts = df[str(dateTimeCol)].min()
    ### all end at same time
    end_ts = palo_altoTS[str('DateTime')].max()
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
    station_ts['Station Location'].fillna(str(df['Station Location'].unique()[0]), inplace = True)
    
    return station_ts



###### create start of time series data frame
# unique station locations
StationLocs = sorted(palo_altoTS['Station Location'].drop_duplicates())

# dataframe with first locations
print(StationLocs[0])
PA_TS = stationTS_func(palo_altoTS[palo_altoTS['Station Location'] == str(StationLocs[0])].reset_index(drop=True), 'DateTime')
print('Starter TS dataframe: ', PA_TS.shape)


###### append other locations
# for each station merge to time series, and then append to first record dataframe  --> join may be faster than merge, set datetime as index
for station in StationLocs[1:]:
    print(station)
    
    # create station df
    station_ = palo_altoTS[palo_altoTS['Station Location'] == str(station)].reset_index(drop=True)
    newTS = stationTS_func(station_, 'DateTime')
    
    # append to PA_TS
    PA_TS = pd.concat([PA_TS, newTS], axis = 0)
    #1.66 - 5.75 mins

## new timeseries
print('final dataframe', PA_TS.shape)

# fill in 0s for null ports occupied
PA_TS['Ports Occupied'] = PA_TS['Ports Occupied'].fillna(0)


# COMMAND ----------

# ##### station added at different times need to redo this, and create a start and end at the station location level


# ######## Fill in missing timestamps #########

# ##### create start and end time series 
# # get start and end of entire data set
# start_ts = pa_clean['Start Date'].min() - dt.timedelta(minutes = pa_clean.loc[0, 'Start Date'].minute % 10)
# end_ts = pa_clean['End Date'].max() - dt.timedelta(minutes = pa_clean.loc[0, 'Start Date'].minute % 10)

# # creating all 10 minute intervals in time frame
# all_times = pd.date_range(start = start_ts, end = end_ts, freq = '10min')

# # printing to confirm
# print(start_ts)
# print(end_ts)
# print(len(all_times))

# # Create df with only times, one column
# expanded_ts = pd.DataFrame(data = {'DateTime': all_times})



# ##### single dataframe 
# # create dataframe with first record
# PA_TS = palo_altoTS[palo_altoTS['DateTime'] == palo_altoTS['DateTime'].min()]


# # for each station merge to time series, and then append to first record dataframe  --> join may be faster than merge, set datetime as index
# # looped through pa_clean bc it has less records
# for station in sorted(pa_clean['Station Name'].drop_duplicates()):
#     print(station)
#     # merge to expanded _ts
#     station_ = palo_altoTS[palo_altoTS['Station Name'] == str(station)].reset_index(drop=True)
#     station_ts =  pd.merge(left = expanded_ts, right=station_, how='left', left_on='DateTime', right_on='DateTime')
#     #replace empty station name col
#     station_ts['Station Name'] = str(station)
    
#     # append to PA_TS
#     PA_TS = pd.concat([PA_TS, station_ts], axis = 0)
#     #1.66 - 5.75 mins

# print('new dataframe', PA_TS.shape)

# # fill in 0s for null ports occupied
# PA_TS['Ports Occupied'] = PA_TS['Ports Occupied'].fillna(0)


# COMMAND ----------

PA_TS.shape

# before using duration (22833435, 5)

# COMMAND ----------

PA_TS.head()

# COMMAND ----------

PA_TS['Station Location'].unique()

# COMMAND ----------

######## add total ports per station area  #######


#### Create Static Station Info 
header = ['Station Location', 'Total Ports']
stations = [
            ['BRYANT', 12],
            ['CAMBRIDGE', 10],
            ['HAMILTON', 4],
            ['HIGH', 8],
            ['MPL', 3],
            ['RINCONADA_LIB', 3],
            ['SHERMAN', 26],
            ['TED_THOMPSON', 8],
            ['WEBSTER', 6]
]


# parking limitations
# address info


site_names = [station_info[0] for station_info in stations]
plug_nums = [station_info[1] for station_info in stations]



static_station_data = pd.DataFrame(data = {header[0]: site_names,
                                           header[1]: plug_nums,})


static_station_data

#### merge with PA_TS
print(PA_TS.shape)
PA_TS =  pd.merge(left = PA_TS, right=static_station_data, how='left', left_on='Station Location', right_on='Station Location')
PA_TS.head()
print('Added Total Ports: ', PA_TS.shape)


###### calculate available ports #######
PA_TS.loc[:,'Ports Available'] = PA_TS['Total Ports'] - PA_TS['Ports Occupied']
print('Added Available Ports:', PA_TS.shape)


#### calculate percent available ####
PA_TS.loc[:,'Ports Perc Available'] = PA_TS['Ports Available'] / PA_TS['Total Ports']
print('Added Perc Available Ports:', PA_TS.shape)


#### calculate percent occupied ######
PA_TS.loc[:,'Ports Perc Occupied'] = PA_TS['Ports Occupied'] / PA_TS['Total Ports']
print('Added Perc Occupied Ports:', PA_TS.shape)

# COMMAND ----------

PA_TS.head()

# COMMAND ----------

######   convert back to local time ######## 
PA_TS['datetime_utc'] = PA_TS['DateTime'].dt.tz_localize('UTC')#.dt.tz_convert('Asia/Kolkata')
PA_TS['datetime_pac'] = PA_TS['datetime_utc'].dt.tz_convert('US/Pacific')
#Pacific will show both daylight savings and standard time -- will be localized

PA_TS.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to S3 Bucket

# COMMAND ----------

# create spark data frame
PaloAlto_ts = spark.createDataFrame(PA_TS) 

cols = PaloAlto_ts.columns
for col in cols:
    PaloAlto_ts = PaloAlto_ts.withColumnRenamed(col, col.replace(" ", "_"))
display(PaloAlto_ts)

## Write to AWS S3
(PaloAlto_ts
     .repartition(1)
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/PaloAlto_ts"))

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA: Time Series Records

# COMMAND ----------

####### Read from s3 #############

PaloAlto_ts = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_ts")
PaloAlto_ts = PaloAlto_ts.toPandas()

PaloAlto_ts.head()

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
PaloAlto_ts['Month'] = PaloAlto_ts['datetime_pac'].dt.month
PaloAlto_ts['Year'] = PaloAlto_ts['datetime_pac'].dt.year
PaloAlto_ts['Date'] = PaloAlto_ts['datetime_pac'].dt.date
PaloAlto_ts['Year-Month'] = PaloAlto_ts['Year'].astype(str) + '-' + PaloAlto_ts['Month'].astype(str)
PaloAlto_ts['DayofWeek'] = PaloAlto_ts['datetime_pac'].dt.weekday
PaloAlto_ts['IsWeekend'] = PaloAlto_ts['DayofWeek'] > 4
PaloAlto_ts['Hour'] = PaloAlto_ts['datetime_pac'].dt.hour


## Add Holidays
PaloAlto_ts = PaloAlto_ts.merge(us_holidays, how = 'left', 
                               left_on = 'Date', 
                               right_on = 'date')\
                        .drop(columns = ['holiday_name', 'date'])\
                        .fillna(0)

## View
PaloAlto_ts.head()

# COMMAND ----------

temp = PaloAlto_ts#[['Station_Location', 'IsWeekend', 'Hour', 'Ports_Perc_Available']]#.groupby(['Station_Location', 'IsWeekend', 'Hour'], as_index = False).mean()
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())

fig, ax = plt.subplots( figsize = (15,10))


sns.lineplot( x = temp['Hour'], y= temp['Ports_Perc_Available'], hue = temp['IsWeekend'], ci='sd');
#         ax[int(i/cols), i%cols].set_title(siteName[i])
#         ax[int(i/cols), i%cols].set_ylim([0, 2]) 

plt.show()

# COMMAND ----------

temp = PaloAlto_ts#[['Station_Location', 'IsWeekend', 'Hour', 'Ports_Perc_Available']]#.groupby(['Station_Location', 'IsWeekend', 'Hour'], as_index = False).mean()
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station_Location'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Hour'], y= temp_filter['Ports_Perc_Available'], hue = temp_filter['IsWeekend'], ci='sd');
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, 2]) 

fig.tight_layout()



# COMMAND ----------

siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 9
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (30,30))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i]]
        sns.lineplot(ax = ax[i], 
                     x = 'Hour', 
                     y= 'Ports_Occupied',  
                     hue = 'DayofWeek',   
                     palette="deep",
                     data = temp_filter);
        
        ax[i].set_title(siteName[i])
        ax[i].set_ylim([0, PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i] ]['Total_Ports'].max()])  

fig

# COMMAND ----------

temp = PaloAlto_ts[['Station_Location', 'Hour', 'holiday', 'Ports_Occupied']]
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station Location'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Hour'], y= temp_filter['Ports_Occupied'], hue = temp_filter['holiday']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        #ax[int(i/cols), i%cols].set_ylim([0, 1])   

fig.tight_layout()

# COMMAND ----------

temp = PaloAlto_ts#[['Station_Location', 'Hour', 'holiday', 'Ports_Occupied']]
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station_Location'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Hour'], y= temp_filter['Ports_Perc_Available'], hue = temp_filter['holiday']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        #ax[int(i/cols), i%cols].set_ylim([0, 1])   

fig.tight_layout()

# COMMAND ----------

temp = PaloAlto_ts[['Station_Location', 'Month','DayofWeek', 'Ports_Occupied', 'Total_Ports']]
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station_Location'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Month'], y= temp_filter['Ports_Occupied'], hue = temp_filter['DayofWeek']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station_Location'] == siteName[i] ]['Total_Ports'].max()])    

fig.tight_layout()

# COMMAND ----------

temp = PaloAlto_ts[['Station_Location', 'Month','IsWeekend', 'Ports_Occupied', 'Total_Ports']]
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station_Location'] == siteName[i]]
        sns.boxplot(ax = ax[int(i/cols), i%cols], x = temp_filter['Month'], y= temp_filter['Ports_Occupied'], hue = temp_filter['IsWeekend']);
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station_Location'] == siteName[i] ]['Total_Ports'].max()+1])    

fig.tight_layout()

# COMMAND ----------

temp = PaloAlto_ts[['Station_Location', 'DayofWeek', 'holiday', 'Ports_Occupied', 'Total_Ports']]
#temp.head()



fig, ax = plt.subplots(figsize = (15,10))
sns.lineplot( x = temp['DayofWeek'], y= temp['Ports_Occupied'], hue = temp['holiday'], ci='sd');
# ax[int(i/cols), i%cols].set_title(siteName[i])
# ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station_Location'] == siteName[i] ]['Total_Ports'].max()]) 

fig.tight_layout()

# COMMAND ----------

temp = PaloAlto_ts[['Station_Location', 'DayofWeek', 'holiday', 'Ports_Occupied', 'Total_Ports']]
#temp.head()


# #line charts like jennys
siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = temp[temp['Station_Location'] == siteName[i]]
        sns.lineplot(ax = ax[int(i/cols), i%cols], x = temp_filter['DayofWeek'], y= temp_filter['Ports_Occupied'], hue = temp_filter['holiday'], ci='sd');
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, temp[temp['Station_Location'] == siteName[i] ]['Total_Ports'].max()]) 

fig.tight_layout()


# COMMAND ----------

siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 5
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i]]
        sns.scatterplot(ax = ax[int(i/cols), i%cols], 
                        x = 'Date', 
                        y= 'Ports_Occupied',  
                        palette="deep",
                       linewidth=0,
                       data = temp_filter);
        
        ax[int(i/cols), i%cols].set_title(siteName[i])
        ax[int(i/cols), i%cols].set_ylim([0, PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i] ]['Total_Ports'].max()])  

fig.tight_layout()

# COMMAND ----------

siteName = sorted(weekdayhour['Station Location'].unique())
rows = 9
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i, n in enumerate(siteName):
    temp_filter = weekdayhour[(weekdayhour['Station Location'] == str(n)) & (weekdayhour['datetime_pac'] < '2021-1-1')]
    sns.lineplot(ax = ax[int(i)], x = temp_filter['datetime_pac'], y= temp_filter['Ports Perc Available'], hue = temp_filter['IsWeekend']);
    ax[int(i)].set_title(str(n))

fig.tight_layout()


### can see that there is increased utilization in the later years for all stations
### would like to create a view of this that is more user friendly and possibly with a play feature

# COMMAND ----------

siteName = sorted(weekdayhour['Station Location'].unique())
rows = 9
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i, n in enumerate(siteName):
    temp_filter = weekdayhour[(weekdayhour['Station Location'] == str(n)) & (weekdayhour['Ports Perc Available'] < .5)]
    sns.lineplot(ax = ax[int(i)], x = temp_filter['datetime_pac'], y= temp_filter['Ports Perc Available'], hue = temp_filter['IsWeekend']);
    ax[int(i)].set_title(str(n))

fig.tight_layout()

# COMMAND ----------

siteName = sorted(PaloAlto_ts['Station_Location'].unique())
rows = 9
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (60,30))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i]]
        sns.boxplot(ax = ax[i], 
                        x = 'Year-Month', 
                        y= 'Ports_Occupied',  
                        palette="deep",
                       data = temp_filter);
        
        ax[i].set_title(siteName[i])
        ax[i].set_ylim([0, PaloAlto_ts[PaloAlto_ts['Station_Location'] == siteName[i] ]['Total_Ports'].max()])  

fig

# COMMAND ----------

print(PaloAlto_ts.shape)
fully_occupied = PaloAlto_ts[PaloAlto_ts['Ports_Occupied'] == PaloAlto_ts['Total_Ports']]
print(fully_occupied.shape)

print('Percent Palo_Alto records are fully occupied: {:.2%}'.format(fully_occupied.shape[0]/PaloAlto_ts.shape[0]))

# COMMAND ----------

fig, ax = plt.subplots( figsize = (10,10))
sns.histplot(data=fully_occupied, x="Station_Location")
plt.xticks(rotation=70)


fig.tight_layout()

# COMMAND ----------

#### get percent of fully occupied records

names = sorted(fully_occupied['Station_Location'].unique())

percent_fullyOccupied = list()
percent_Weekend = list()
percent_holiday=list()

for name in names:
    percent_fullyOccupied.append(
        str(
            round(100*(
                fully_occupied[fully_occupied['Station_Location'] == name].shape[0] / PaloAlto_ts[PaloAlto_ts['Station_Location'] == name].shape[0])
        , 2))
        + '%')
    

for name in names:
    percent_Weekend.append(
        str(
            round(100*(
                fully_occupied[(fully_occupied['Station_Location'] == name) & (fully_occupied['IsWeekend'] == 1)].shape[0] / PaloAlto_ts[PaloAlto_ts['Station_Location'] == name].shape[0])
        , 2))
        + '%')

    
for name in names:
    percent_holiday.append(
        str(
            round(100*(
                fully_occupied[(fully_occupied['Station_Location'] == name) & (fully_occupied['holiday'] == 1)].shape[0] / PaloAlto_ts[PaloAlto_ts['Station_Location'] == name].shape[0])
        , 2))
        + '%')


df = pd.DataFrame()
data = {'Station':names, 'Percent Fully Occupied':percent_fullyOccupied, 'Percent Fully Occupied & Weekend':percent_Weekend, 'Percent Fully Occupied & Holiday': percent_holiday}

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

test = fully_occupied[['Station_Location', 'Year', 'Station_Name']].groupby(['Station_Location','Year']).count().rename(columns={'Station_Name':'Count'}).reset_index()

# Set your custom color palette
colors = ["#7BD159", "#59B7D1", "#AF59D1", "#D17359", "#F8E67C", "#378B49", "#7C8EF8", "#F87CCC"]
sns.set_palette(sns.color_palette(colors))



x_var = 'Year'  
hue_var = 'Station_Location'  
y_var = 'Count'


rows = 2
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (15,15))
fig.subplots_adjust(hspace=0.18, wspace=0.15)

sns.lineplot( ax = ax[0], x = x_var, y= y_var, data = test, ci='sd', linewidth = 2);
ax[0].set_title('Count of Records with Fully Occupied Charging Ports', fontsize = 16);
ax[0].set( ylabel = 'Count of Records', xlabel = 'Year', ylim= (0, 18000))
ax[0].yaxis.set_ticks(np.arange(0, 18000, 2500))


sns.lineplot( ax = ax[1], x = x_var, y= y_var, hue = hue_var, data = test, ci='sd', linewidth = 2);
ax[1].set_title('Count of Records with Fully Occupied Charging Ports By Charging Station Location', fontsize = 16);
ax[1].legend(loc="upper right", frameon=True, title = 'Charging Station Location')
ax[1].set( ylabel = 'Count of Records', xlabel = 'Year', ylim= (0, 18000))
ax[1].yaxis.set_ticks(np.arange(0, 18000, 2500))

#fig.suptitle('Count of Records with Fully Occupied Charging Ports', fontsize=18)

plt.show()

# COMMAND ----------

siteName = sorted(fully_occupied['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = fully_occupied[fully_occupied['Station_Location'] == siteName[i]]
        sns.histplot(ax = ax[int(i/cols), i%cols], x = temp_filter['DayofWeek']);
        ax[int(i/cols), i%cols].set_title(siteName[i])

fig.tight_layout()


# COMMAND ----------

siteName = sorted(fully_occupied['Station_Location'].unique())
rows = 1
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot(x = 'DayofWeek', data=fully_occupied);
#ax[int(i/cols), i%cols].set_title(siteName[i])

fig.tight_layout()

# COMMAND ----------

siteName = sorted(fully_occupied['Station_Location'].unique())
rows = 1
cols = 1

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot(x = 'DayofWeek', hue='holiday',multiple='stack', discrete=True, data=fully_occupied);
#ax[int(i/cols), i%cols].set_title(siteName[i])

fig.tight_layout()

# COMMAND ----------

rows = 1
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot( ax = ax[0], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=fully_occupied);
ax[0].set_title('Total Records Fully Occupied by Hour and Part of Week', fontsize = 16);
ax[0].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
ax[0].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 11000))
ax[0].yaxis.set_ticks(np.arange(0, 11000, 2000))


sns.histplot( ax = ax[1], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=fully_occupied);
ax[1].set_title('Total Records Fully Occupied by Hour and Holiday', fontsize = 16);
ax[1].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
ax[1].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 11000))
ax[1].yaxis.set_ticks(np.arange(0, 11000, 2000))



fig.tight_layout()

# COMMAND ----------

df = PaloAlto_ts[PaloAlto_ts['Ports_Occupied'] != 0]

rows = 1
cols = 2

fig, ax = plt.subplots(rows, cols, figsize = (15,10))

sns.histplot( ax = ax[0], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=df);
ax[0].set_title('Records with a minimum of 1 Port in Use \nby Hour and Part of Week', fontsize = 16);
ax[0].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
ax[0].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 100000))
#ax[0].yaxis.set_ticks(np.arange(0, 11000, 2000))


sns.histplot( ax = ax[1], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=df);
ax[1].set_title('Records with a minimum of 1 Port in Use \nby Hour and Holiday', fontsize = 16);
ax[1].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
ax[1].set( ylabel = 'Count of Records', xlabel = 'Hour', ylim= (0, 100000))
#ax[1].yaxis.set_ticks(np.arange(0, 11000, 2000))



fig.tight_layout()

# COMMAND ----------

### minimum of 1 port occupied stations by weekend and hour
df = PaloAlto_ts[PaloAlto_ts['Ports_Occupied'] != 0]
sitename = sorted(df['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = df[df['Station_Location'] == siteName[i]]
#         sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=temp_filter);
#         ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Part of Week for ' + str(siteName[i]), fontsize = 16);
#         ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
#         ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
#         ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


        sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=temp_filter);
        ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Holiday', fontsize = 16);
        ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
        ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
        ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


fig.tight_layout()

# COMMAND ----------

### minimum of 1 port occupied stations by weekend and hour
df = fully_occupied
sitename = sorted(df['Station_Location'].unique())
rows = 3
cols = 3

fig, ax = plt.subplots(rows, cols, figsize = (40,20))
for i in range(rows * cols):
    if i < len(siteName):
        temp_filter = df[df['Station_Location'] == siteName[i]]
        sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='IsWeekend',multiple='stack', discrete=True, data=temp_filter);
        ax[int(i/cols), i%cols].set_title('Fully Occupied Charging Stations by\n Hour and Part of Week for ' + str(siteName[i]), fontsize = 16);
        ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Weekend', 'Weekday'], title = 'Part of Week')
        ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
        ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


#         sns.histplot( ax = ax[int(i/cols), i%cols], x = 'Hour', hue='holiday',multiple='stack', discrete=True, data=temp_filter);
#         ax[int(i/cols), i%cols].set_title('Records with a minimum of 1 Port in Use by\n Hour and Holiday', fontsize = 16);
#         ax[int(i/cols), i%cols].legend(loc = 'upper right', labels = ['Yes', 'No'], title = 'Holiday?')
#         ax[int(i/cols), i%cols].set( ylabel = 'Count of Records', xlabel = 'Hour')
#         ax[int(i/cols), i%cols].xaxis.set_ticks(np.arange(0, 24, 1))


fig.tight_layout()