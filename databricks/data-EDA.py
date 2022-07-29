# Databricks notebook source
# MAGIC %md
# MAGIC ##Data Sources, and references for understanding datasets
# MAGIC 
# MAGIC * [TomTom Data](https://developer.tomtom.com/extended-search-api/documentation/extended-search-api/ev-charging-stations-availability)
# MAGIC     * [Charger Connector Types](https://developer.tomtom.com/search-api/documentation/product-information/supported-connector-types) 
# MAGIC * NREL
# MAGIC     * [Data Fields of Alternative Fuel Stations Download](https://afdc.energy.gov/data_download/alt_fuel_stations_format)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Cloud Storage Setup

# COMMAND ----------

pip install markupsafe==2.0.1

# COMMAND ----------

## Imports
import json
import time
import urllib
import requests
import pandas as pd
import seaborn as sns
import altair as alt
from vega_datasets import data
from pandas.io.json import json_normalize
import datetime as dt
from datetime import timedelta
import plotly.graph_objects as go
import pytz
from fp.fp import FreeProxy
import pyspark.sql.functions as F
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
from pyspark.sql.types import *
import folium

## Settings
warnings.filterwarnings("ignore")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
sns.set(rc={'figure.figsize':(16,9)})

blob_container = "w210" # The name of your container created in https://portal.azure.com
storage_account = "w210" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w210-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "azure-storage-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

## AWS
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
# MAGIC ## Functions

# COMMAND ----------

# DBTITLE 1,Functions
def plot_line_graphs(pdf, num_cols=4):
  
  ## Convert to pandas if required
    if type(pdf) != type(pd.DataFrame()):
        print("Converting to pandas.")
    try:
        pdf = pdf.toPandas()
    except Exception as e:
        return e
  
    ## Retrieve only the numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_pdf = pdf.select_dtypes(include=numerics)

    ## Plotting setup
    fig_i, col_i, row_i  = 0, 0, 0
    fig_count = len(num_pdf.columns)
    col_count = num_cols
    row_count = math.ceil(fig_count / col_count)

    ## Plot distributions
    fig, axx = plt.subplots(row_count, col_count, figsize=(40,16))
    for col in num_pdf.columns:
        sns.lineplot(num_pdf[col], ax=axx[row_i][col_i])
        fig_i += 1
        col_i = fig_i % col_count
        if col_i == 0:
            row_i = row_i + 1
            
    plt.show()

def plot_map_points(lat, lon):
    
    # determine range to print based on min, max lat and lon of the data
    margin = 2 # buffer to add to the range
    lat_min = min(lat) - margin
    lat_max = max(lat) + margin
    lon_min = min(lon) - margin
    lon_max = max(lon) + margin

    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
                llcrnrlat=lat_min,
                urcrnrlon=lon_max,
                urcrnrlat=lat_max,
                lat_0=(lat_max - lat_min)/2,
                lon_0=(lon_max-lon_min)/2,
                projection='tmerc',
                resolution = 'h',
                area_thresh=10000.,
                )
    m.drawcoastlines()
    m.fillcontinents(color = 'white',lake_color='#46bcec')
    m.drawcountries()
    m.drawstates()
    m.drawcounties()
    m.drawmapboundary(fill_color='#46bcec')
    
    # convert lat and lon to map projection coordinates
    lons, lats = m(lon, lat)
    # plot points as red dots
    m.scatter(lons, lats, marker = 'o', color='r', zorder=5)
    plt.show()
    
def convert_datetime_timezone(datet, tz1, tz2):
    tz1 = timezone(tz1)
    tz2 = timezone(tz2)
    datet = tz1.localize(datet)
    datet = datet.astimezone(tz2)
    return datet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading & Merging

# COMMAND ----------

# MAGIC %md
# MAGIC `tomtom_stations_pdf`: pandas dataframe for static station data
# MAGIC 
# MAGIC `tomtom_pdf`: pandas dataframe for 10-minute time series station availability data
# MAGIC 
# MAGIC `standLocs`: `tomtom_stations_pdf` with selected columns
# MAGIC 
# MAGIC `mergedtomtom`: `tomtom_stations_pdf` and `standLocs` merged based on station id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tomtom Static Station Data

# COMMAND ----------

## Load in the station data
tomtom_stations_df = spark.read.csv(f"{blob_url}/tomtom_stations.csv", header=True, inferSchema=True)
tomtom_stations_pdf = tomtom_stations_df.toPandas()
tomtom_stations_pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Series Data

# COMMAND ----------

tomtom_df = spark.read.parquet(f"/mnt/{mount_name}/tomtom_10min_data")
print(tomtom_df.cache().count())
display(tomtom_df.cache())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Previously Collected on Colab
# MAGIC 
# MAGIC Not currently used

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}/colab_data/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge Data

# COMMAND ----------

standLocs = tomtom_stations_pdf[['Standard Location', 'position.lat', 'position.lon', 'id', 'poi.url']]
standLocs['id'] = standLocs['id'].astype(str)

mergedtomtom = pd.merge(tomtom_pdf, standLocs,  
                                 left_on='station_id', 
                                 right_on='id', 
                                 how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformations

# COMMAND ----------

tomtom_pdf = tomtom_df.drop("availability.perPowerLevel").toPandas()


## Rename some locations
location_renaming_dict = {"TJHS": "Northern Virginia",
                          "Loudoun": "Northern Virginia",
                          "Fairfax": "Northern Virginia",
                          "Palo Alto Extended": "Palo Alto"}
tomtom_pdf['location'] = tomtom_pdf['location'].replace(location_renaming_dict)

## Convert timestamp column
tomtom_pdf["timestamp"] = pd.to_datetime(tomtom_pdf["timestamp"])

## Fix timezones
location_timezones =  {"Palo Alto": ("US/Pacific", 0),
                       "LA": ("US/Pacific", 0),
                       "Boulder": ("US/Mountain", 1),
                       "Northern Virginia": ("US/Eastern", 3)}
for location in location_timezones:
    location_mask = (tomtom_pdf["location"] == location)
    tomtom_pdf.loc[location_mask, "timestamp"] = tomtom_pdf[location_mask]["timestamp"].apply(lambda dt: dt + timedelta(hours=location_timezones[location][1]))

## Create a date-hour column
tomtom_pdf["datehour"] = pd.to_datetime(tomtom_pdf["timestamp"].dt.strftime('%m-%d-%Y %H'))
tomtom_pdf = tomtom_pdf.sort_values(by="datehour")

tomtom_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations (Static)

# COMMAND ----------

tomtom_stations_pdf['Standard Location'].value_counts()

# COMMAND ----------

# DBTITLE 1,Get station location count, broken down by location and standard location label
tomtom_stations_pdf.groupby(['Location', 'Standard Location']).count().reset_index()[['Location', 'Standard Location', 'id']]

# COMMAND ----------

# DBTITLE 1,Station location count by location and standard location label (Pivot)
pivot = pd.pivot_table(tomtom_stations_pdf, values = 'id', columns = 'Standard Location', index = 'Location', aggfunc = 'count').fillna(0)
pivot['Total'] = pivot.sum(axis = 1)
pivot.loc['Total', :] = pivot.sum(axis = 0)
pivot

# COMMAND ----------

# DBTITLE 1,Network provider information
temp = pd.DataFrame(tomtom_stations_pdf['poi.url'].value_counts()).reset_index().rename(columns = {'index': 'provider', 'poi.url': 'count'})
network_provider = []
for row in range(temp.shape[0]):
    provider = temp.loc[row, 'provider']
    network_provider.append(provider.split('.')[1] if provider.split('.')[0] == 'www' else provider.split('.')[0])
temp['provider'] = network_provider
temp = temp.groupby('provider').sum().reset_index()
temp['proportion'] = temp['count'] / temp['count'].sum()
temp.sort_values(by = 'count', ascending = False, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time Series Data

# COMMAND ----------

tomtom_df.count()

# COMMAND ----------

display(tomtom_df.cache())

# COMMAND ----------

tomtom_pdf.describe()

# no missing data points so far - 6/4 12am
# total - outliers station with 42! 

# COMMAND ----------

tomtom_pdf.info()

# date needs to be converted to date

# COMMAND ----------

tomtom_pdf.nunique()

# COMMAND ----------

# DBTITLE 1,Distribution of connector count for each station location
station_pdf = tomtom_pdf.groupby(["station_id"]).mean()
sns.histplot(data=station_pdf, x="total")

# COMMAND ----------

print(sorted(tomtom_pdf['total'].unique()))

# COMMAND ----------

mergedtomtom[['location', 'total', 'station_id']].drop_duplicates().groupby(['location', 'total']).size().reset_index().rename(columns = {0:'count'})

# COMMAND ----------

# DBTITLE 1,Total Number of Stations by Datetime
tomtom_pdf["minute_truncated"] = tomtom_pdf["minute"].apply(lambda m: math.floor(m / 10) * 10)
temp_df = tomtom_pdf.groupby(["date", "hour", "minute_truncated"]).sum()
temp_df = temp_df.reset_index()
sns.scatterplot(temp_df["date"], temp_df["total"])

# COMMAND ----------

# DBTITLE 1,Average Available Stations by Datehour
location_hour_pdf = tomtom_pdf.groupby(["datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()
sns.scatterplot(location_hour_pdf["datehour"], location_hour_pdf["availability.current.available"])

# COMMAND ----------

# DBTITLE 1,Percent Available Stations by Datehour
tomtom_pdf["percent_available"] = (tomtom_pdf["availability.current.available"] / tomtom_pdf["total"]) * 100
sns.lineplot(tomtom_pdf["datehour"], tomtom_pdf["percent_available"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merged Data

# COMMAND ----------

mergedtomtom.groupby(['location', 'Standard Location', 'total']).size().reset_index().rename(columns = {0:'count'})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC General thoughts, need to coordinate availability with other data
# MAGIC Is it just the time of day that is causing these low availabilities?
# MAGIC IS there other data that may answer it? events? traffic? holidays? 

# COMMAND ----------

# DBTITLE 1,Get network provider breakdown by number of connectors/ports
temp = pd.DataFrame(mergedtomtom.groupby(['poi.url', 'id']).mean()['total']).reset_index().groupby('poi.url').sum().reset_index().rename(columns = {'poi.url': 'provider', 'total': 'count ports'})
network_provider = []
for row in range(temp.shape[0]):
    provider = temp.loc[row, 'provider']
    network_provider.append(provider.split('.')[1] if provider.split('.')[0] == 'www' else provider.split('.')[0])
temp['provider'] = network_provider
temp = temp.groupby('provider').sum().reset_index()
temp['proportion'] = temp['count ports'] / temp['count ports'].sum()
temp = temp.sort_values(by = 'count ports', ascending = False, ignore_index=True)
temp.loc['Total', :] = ['all', temp['count ports'].sum(), temp['proportion'].sum()]
temp

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapping Static Stations

# COMMAND ----------

lats = list(tomtom_stations_pdf['position.lat'])
lons = list(tomtom_stations_pdf['position.lon'])
loc_types = list(tomtom_stations_pdf['Standard Location'])
locations = zip(lats, lons, loc_types)

color_dict = {'Office Buildings': "red",       
            'Multi-Unit Dwelling': 'green',    
            'Public Parking': 'blue',        
            'Retail/Restaurant': 'yellow',     
            'Other': 'black',         
            'University': 'purple',       
            'Hotel': 'yellow',              
            'Government Building': 'red',    
            'Mixed-Use': 'yellow',          
            'Airport': 'orange',        
             "Park": 'black'}

m = folium.Map(location=[37.0902, -95.7129], tiles="Stamen Terrain", zoom_start=5)
for lat, lon, loc_type in locations:
    folium.Marker([lat, lon], icon=folium.Icon(color=color_dict[loc_type], icon="bolt", prefix='fa-solid fa')).add_to(m)
m

# COMMAND ----------

# MAGIC %md
# MAGIC ## More Complex EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datehour Status Breakdowns

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for availability
location_hour_pdf = tomtom_pdf.groupby(["location", "datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()
locations = set(list(location_hour_pdf["location"]))
sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["availability.current.available"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for percent availability
location_hour_pdf = tomtom_pdf[tomtom_pdf["timestamp"] >= "2022-05-23"].groupby(["location", "datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()

location_hour_pdf["percent_available"] = (location_hour_pdf["availability.current.available"] / location_hour_pdf["total"]) * 100

sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["percent_available"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for occupied status
location_hour_pdf = tomtom_pdf.groupby(["location", "datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()
locations = set(list(location_hour_pdf["location"]))
sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["availability.current.occupied"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for unknown status
location_hour_pdf = tomtom_pdf.groupby(["location", "datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()
sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["availability.current.unknown"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for percent unknown
location_hour_pdf = tomtom_pdf.groupby(["location", "datehour"]).mean()

location_hour_pdf["percent_unknown"] = (location_hour_pdf["availability.current.unknown"] / location_hour_pdf["total"]) * 100
location_hour_pdf = location_hour_pdf.reset_index()

sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["percent_unknown"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Datehour breakdown for out of service status
location_hour_pdf = tomtom_pdf.groupby(["location", "datehour"]).mean()
location_hour_pdf = location_hour_pdf.reset_index()

locations = set(list(location_hour_pdf["location"]))

sns.lineplot(location_hour_pdf["datehour"], location_hour_pdf["availability.current.outOfService"], hue=location_hour_pdf["location"])

# COMMAND ----------

# DBTITLE 1,Percent available by state and location type
temp = mergedtomtom.groupby(['location', 'Standard Location', 'hour']).mean().reset_index()
# print(temp['Standard Location'].unique())

fig, axes = plt.subplots(2,2)
fig.suptitle('Comparing Charging Profiles for Different Building Types')
location_types = ['Office Buildings', 'Multi-Unit Dwelling', 'Retail/Restaurant', 'Other']
sns.lineplot(ax = axes[0, 0],
             x = temp[temp['Standard Location'] == location_types[0]]['hour'], 
             y = temp[temp['Standard Location'] == location_types[0]]['percent_available'],
             ci = 'sd',
             hue=temp["location"])
axes[0, 0].set_title(f'Utilization for {location_types[0]}')

sns.lineplot(ax = axes[0, 1],
             x = temp[temp['Standard Location'] == location_types[1]]['hour'], 
             y = temp[temp['Standard Location'] == location_types[1]]['percent_available'],
             hue=temp["location"])
axes[0, 1].set_title(f'Utilization for {location_types[1]}')

sns.lineplot(ax = axes[1, 0],
             x = temp[temp['Standard Location'] == location_types[2]]['hour'], 
             y = temp[temp['Standard Location'] == location_types[2]]['percent_available'],
             hue=temp["location"])
axes[1, 0].set_title(f'Utilization for {location_types[2]}')

sns.lineplot(ax = axes[1, 1],
             x = temp[temp['Standard Location'] == location_types[3]]['hour'], 
             y = temp[temp['Standard Location'] == location_types[3]]['percent_available'],
             hue=temp["location"])
axes[1, 1].set_title(f'Utilization for {location_types[3]}')

fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Individual Station Plots

# COMMAND ----------

# DBTITLE 1,Individual Station Plots
stations = set(list(tomtom_pdf["station_id"]))

## Plotting setup
fig_i, col_i, row_i  = 0, 0, 0
fig_count = len(stations)
col_count = 4
row_count = math.ceil(fig_count / col_count)

## Plot distributions
fig, axx = plt.subplots(row_count, col_count, figsize=(50,50))
for station in stations:
    graph_pdf = tomtom_pdf[tomtom_pdf["station_id"] == station]
    sns.scatterplot(graph_pdf["timestamp"], graph_pdf["availability.current.available"], ax=axx[row_i][col_i]).set_title(station)
    fig_i += 1
    col_i = fig_i % col_count
    if col_i == 0:
        row_i = row_i + 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Availability by Charger Type and State

# COMMAND ----------

hourly_avail = tomtom_pdf[['location', 'hour', 'type', 'percent_available']].groupby(['location', 'hour', 'type']).mean()

seaborn_facet_location2 = sns.catplot(
    x= 'hour',
    y = 'percent_available', 
    row = 'location', 
    col = 'type',
    kind = 'bar',
    data = hourly_avail.reset_index()
)

seaborn_facet_location2


#v

# COMMAND ----------

mergedtomtom[mergedtomtom['location'] == 'Northern Virginia'][['location', 'Standard Location', 'type', 'total', 'station_id']].drop_duplicates().groupby(['location', 'Standard Location', 'type', 'total']).size().reset_index().rename(columns = {0:'count'})

# COMMAND ----------

mergedtomtom

# COMMAND ----------

# MAGIC %md
# MAGIC ### Availability by Charger Type and Location Type

# COMMAND ----------

hourly_avail2 = mergedtomtom[['location', 'hour', 'type', 'percent_available', 'Standard Location']].groupby(['location','Standard Location', 'hour', 'type']).mean()

seaborn_facet_location3 = sns.catplot(
    x= 'hour',
    y = 'percent_available', 
    row = 'location', 
    col = 'Standard Location',
    kind = 'bar',
    data = hourly_avail2.reset_index()
)

seaborn_facet_location3

#Boulder Colorado
#### governement buildings tend to have an average of 60% avail or less. Multi-Use areas are highly used. CAn see lower availability 3pm and later at other locations

# LA
#### Hotels tend to have consistent averages of 60% or less
### muti-unit dwellings as well
# Other, and retails/restuarant locations becomes highly unavailable in later part of the day 

#Northern Virginia 
#### Public Parking EVs have the most utilization and seem to have rare availability, with large parts of the day with averages of less than 40% and some less than 20%

#Palo Alto
#### Retail/restuarant seems to have the most availability

# COMMAND ----------

hourlyNV = mergedtomtom[mergedtomtom['location']=='Northern Virginia'][['location', 'hour', 'type', 'percent_available', 'Standard Location']].groupby(['location','Standard Location', 'hour', 'type']).mean()

seaborn_facet_location3 = sns.catplot(
    x= 'hour',
    y = 'percent_available', 
    #row = 'location', 
    col = 'Standard Location',
    kind = 'bar',
    data = hourlyNV.reset_index()
)

seaborn_facet_location3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Low Availability Analysis

# COMMAND ----------

print('Share of Data with less than 50% availability: {:.0f}%'.format( 100*(len(tomtom_pdf[tomtom_pdf['percent_available']< 50] ) / len(tomtom_pdf))))


# COMMAND ----------

# thoughts -- want to do a map that plays the data in a time series, and shows different stations with availability -- can watch individual stations drop to less than 50%

df = mergedtomtom[mergedtomtom['location']=='Palo Alto']

df['text'] = df['location'] + '<br>Standard Location ' + (df['Standard Location']).astype(str)
limits = [(0,49), (50,100)]
colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
cities = []
scale = 1

fig = go.Figure()

for i in range(len(limits)):
    lim = limits[i]
    df_sub = df[lim[0]:lim[1]]
    fig.add_trace(go.Scattergeo(
        locationmode = 'geojson-id',
        lon = df_sub['position.lon'],
        lat = df_sub['position.lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['percent_available'],
            color = colors[i],
            line_color='rgb(40,40,40)',
            line_width=0.5,
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])))

fig.update_layout(
        title_text = 'Map',
        showlegend = True,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        )
    )

fig.show()

# COMMAND ----------

tomtom_pdf["low_availability_flag"] = tomtom_pdf["percent_available"].apply(lambda x: x <= 50)
g = sns.scatterplot(tomtom_pdf["datehour"], tomtom_pdf["percent_available"], hue=tomtom_pdf["low_availability_flag"])

# COMMAND ----------

tomtom_pdf["low_availability_flag"] = tomtom_pdf["percent_available"].apply(lambda x: x <= 50)
g = sns.lineplot(tomtom_pdf["datehour"], tomtom_pdf["percent_available"], hue=tomtom_pdf["low_availability_flag"])

# COMMAND ----------

pd.to_datetime(g_pdf["date"])

# COMMAND ----------

g_pdf = tomtom_pdf.groupby(["date", "hour", "minute_truncated"]).agg({"total": "sum", "low_availability_flag": "sum"})
g_pdf = g_pdf.reset_index()
g_pdf["date"] = pd.to_datetime(g_pdf["date"])
g_pdf["datetime"] = pd.to_datetime(dict(g_pdf["date"].dt.year, 'hour', 'minute_truncated'))
g_pdf["percent_low_availability"] = (g_pdf["low_availability_flag"] / g_pdf["total"]) * 100
g_pdf
# g = sns.lineplot(tomtom_pdf["datehour"], tomtom_pdf["percent_available"], hue=tomtom_pdf["low_availability_flag"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scratch Work

# COMMAND ----------

tomtom_stations_pdf[tomtom_stations_pdf['Location']== 'Palo Alto'][[
    'address.freeformAddress', 'position.lat', 'position.lon',
    'Location', 'LocationNotes', 'Standard Location'
]]

# veiwing stations to identify how to merge with palo alto historical