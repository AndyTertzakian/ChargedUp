# Databricks notebook source
## Imports
import json
import time
import urllib
import requests
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.io.json import json_normalize
import datetime as dt
from datetime import timedelta
import pytz
from fp.fp import FreeProxy
import pyspark.sql.functions as F
import warnings
import matplotlib.pyplot as plt
import math
import geopandas
import haversine as hs
import sys

## Settings
warnings.filterwarnings("ignore")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
sns.set(rc={'figure.figsize':(16,9)})
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 10)

## Azure
blob_container = "w210" # The name of your container created in https://portal.azure.com
storage_account = "w210" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w210-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "azure-storage-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

secret_scope = "w210-scope"
GOOGLE_API_KEY = dbutils.secrets.get(scope=secret_scope, key="google-api-key")

## AWS
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

def flatten(t):
    return [item for sublist in t for item in sublist]

# COMMAND ----------

# DBTITLE 1,Load Tomtom data
tomtom_stations_df = spark.read.csv(f"/mnt/{mount_name}/tomtom_stations.csv", header=True)
tomtom_stations_pdf = tomtom_stations_df.toPandas()

tomtom_stations_pdf["position.lat"] = tomtom_stations_pdf["position.lat"].astype('float64')
tomtom_stations_pdf["position.lon	"] = tomtom_stations_pdf["position.lon"].astype('float64')

## Create a columns
tomtom_stations_pdf["loc"] = tomtom_stations_pdf[["position.lat","position.lon	"]].apply(tuple, axis=1)

tomtom_stations_pdf

# COMMAND ----------

# DBTITLE 1,Load NREL data
## Read in the NREL data
nrel_stations_df = spark.read.csv(f"{blob_url}/nrel_stations.csv", header=True)
nrel_stations_pdf = nrel_stations_df.toPandas()
nrel_stations_pdf = nrel_stations_pdf[(nrel_stations_pdf["state"] == "CA") | (nrel_stations_pdf["state"] == "CO") | (nrel_stations_pdf["state"] == "VA")]

nrel_stations_pdf["latitude"] = nrel_stations_pdf["latitude"].astype('float64')
nrel_stations_pdf["longitude"] = nrel_stations_pdf["longitude"].astype('float64')

drop_cols = ["access_detail_code", "cards_accepted", "expected_date", "bd_blends", "bd_blends", "cng_fill_type_code", "cng_psi", 
             "cng_renewable_source", "cng_total_compression", "cng_total_storage", "cng_vehicle_class", "e85_blender_pump", "e85_other_ethanol_blends", 
             "hy_is_retail", "hy_pressures", "hy_standards", "hy_status_link", "lng_renewable_source", "lng_vehicle_class", "lpg_primary", "lpg_nozzle_types",
             "ng_fill_type_code", "ng_psi", "ng_vehicle_class", "access_days_time_fr", "intersection_directions_fr", "bd_blends_fr", "cng_dispenser_num", "groups_with_access_code_fr", "ev_pricing_fr"]

## Display the dataframe
nrel_stations_pdf = nrel_stations_pdf.drop(drop_cols, axis=1)

nrel_stations_pdf

# COMMAND ----------

# DBTITLE 1,Perform geospatial proximity analysis
def distance_from(loc1,loc2): 
    dist=hs.haversine(loc1,loc2)
    return round(dist, 6)


def get_closest_station_id(loc):

    loc1 = loc
    if type(loc) != tuple:
        return None
    
    tomtom_lat, tomtom_long = loc
    if tomtom_lat == np.nan or tomtom_long == np.nan:
        return -1
    
    ## Extract the location details
    nrel_station_ids = list(nrel_stations_pdf['id'])
    lats = list(nrel_stations_pdf['latitude'])
    lons = list(nrel_stations_pdf['longitude'])
    nrel_locations = zip(nrel_station_ids, lats, lons)
    
    closest_distance, closest_nrel_station_id = sys.maxsize, None
    for nrel_station_id, nrel_lat, nrel_lon in nrel_locations:
        
        loc2 = (nrel_lat, nrel_lon)
        
        distance = hs.haversine(loc1,loc2)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_nrel_station_id = nrel_station_id
    
    print(f"The best station is {closest_nrel_station_id}, with a distance of: {closest_distance}")
    return closest_nrel_station_id

tomtom_stations_pdf["nrel_station_id"] = tomtom_stations_pdf["loc"].apply(lambda loc: get_closest_station_id(loc))
tomtom_stations_pdf

# COMMAND ----------

# DBTITLE 1,Perform join
joined_pdf = tomtom_stations_pdf.set_index("nrel_station_id").join(nrel_stations_pdf.set_index("id"), how="left", rsuffix="nrel")
joined_pdf = joined_pdf.drop(["chargingPark.connectors", "entryPoints", "poi.classifications", "loc"], axis=1)
# joined_pdf
joined_df = spark.createDataFrame(joined_pdf)
display(joined_df)