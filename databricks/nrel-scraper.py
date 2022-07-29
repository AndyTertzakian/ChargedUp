# Databricks notebook source
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

## Settings
warnings.filterwarnings("ignore")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
sns.set(rc={'figure.figsize':(16,9)})

## AZURE
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
secret_scope = "w210-scope"
access_key = dbutils.secrets.get(scope=secret_scope, key="aws-access-key")
secret_key = dbutils.secrets.get(scope=secret_scope, key="aws-secret-key")
encoded_secret_key = aws_secret_key.replace("/", "%2F")
aws_bucket_name = "w210v2"
mount_name = "w210v2"

try:
    dbutils.fs.mount("s3a://%s:%s@%s" % (aws_access_key, encoded_secret_key, aws_bucket_name), "/mnt/%s" % mount_name)
except Exception as e:
    print("already mounted :)")

# COMMAND ----------

api_key = dbutils.secrets.get(scope=secret_scope, key="nrel-api-key")
# resp = requests.get(f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?api_key={api_key}&fuel_type=ELEC&state=CA&limit=all")
resp = requests.get(f"https://developer.nrel.gov/api/alt-fuel-stations/v1.json?api_key={api_key}&fuel_type=ELEC&limit=all")
resp

# COMMAND ----------

len(resp.json()["fuel_stations"])

# COMMAND ----------

nrel_pdf = pd.DataFrame(resp.json()["fuel_stations"])
nrel_pdf = nrel_pdf.fillna(np.NAN)
nrel_pdf = nrel_pdf.drop(["federal_agency", "ev_network_ids", "ev_connector_types"], axis = 1)
nrel_pdf

# COMMAND ----------

input_pdf = nrel_pdf
nrel_df = spark.createDataFrame(input_pdf)
nrel_df.repartition(1).write.format('csv').mode('overwrite').save(f"/mnt/{mount_name}/nrel_stations.csv", header=True)
display(nrel_df)