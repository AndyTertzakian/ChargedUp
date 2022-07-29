# Databricks notebook source
## Imports
import json
import time
import urllib
import requests
import pandas as pd
import seaborn as sns
from pandas.io.json import json_normalize
import datetime as dt
from fp.fp import FreeProxy
import pyspark.sql.functions as F
import warnings
from pytz import timezone

## Settings
warnings.filterwarnings("ignore")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

blob_container = "w210" # The name of your container created in https://portal.azure.com
storage_account = "w210" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w210-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "azure-storage-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

display(dbutils.fs.ls(f"{blob_url}/tomtom"))

# COMMAND ----------

request_url = "https://api.weather.gov/points/39.7456,-97.0892"
resp = requests.get(request_url)
resp_json = resp.json()
resp_json

# COMMAND ----------

 https://api.worldweatheronline.com/premium/v1/past-weather.ashx