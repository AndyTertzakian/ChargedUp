# Databricks notebook source
# import libraries
import pandas as pd
import math
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
import pyspark.sql.functions as F
import math
import folium
import holidays
from calendar import monthrange
import altair as alt

from statsmodels.tsa.seasonal import seasonal_decompose


## Modeling Libraries

# Model Evaluation
from sklearn.metrics import mean_squared_error as mse

# ARIMA/VAR Models
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from keras.models import Model, Sequential

# Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet



import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)



import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

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

slrp_prophet = spark.read.parquet(f"/mnt/{mount_name}/data/Slrp_prophet")
boulder_prophet = spark.read.parquet(f"/mnt/{mount_name}/data/boulder_prophet")
palo_alto_prophet = spark.read.parquet(f"/mnt/{mount_name}/data/paloAlto_prophet")

# COMMAND ----------

display(slrp_prophet)

# COMMAND ----------

display(boulder_prophet)

# COMMAND ----------

display(palo_alto_prophet)

# COMMAND ----------

slrp_prophet_edit = slrp_prophet.withColumn("Location", F.lit("Berkeley"))
slrp_prophet_edit = slrp_prophet_edit.withColumn("SiteName", F.lit("Slrp"))
display(slrp_prophet_edit)

# COMMAND ----------

print(slrp_prophet_edit.count())

# COMMAND ----------

print(boulder_prophet.count())

# COMMAND ----------

all_prophet = slrp_prophet_edit.union(boulder_prophet).union(palo_alto_prophet)

# COMMAND ----------

## Write to AWS S3
(all_prophet
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/all_prophet"))

# COMMAND ----------

all_prophet_df = all_prophet.toPandas()

# COMMAND ----------

def calc_grouped_rmse(df):
    return np.sqrt(np.mean((df['predictions_(rounded)'] - df['Actuals'])**2))

# COMMAND ----------

all_prophet_df.groupby(['SiteName', 'Location']).apply(calc_grouped_rmse).groupby('Location').mean()

# COMMAND ----------

slrp_ARIMA = spark.read.parquet(f"/mnt/{mount_name}/data/BerkeleySlrp_ARIMA/")
boulder_ARIMA = spark.read.parquet(f"/mnt/{mount_name}/data/Boulder_ARIMA_Tail3months/")
palo_alto_ARIMA = spark.read.parquet(f"/mnt/{mount_name}/data/PaloAlto_ARIMA_Tail3months/")

# COMMAND ----------

display(slrp_ARIMA)

# COMMAND ----------

display(boulder_ARIMA)

# COMMAND ----------

display(palo_alto_ARIMA)

# COMMAND ----------

slrp_ARIMA_edit = slrp_ARIMA.withColumn("Location", F.lit("Berkeley"))
slrp_ARIMA_edit = slrp_ARIMA_edit.withColumn("SiteName", F.lit("Slrp"))
slrp_ARIMA_edit = slrp_ARIMA_edit.select(boulder_ARIMA.columns)

# COMMAND ----------

display(slrp_ARIMA_edit)

# COMMAND ----------

all_ARIMA = slrp_ARIMA_edit.union(boulder_ARIMA).union(palo_alto_ARIMA)
## Write to AWS S3
(all_ARIMA
    .write
    .format("parquet")
    .mode("overwrite")
    .save(f"/mnt/{mount_name}/data/all_arima_1var"))

# COMMAND ----------

all_ARIMA_df = all_ARIMA.toPandas()
all_ARIMA_df.groupby(['SiteName', 'Location']).apply(calc_grouped_rmse).groupby('Location').mean()

# COMMAND ----------

