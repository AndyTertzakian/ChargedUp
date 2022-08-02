# Databricks notebook source
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
from plotly.subplots import make_subplots
import pytz
from fp.fp import FreeProxy
import pyspark.sql.functions as F
import warnings
import matplotlib.pyplot as plt
import math
from keras.models import Model, Sequential
from pyspark.sql.types import *
from folium import Map, CircleMarker, Vega, Popup
import folium
import vincent
import numpy as np
import math
import branca

## Settings
# warnings.filterwarnings("ignore")
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

# display(dbutils.fs.ls(f"/mnt/{mount_name}/"))

# COMMAND ----------

station_to_coords = {
                    ## Berkeley
                    'Slrp': {"lat": 37.8686688, "long": -122.2634667},
                    
                    ## Palo Alto
                    'BRYANT': {"lat": 37.4467274, "long": -122.1630922},
                    'WEBSTER': {"lat": 37.4484262,"long": -122.1620252},
                    'HAMILTON': {"lat": 37.449283, "long": -122.1582532},
                    'HIGH': {"lat": 37.4431333, "long": -122.1719076},
                    'TED_THOMPSON': {"lat": 37.4279917, "long": -122.1449875}, # Parkade
                    'MPL': {"lat": 37.422751, "long": -122.1140535},           # Mitchell Park Library
                    'RINCONADA_LIB': {"lat": 37.4497552, "long": -122.1491587},
                    'CAMBRIDGE': {"lat": 37.4272462, "long": -122.1460422}, 
                      
                    ## Boulder
#                      '2280 Junction Pl': {"lat": 40.024670, "long": -105.251999},
#                      '1275 Alpine Ave': {"lat": 40.025871, "long": -105.280724},
#                      '900 Baseline Rd': {"lat": 39.999850, "long": -105.281390},
                     '1745 14th street': {"lat": 40.015280, "long": -105.276370},
                     '1500 Pearl St': {"lat": 40.0184689,"long":-105.2774864},
                     '1770 13th St': {"lat": 40.0154509,"long":-105.2794388},
                     '1360 Gillaspie Dr': {"lat": 39.9746415,"long":-105.2508207},
                     '900 Walnut St': {"lat":40.0159843,"long":-105.2846621},
                     '1100 Spruce St': {"lat":40.0185517,"long": -105.2836497},
                     '1400 Walnut St': {"lat":40.0169488,"long": -105.2786352},
#                      '7315 Red Deer Dr': {"lat":40.0239185,"long": -105.1848833},
                     '2052 Junction Pl':{"lat":40.0259532,"long": -105.2529939},
                     '1100 Walnut': {"lat":40.0163663,"long": -105.2827991},
                     '1739 Broadway': {"lat":40.0144993,"long": -105.2814581},
                     '3172 Broadway': {"lat":40.0327682,"long": -105.2831966},
#                      '2150 13th St': {"lat":40.0198099,"long": -105.2811432},
                     '5660 Sioux Dr': {"lat":39.9921117,"long": -105.222267},
                     '5565 51st St': {"lat":40.0800904,"long": -105.2386908},
                     '1505 30th St': {"lat":40.0126607,"long": -105.2571956},
                     '3335 Airport Rd': {"lat":40.0373216,"long": -105.2327793},
                     '600 Baseline Rd': {"lat":39.9987901,"long": -105.2848274},
#                      '5050 Pearl St': {"lat":40.0252908,"long": -105.2374913},
                     '2667 Broadway': {"lat":40.0253516,"long": -105.2845029},
#                      '2240 Broadway': {"lat":40.0201865,"long": -105.2824309},
                     '5333 Valmont Rd': {"lat":40.0297918,"long": -105.2320102}
                    }

# COMMAND ----------

station_df_dict = {station: {"preds": None, "errs": None} for station in station_to_coords}

for station in station_df_dict:
    
    ## Show which station we are currently working on
    print(station)
    
    ## If the tables are already stored
    if spark._jsparkSession.catalog().tableExists('default', station.replace(" ", "_")):
        
        print("\tEXISTS")
        
        ## Retrieve the table from hive and convert it to pandas
        station_df_dict[station]["errs"] = spark.sql(f'select * from {station.replace(" ", "_")}').toPandas()
        station_df_dict[station]["errs"] = station_df_dict[station]["errs"].sort_values(["stream_count", "datetime"])
        station_df_dict[station]["errs"] = station_df_dict[station]["errs"].reset_index(drop=True)
        
        ## Convert the latitudes and longitudes
#         station_df_dict[station]["errs"]["lat"] = station_df_dict[station]["errs"]["station"].apply(lambda station: station_to_coords[station]["lat"])
#         station_df_dict[station]["errs"]["long"] = station_df_dict[station]["errs"]["station"].apply(lambda station: station_to_coords[station]["long"])
        
        
    else:
        
        print("\tNEW")
        
#     palo_alto_preds_df = spark.read.parquet(f"/mnt/{mount_name}/data/streaming_predictions/{station.lower()}_preds")
#     station_df_dict[station]["preds"] = palo_alto_preds_df
    
        pred_errors_df = spark.read.parquet(f"/mnt/{mount_name}/data/streaming_predictions/{station.lower()}_pred_errors")
        station_df_dict[station]["errs"] = pred_errors_df

        station_df_dict[station]["errs"].write.mode("overwrite").saveAsTable(station.replace(" ", "_"))
        station_df_dict[station]["errs"] = spark.sql(f'select * from {station.replace(" ", "_")}').toPandas()
        station_df_dict[station]["errs"] = station_df_dict[station]["errs"].sort_values(["stream_count", "datetime"])
        station_df_dict[station]["errs"] = station_df_dict[station]["errs"].reset_index(drop=True)

# COMMAND ----------

station_df

# COMMAND ----------

for station in station_to_coords:
    station_df =  station_df_dict[station]["errs"]
    station_df = station_df[station_df["datetime"] <= station_df["datetime"].min() + dt.timedelta(days=7)]

    frames = []
    prediction_steps = station_df["stream_count"].unique()
    for prediction_step in prediction_steps:
        
        frame_df = station_df[station_df["stream_count"] == prediction_step]
        frame_df_2 = station_df[(station_df["stream_count"] <= prediction_step) & (station_df["timesteps_out"] == 0)]

    #     frame_df = frame_df_2.append(frame_df)
        frame = go.Frame(data=[go.Scatter(x=frame_df["datetime"], y=frame_df["predicted_rounded"]),
                               go.Scatter(x=frame_df_2["datetime"], y=frame_df_2["ports_available"])])
        
        frames.append(frame)

    graph_df = station_df[station_df["stream_count"] == prediction_steps[-2]]
    graph_df_2 = station_df[(station_df["stream_count"] < prediction_steps[-2]) & (station_df["timesteps_out"] == 0 )]

    fig = go.Figure(
        data=[
              go.Scatter(x=graph_df["datetime"], y=graph_df["ports_available"], name = 'Predicted Future Availability'), 
              go.Scatter(x=graph_df_2["datetime"], y=graph_df_2["predicted_rounded"], name="Past Availability"),
    #           go.Bar(x=)
             ],
        layout=go.Layout(

            width = 850,
            height = 400,
            xaxis=dict(range=[station_df["datetime"].min(), station_df["datetime"].min() + dt.timedelta(days=7)], 
                       autorange=False),
            yaxis=dict(range=[0, 10], autorange=False),
            title=station + " - Station Availability Forecast",
            legend = dict(orientation = 'h', xanchor = "center", x = 0.72, y= 1.15), #Adjust legend position
                 yaxis_title='Availability',
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args = [None, {"frame": {"duration": 1000, 
                                                       "redraw": True},
                                             "fromcurrent": True, 
                                             "transition": {"duration": 500}}],
                             ),
                          {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                        ]
            )]
        ),
        frames=frames
    )

    fig.show()
    filename = station.replace(" ", "_")
    dbutils.fs.put(f"/FileStore/maps/{filename}.html", fig.to_html(), True)

# COMMAND ----------

for station in station_to_coords:
      
    print(station)

    df_station = station_df_dict[station]["errs"]
    df_station = df_station[df_station["stream_count"] == df_station["stream_count"].max()]

    fig=make_subplots(specs=[[{"secondary_y":True}]])
    
    if False:
        fig.add_trace(                           #Add a bar chart to the figure
                go.Line(
                x=df_station['datetime'],
                y=df_station['ports_available'],
                line= {"shape": 'hv'},
                name="Availability",
                hoverinfo='none'                 #Hide the hoverinfo
                ),
                secondary_y=False)               #The bar chart uses the primary y-axis (left)

    fig.add_trace(                           #Add a bar chart to the figure
            go.Line(
            x=df_station['datetime'],
            y=df_station['predicted_rounded'],
            line= {"shape": 'hv'},
            name="Predicted Availability",
            hoverinfo='none'                 #Hide the hoverinfo
            ),
            secondary_y=False)               #The bar chart uses the primary y-axis (left)


    fig.update_layout(autosize=False,
                      width = 850,
                      height = 400,
                 hoverlabel_bgcolor='#DAEEED',  #Change the background color of the tooltip to light gray
                 title_text=f"Station Availability - {station}", #Add a chart title
                 title_font_family="Times New Roman",
                 title_font_size = 14,
                 title_font_color="darkblue", #Specify font color of the title
                 title_x=0.5, #Specify the title position
                 updatemenus=[dict(
                                type="buttons",
                                buttons=[
                                        dict(label="Play",
                                             method="animate",
                                             args=[None])
                                        ]
                                 )
                             ],
                 xaxis=dict(
                        tickfont_size=10,
                        tickangle = 270,
                        showgrid = True,
                        zeroline = True,
                        showline = True,
                        showticklabels = True,
                        dtick="D1", #Change the x-axis ticks to be monthly
                        tickformat="%h %d %b\n%Y"
                        ),
                 legend = dict(orientation = 'h', xanchor = "center", x = 0.72, y= 1.15), #Adjust legend position
                 yaxis_title='Availability',
                 yaxis2_title='Predicted Availability')

    fig.update_xaxes(
    range= [df_station["datetime"].max() - dt.timedelta(hours=6), df_station["datetime"].max()],
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
    #         dict(count=6, label="6m", step="month", stepmode="backward"),
    #         dict(count=1, label="YTD", step="year", stepmode="todate"),
    #         dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
    
    fig.show()
    filename = station.replace(" ", "_")
    dbutils.fs.put(f"/FileStore/maps/{filename}.html", fig.to_html(), True)
    break

# COMMAND ----------

# map_df = station_df_dict["BRYANT"]["errs"]

lats = [station_to_coords[station]["lat"] for station in [station_name for station_name in station_to_coords]]
lons = [station_to_coords[station]["long"] for station in [station_name for station_name in station_to_coords]]

locations = zip(lats, lons)
locations

m = folium.Map(location=[37.446727,	-122.163092], tiles="Stamen Terrain", zoom_start=12)
for lat, lon in locations:

    url = "https://andytertzakian.github.io/chargedup.github.io/maps/BRYANT.html"
    html="""<iframe src=\"""" + url + """\" width="850" height="400"  frameborder="0">"""

    popup = folium.Popup(folium.Html(html, script=True))

    folium.Marker([lat, lon],
                  popup = popup,
                  icon=folium.Icon(color="green", icon="bolt", prefix='fa-solid fa')).add_to(m)
m

# COMMAND ----------

dbutils.fs.put(f"/FileStore/maps/MAP.html", m)

# COMMAND ----------

dir(m)