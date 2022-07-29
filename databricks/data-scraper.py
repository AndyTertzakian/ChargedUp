# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

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

api_keys = {
            0: dbutils.secrets.get(scope=secret_scope, key="tomtom-api-key"), ## berkeley
            1: "5OtsXaDDQ7NhNgGipHhXReTqRHFPG434",
            2: "hvTJO9RJu2gbMfn8JGhibKFAtbpqJ6pW",
            3: "zhcVaqDutgYWJaGD7QYz94gsgZe50sDA",
            4: "aPhTHu4FriNZ6rBXxyGXbnhW5qpS32JV",
            5: "kMGdhLC5tXucYuPPdghXtbdveXdyBt1W",
            6: "9U21gS28b5w3ASWW51bEAGLgRoJXkvnA",  ##yyf
            7: "zqmaSA3T6ndOQQ7JvsuHYzS77AyLMEfS",  ##gmail
            8: "UYWNfDgyh8GGB3lOw7BgNZcrcpedbDxS",  ##work
            9: "POexenFH1AgPk6Uy2xAFbxYlcF4AGGZe",
           10: "mWAe9LRSSKrIudJ6QsUmFeO1FDg4oRKr",
           11: "yYflx08VyxlG1283iZNhIuWHGfpZHGhN"                                   ## graburn
           }

def convert_datetime_timezone(datet, tz1, tz2):
    tz1 = timezone(tz1)
    tz2 = timezone(tz2)
    datet = tz1.localize(datet)
    datet = datet.astimezone(tz2)
    return datet

def get_api_key():
    
    us_pacific_time = convert_datetime_timezone(dt.datetime.now(), "UTC", "US/Pacific")
    hour = us_pacific_time.hour 
    minute = us_pacific_time.minute 
    
    if hour % 2 == 0:
        minute_index = (minute - (minute % 10)) / 10
    else:
        minute_index = (minute - (minute % 10)) / 10
        minute_index += 6
        
    print(f"The api key selected: {api_keys[minute_index]}")

    return api_keys[minute_index]

# COMMAND ----------

headers = {'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"}

station_parameters =  {'key': f'{get_api_key()}',
                       'limit': 30,
                       'countryset': 'USA'}

availability_parameters =  {'key': f'{get_api_key()}',
                            'chargingAvailability': None}

locations_parameters = {
        "UCI":{
            'lat': 33.6405,
            'lon': -117.8443,
            'radius': 10000
        },
        "DTLA": {
            'lat': 34.0488,
            'lon': -118.2518,
            'radius': 10000
        },
        "UCB": {
             'lat': 37.8719,
            'lon': -122.2585,
            'radius': 10000
        },
       "NYC": {
            'lat': 40.7128,
            'lon': -74.0060,
            'radius': 10000
       } 
}

stations = [{ "station_ID": "840089001992231", "location": "Boulder"},
            { "station_ID": "840089001992274", "location": "Boulder"},
            { "station_ID": "840089003010633", "location": "Boulder"},
            { "station_ID": "840089001992287", "location": "Boulder"},
            { "station_ID": "840089001725407", "location": "Boulder"},
            { "station_ID": "840089001992298", "location": "Boulder"},
            { "station_ID": "840089001992156", "location": "Boulder"},
            { "station_ID": "840089001884788", "location": "Boulder"},
            { "station_ID": "840089002647580", "location": "Boulder"},
            { "station_ID": "840089001992186", "location": "Boulder"},
            { "station_ID": "840089003010423", "location": "Boulder"},
            { "station_ID": "840089003010802", "location": "Boulder"},
            { "station_ID": "840089001709984", "location": "Boulder"},
            { "station_ID": "840069009572391", "location": "Palo Alto"},
            { "station_ID": "840069009633761", "location": "Palo Alto"},
            { "station_ID": "840069018354510", "location": "Palo Alto"},
            { "station_ID": "840069009503788", "location": "Palo Alto"},
            { "station_ID": "840069021369779", "location": "Palo Alto"},
            { "station_ID": "840069021369548", "location": "Palo Alto"},
            { "station_ID": "840069009489523", "location": "Palo Alto"},
            { "station_ID": "840069009581369", "location": "Palo Alto"},
            { "station_ID": "840519002344025", "location": "Northern Virginia"},
            { "station_ID": "840519002261042", "location": "Northern Virginia"},
            { "station_ID": "840519002070381", "location": "Northern Virginia"},
            { "station_ID": "840519002897062", "location": "Northern Virginia"},
            { "station_ID": "840519002896617", "location": "Northern Virginia"},
            { "station_ID": "840519002233150", "location": "Northern Virginia"},
            { "station_ID": "840519002896618", "location": "Northern Virginia"},
            { "station_ID": "840519002897186", "location": "Northern Virginia"},
            { "station_ID": "840519002086241", "location": "Northern Virginia"},
            { "station_ID": "840519002508573", "location": "Northern Virginia"},
            { "station_ID": "840519002896640", "location": "Northern Virginia"},
            { "station_ID": "840519002240196", "location": "Northern Virginia"},
            { "station_ID": "840519002076860", "location": "Northern Virginia"},
            { "station_ID": "840519002246187", "location": "Northern Virginia"},
            { "station_ID": "840519002896633", "location": "Northern Virginia"},
            { "station_ID": "840519002101427", "location": "Northern Virginia"},
            { "station_ID": "840519002244304", "location": "Northern Virginia"},
            { "station_ID": "840519002897990", "location": "Northern Virginia"},
            { "station_ID": "840519002508658", "location": "Northern Virginia"},
            { "station_ID": "840519002897761", "location": "Northern Virginia"},
            { "station_ID": "840519002084127", "location": "Northern Virginia"},
            { "station_ID": "840519002344119", "location": "Northern Virginia"},
            { "station_ID": "840519002898003", "location": "Northern Virginia"},
            { "station_ID": "840519002896861", "location": "Northern Virginia"},
            { "station_ID": "840519002612683", "location": "Northern Virginia"},
            { "station_ID": "840519002897867", "location": "Northern Virginia"},
            { "station_ID": "840519002247877", "location": "Northern Virginia"},
            { "station_ID": "840519002612665", "location": "Northern Virginia"},
            { "station_ID": "840519002898011", "location": "Northern Virginia"},
            { "station_ID": "840519002897967", "location": "Northern Virginia"},
            { "station_ID": "840069009483217", "location": "Palo Alto"},
            { "station_ID": "840069015737097", "location": "Palo Alto"},
            { "station_ID": "840069021371556", "location": "Palo Alto"},
            { "station_ID": "840069018355255", "location": "Palo Alto"},
            { "station_ID": "840069019889058", "location": "Palo Alto"},
            { "station_ID": "840069019890026", "location": "Palo Alto"},
            { "station_ID": "840069015576267", "location": "Palo Alto"},
            { "station_ID": "840069021372096", "location": "Palo Alto"},
            { "station_ID": "840069021368522", "location": "Palo Alto"},
            { "station_ID": "840069018355288", "location": "Palo Alto"},
            { "station_ID": "840069021369722", "location": "Palo Alto"},
            { "station_ID": "840069018355185", "location": "Palo Alto"},
            { "station_ID": "840069021369641", "location": "Palo Alto"},
            { "station_ID": "840069006968168", "location": "Palo Alto"},
            { "station_ID": "840069006969514", "location": "Palo Alto"},
            { "station_ID": "840069020352343", "location": "Palo Alto"},
            { "station_ID": "840069021368062", "location": "Palo Alto"},
            { "station_ID": "840089002698770", "location": "Boulder"},
            { "station_ID": "840089001718628", "location": "Boulder"},
            { "station_ID": "840089001708984", "location": "Boulder"},
            { "station_ID": "840089002698744", "location": "Boulder"},
            { "station_ID": "840089002698798", "location": "Boulder"},
            { "station_ID": "840089002746614", "location": "Boulder"},
            { "station_ID": "840089003012110", "location": "Boulder"},
            { "station_ID": "840089001704670", "location": "Boulder"},
            { "station_ID": "840089001992150", "location": "Boulder"},
            { "station_ID": "840089001992249", "location": "Boulder"},
            { "station_ID": "840089001992192", "location": "Boulder"},
            { "station_ID": "840089003010955", "location": "Boulder"},
            { "station_ID": "840089003010935", "location": "Boulder"},
            { "station_ID": "840089001876586", "location": "Boulder"},
            { "station_ID": "840089002746495", "location": "Boulder"},
            { "station_ID": "840089002647457", "location": "Boulder"},
            { "station_ID": "840089003011988", "location": "Boulder"},
            { "station_ID": "840089001717871", "location": "Boulder"},
            { "station_ID": "840089003011019", "location": "Boulder"},
            { "station_ID": "840089001718441", "location": "Boulder"},
            { "station_ID": "840089003012299", "location": "Boulder"},
            { "station_ID": "840089003012158", "location": "Boulder"},
            { "station_ID": "840089002746551", "location": "Boulder"},
            { "station_ID": "840089002746512", "location": "Boulder"},
            { "station_ID": "840089001877903", "location": "Boulder"},
            { "station_ID": "840089002698779", "location": "Boulder"},
            { "station_ID": "840089002647467", "location": "Boulder"},
            { "station_ID": "840089003010690", "location": "Boulder"},
            { "station_ID": "840089003010745", "location": "Boulder"},
            { "station_ID": "840089001727535", "location": "Boulder"},
            { "station_ID": "840089001992241", "location": "Boulder"},
            { "station_ID": "840089002746599", "location": "Boulder"},
            { "station_ID": "840089001723084", "location": "Boulder"},
            { "station_ID": "840069020352930", "location": "LA"},
            { "station_ID": "840069009474542", "location": "LA"}, 
            { "station_ID": "840069009609364", "location": "LA"},
            { "station_ID": "840069021372685", "location": "LA"},
            { "station_ID": "840069021372931", "location": "LA"},
            { "station_ID": "840069009630389", "location": "LA"},
            { "station_ID": "840069009583685", "location": "LA"},
            { "station_ID": "840069018354929", "location": "LA"},
            { "station_ID": "840069021378602", "location": "LA"},
            { "station_ID": "840069018355046", "location": "LA"},
            { "station_ID": "840069018355167", "location": "LA"},
            { "station_ID": "840069019889255", "location": "LA"},
            { "station_ID": "840069021370446", "location": "LA"},
            { "station_ID": "840069015630309", "location": "LA"},
            { "station_ID": "840069009561765", "location": "LA"},
            { "station_ID": "840069018355237", "location": "LA"},
            { "station_ID": "840069021370006", "location": "LA"},
            { "station_ID": "840069009475214", "location": "LA"},
            { "station_ID": "840069019889438", "location": "LA"},
            { "station_ID": "840069019889203", "location": "LA"},
            { "station_ID": "840069018355186", "location": "LA"},
            { "station_ID": "840069009598043", "location": "LA"},
            { "station_ID": "840069021378553", "location": "LA"},
            { "station_ID": "840069009526894", "location": "LA"},
            { "station_ID": "840069019888952", "location": "LA"},
            { "station_ID": "840069020353168", "location": "LA"},
            { "station_ID": "840069018353574", "location": "LA"},
            { "station_ID": "840069018353718", "location": "LA"},
            { "station_ID": "840069020353001", "location": "LA"},
            { "station_ID": "840069009614065", "location": "LA"},
            { "station_ID": "840069020353170", "location": "LA"},
            { "station_ID": "840069018353747", "location": "LA"},
            { "station_ID": "840069009642384", "location": "LA"},
            { "station_ID": "840069009510227", "location": "LA"},
            { "station_ID": "840069021368382", "location": "LA"},
            { "station_ID": "840069015610101", "location": "LA"},
            { "station_ID": "840069021372855", "location": "LA"},
            { "station_ID": "840069020353509", "location": "LA"},
            { "station_ID": "840069020352808", "location": "LA"},
            { "station_ID": "840069009500335", "location": "LA"},
            { "station_ID": "840069021369999", "location": "LA"},
            { "station_ID": "840069019888910", "location": "LA"},
            { "station_ID": "840069015625303", "location": "LA"},
            { "station_ID": "840069009534730", "location": "LA"},
            { "station_ID": "840069009577855", "location": "LA"},
            { "station_ID": "840069021368814", "location": "LA"},
            { "station_ID": "840069021367983", "location": "LA"},
            { "station_ID": "840069015664372", "location": "LA"},
            { "station_ID": "840069009508130", "location": "LA"},
            { "station_ID": "840069018354060", "location": "LA"},
            { "station_ID": "840069015640346", "location": "LA"},
            { "station_ID": "840069009483545", "location": "LA"},
            { "station_ID": "840069021377385", "location": "LA"},
            { "station_ID": "840069021369034", "location": "LA"},
            { "station_ID": "840069019888869", "location": "LA"},
            { "station_ID": "840069015609793", "location": "LA"},
            { "station_ID": "840069019889982", "location": "LA"},
            { "station_ID": "840069009524407", "location": "LA"},
            { "station_ID": "840069009477304", "location": "LA"},
            { "station_ID": "840069009626174", "location": "LA"},
            { "station_ID": "840069021368039", "location": "LA"},
            { "station_ID": "840069019889867", "location": "LA"},
            { "station_ID": "840069009553063", "location": "LA"},
            { "station_ID": "840069018353652", "location": "LA"},
            { "station_ID": "840069021369929", "location": "LA"},
            { "station_ID": "840069018353914", "location": "LA"},
            { "station_ID": "840069009508214", "location": "LA"},
            { "station_ID": "840069009494643", "location": "Palo Alto"},
            { "station_ID": "840069021369357", "location": "Palo Alto"},
            { "station_ID": "840069020352619", "location": "Palo Alto"},
            { "station_ID": "840069021378220", "location": "Palo Alto"},
            { "station_ID": "840069009491489", "location": "Palo Alto"},
            { "station_ID": "840069009583497", "location": "Palo Alto"},
            { "station_ID": "840069009486667", "location": "Palo Alto"},
            { "station_ID": "840069009593149", "location": "Palo Alto"},
            { "station_ID": "840069009468619", "location": "Palo Alto"},
            { "station_ID": "840069021368469", "location": "Palo Alto"},
            { "station_ID": "840069009502332", "location": "Palo Alto"},
#             { "station_ID": "840069021367995", "location": "Palo Alto"}, # FB
#             { "station_ID": "840069021372284", "location": "Palo Alto"}, # FB
#             { "station_ID": "840069021372285", "location": "Palo Alto"}, # FB
#             { "station_ID": "840069021370487", "location": "Palo Alto"}, # FB
#             { "station_ID": "840068000048250", "location": "Palo Alto"}, # FB
#             { "station_ID": "840069018354282", "location": "Palo Alto"}, # FB
            { "station_ID": "840069015633623", "location": "Palo Alto"},
#             { "station_ID": "840069021372793", "location": "Palo Alto"}, # FB
            { "station_ID": "840069019890146", "location": "Palo Alto"},
            { "station_ID": "840069021371883", "location": "Palo Alto"},
            { "station_ID": "840069009528970", "location": "Palo Alto"},
            { "station_ID": "840069020353033", "location": "Palo Alto"},
            { "station_ID": "840069021378425", "location": "Palo Alto"},
            { "station_ID": "840089003012307", "location": "Boulder"},
            { "station_ID": "840519002508624", "location": "Northern Virginia"},
            { "station_ID": "840519002562740", "location": "Northern Virginia"},
            { "station_ID": "840519002508551", "location": "Northern Virginia"},
            { "station_ID": "840519002897042", "location": "Northern Virginia"},
            { "station_ID": "840519002897055", "location": "Northern Virginia"},
            { "station_ID": "840519002612711", "location": "Northern Virginia"},
            { "station_ID": "840519002100453", "location": "Northern Virginia"},
            { "station_ID": "840519002066840", "location": "Northern Virginia"},
            { "station_ID": "840519002258599", "location": "Northern Virginia"},
            { "station_ID": "840519002087121", "location": "Northern Virginia"},
            { "station_ID": "840519002344023", "location": "Northern Virginia"},
            { "station_ID": "840519002897047", "location": "Northern Virginia"},
            { "station_ID": "840519002508625", "location": "Northern Virginia"},
            { "station_ID": "840519002224657", "location": "Northern Virginia"},
            { "station_ID": "840519002080286", "location": "Northern Virginia"},
            { "station_ID": "840519002086370", "location": "Northern Virginia"},
            { "station_ID": "840519002344118", "location": "Northern Virginia"},
            { "station_ID": "840519002344042", "location": "Northern Virginia"},
            { "station_ID": "840519002612670", "location": "Northern Virginia"},
            { "station_ID": "840519002562758", "location": "Northern Virginia"},
            { "station_ID": "840119000473347", "location": "Northern Virginia"},
            { "station_ID": "840519002343985", "location": "Northern Virginia"}]
len(stations)

# COMMAND ----------

num_api_keys = 12
requests_per_key_per_day = 2500
request_limit = requests_per_key_per_day * num_api_keys

runs_per_hour = 6 ## Every 10 mins
requests_per_run = len(stations)
requests_per_hour = requests_per_run * runs_per_hour
requests_per_day = requests_per_hour * 24

request_differential = request_limit - requests_per_day
request_differential 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions

# COMMAND ----------

def get_proxy():
    proxies = None
    while proxies == None:
        try:
            ## Retrieve a new proxy
            print("Attempting to find proxy.")
            proxy = FreeProxy(rand=True, https=True).get()
            proxies = {'https': proxy}
            print(proxies)

            ## Test the new proxy
            print("Testing proxy.")
            resp = requests.get("https://tomtom.com", headers=headers, verify=False, proxies=proxies)
            if not resp.ok:
                proxies = None
                print("Proxy failed.")

        except Exception as e:
            print(f'Failed to find proxy: {str(e)}')
            proxies = None
            print("Trying again.")
        finally:
            time.sleep(1.5)
    print("Done.")
    return proxies

# COMMAND ----------

def get_stations(location, proxies):
    attempt_count = 0 
    station_results = None
    while station_results == None and attempt_count <= 5:
        try:
            ## Try no more than than the attempt limit
            attempt_count +=1 

            ## Set the location
            station_parameters.update(locations_parameters[location])
            
            ## 
            print("Attempting to retrieve stations")
            resp = requests.get('https://api.tomtom.com/search/2/poiSearch/ev+charging+station.json', params=station_parameters, headers=headers, verify=False, proxies=proxies)
            resp_json = resp.json()
            station_results = resp_json['results']

        except Exception as e:
            print(f"There was an issue retrieving charging stations: \n {e}")
            print("-"*50)
        finally:
            ## Don't spam the API
            time.sleep(1)
    
    print("Stations retrieved.")
    print(station_results)
    return station_results

# COMMAND ----------

def get_station_availability_from_list(stations, proxies):
  
    ## Loop through every station in our results
    station_count = 1
    station_connectors = []
    attempt_count = 0
    for station in stations:

        ## Make some attempts at each step
        availability_results = None
        while availability_results == None and attempt_count <= 10:
            try:    

                if attempt_count > 3:
                    proxies = None
                    print("REMOVED PROXIES")
                    
                ## Get the current station ID
                station_id = str(station["station_ID"])
                print(f"Station ID: {station_id} - station {station_count} of {len(stations)}")
                availability_parameters.update({"chargingAvailability": station_id})    

                ## Make the request for the current station's availability
                resp = requests.get(f'https://api.tomtom.com/search/2/chargingAvailability.json', params=availability_parameters, headers=headers, verify=False, proxies=proxies, timeout=60)
                availability_results = resp.json()["connectors"]
                

                ## Add in a timestamp and location
                for availability_result in availability_results:
                    availability_result["timestamp"] = dt.datetime.now()
                    availability_result["location"] = station["location"]
                    availability_result["station_id"] = str(station_id)

                ## Appened to our list of all results
                station_connectors.extend(availability_results)
                station_count += 1

            except Exception as e:
                 ## Try no more than than the attempt limit
                attempt_count +=1 
                print(f"There was an issue retrieving charging stations: \n {e}")
                print("-"*50)
            finally:
                ## Don't spam the API
                time.sleep(0.5)

    print("Availabilities retrieved.")
    print(station_connectors)
    return station_connectors

# COMMAND ----------

def get_station_availability_from_results(station_results, proxies):
  
    ## Loop through every station in our results
    station_count = 1
    station_connectors = []
    attempt_count = 0
    for i in range(len(station_results)):

        ## Make some attempts at each step
        availability_results = None
        while availability_results == None and attempt_count <= 10:
            try:    

                if attempt_count > 3:
                    proxies = None
                    print("REMOVED PROXIES")

                ## Get the current station ID
                station_id = station_results[i]['id']
                print(f"Station ID: {station_id} - station {station_count} of {len(station_results)}")
                availability_parameters.update({"chargingAvailability": station_id})    

                ## Make the request for the current station's availability
                resp = requests.get(f'https://api.tomtom.com/search/2/chargingAvailability.json', params=availability_parameters, headers=headers, verify=False, proxies=proxies, timeout=60)
                availability_results = resp.json()["connectors"]

                ## Add in a timestamp and location
                for availability_result in availability_results:
                    availability_result["timestamp"] = dt.datetime.now()
                    availability_result["location"] = "UCI"
                    availability_result["station_id"] = station_id

                ## Appened to our list of all results
                station_connectors.extend(availability_results)
                station_count += 1

            except Exception as e:
                 ## Try no more than than the attempt limit
                attempt_count +=1 
                print(f"There was an issue retrieving charging stations: \n {e}")
                print("-"*50)
            finally:
                ## Don't spam the API
                time.sleep(0.5)

    print("Availabilities retrieved.")
    print(station_connectors)
    return station_connectors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execution

# COMMAND ----------

## Retrieve a proxy
use_proxies = False
proxies = None
if use_proxies:
    proxies = get_proxy()
    
from_locations = False ## If false, will use list instead

## Get the current time
us_pacific_time = convert_datetime_timezone(dt.datetime.now(), "UTC", "US/Pacific")

if from_locations == False:
    print("Getting results from stations list.")
    ## Then, get the station avalability for the given stations
    station_connectors = get_station_availability_from_list(stations, proxies)

    ## Store the data into a dataframe
    availability_results_pdf = json_normalize(station_connectors)

    ## Add a date column
    availability_results_pdf["date"] = availability_results_pdf["timestamp"].dt.date
    availability_results_pdf["hour"] = availability_results_pdf["timestamp"].dt.hour
    availability_results_pdf["minute"] = availability_results_pdf["timestamp"].dt.minute

    ## Convert to a spark dataframe
    availability_results_df = spark.createDataFrame(availability_results_pdf) 
    display(availability_results_df)
    
    ## Write to Azure ADLS
    if False:
        (availability_results_df
            .repartition(1)
            .write
            .format("parquet")
            .mode("append")
            .save(blob_url + "/tomtom_10min_data"))
    
#     df = spark.read.parquet(f"{blob_url}/tomtom_10min_data")
    
    ## Write to AWS S3
    (availability_results_df
        .repartition(1)
        .write
        .format("parquet")
        .mode("append")
        .save(f"/mnt/{mount_name}/tomtom_10min_data"))
    
else:
    print("Getting results from locations list.")
    locations = ["UCI", "UCB", "DTLA", "NYC"]
    for location in locations:
        ## Next, attempt to retrieve the stations for the given location
        station_results = get_stations(location, proxies)

        ## Then, get the station avalability for the given stations
        station_connectors = get_station_availability_from_results(station_results, proxies)

        ## Store the data into a dataframe
        availability_results_pdf = json_normalize(station_connectors)

        ## Add a date column
        availability_results_pdf["date"] = availability_results_pdf["timestamp"].dt.date
        availability_results_pdf["hour"] = availability_results_pdf["timestamp"].dt.hour
        availability_results_pdf["minute"] = availability_results_pdf["timestamp"].dt.minute

        ## Convert to a spark dataframe
        availability_results_df = spark.createDataFrame(availability_results_pdf) 
        display(availability_results_df)

        (availability_results_df
            .write
            .format("parquet")
            .mode("append")
            .save(blob_url + "/tomtom_full"))