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
from datetime import timedelta
import pytz
from fp.fp import FreeProxy
import pyspark.sql.functions as F
import warnings
import matplotlib.pyplot as plt
import math

## Settings
warnings.filterwarnings("ignore")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
sns.set(rc={'figure.figsize':(16,9)})

blob_container = "w210" # The name of your container created in https://portal.azure.com
storage_account = "w210" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w210-scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "azure-storage-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

secret_scope = "w210-scope"
GOOGLE_API_KEY = dbutils.secrets.get(scope=secret_scope, key="google-api-key")

def flatten(t):
    return [item for sublist in t for item in sublist]

# COMMAND ----------

places = [{address: "419 W College St, Los Angeles, CA 90012","latitude": 34.06482, "longitude": -118.23814},
{address: "816 Bartlett St, Los Angeles, CA 90012","latitude": 34.06267, "longitude": -118.24507},
{address: "200 N San Fernando Rd, Los Angeles, CA 90031","latitude": 34.07688, "longitude": -118.22205},
{address: "1910 W Temple St, Los Angeles, CA 90026","latitude": 34.06898, "longitude": -118.26621},
{address: "1440 E Imperial Ave, El Segundo, CA 90245","latitude": 33.93057, "longitude": -118.40074},
{address: "Los Angeles, CA 90245","latitude": 33.92996, "longitude": -118.39709},
{address: "9045 Lincoln Blvd, Los Angeles, CA 90045","latitude": 33.95631, "longitude": -118.41732},
{address: "9045 Lincoln Blvd, Los Angeles, CA 90045","latitude": 33.95631, "longitude": -118.41732},
{address: "El Segundo, CA 90245","latitude": 33.92857, "longitude": -118.39534},
{address: "639 Pacific Coast Hwy, El Segundo, CA 90245","latitude": 33.92488, "longitude": -118.39662},
{address: "100-198 E Holly Ave, El Segundo, CA 90245","latitude": 33.92084, "longitude": -118.41494},
{address: "331 Pacific Coast Hwy, El Segundo, CA 90245","latitude": 33.92067, "longitude": -118.39671},
{address: "750 N Nash St, El Segundo, CA 90245","latitude": 33.92625, "longitude": -118.38702},
{address: "2199 Campus Dr, El Segundo, CA 90245","latitude": 33.92485, "longitude": -118.38571},
{address: "101 Pacific Coast Hwy, El Segundo, CA 90245","latitude": 33.91677, "longitude": -118.39722},
{address: "1 LMU Dr, Los Angeles, CA 90045","latitude": 33.96759, "longitude": -118.42176},
{address: "1945 Ignatian Cir, Los Angeles, CA 90045","latitude": 33.96925, "longitude": -118.41555},
{address: "11622 Aviation Blvd, Del Aire, CA 90304","latitude": 33.92882, "longitude": -118.37778},
{address: "2201 E El Segundo Blvd, El Segundo, CA 90245","latitude": 33.9169, "longitude": -118.38652},
{address: "626 Isis Ave, Inglewood, CA 90301","latitude": 33.95571, "longitude": -118.37494},
{address: "5220 Pacific Concourse Dr, Los Angeles, CA 90045","latitude": 33.92715, "longitude": -118.37323},
{address: "5245 Pacific Concourse Dr, Los Angeles, CA 90045","latitude": 33.92833, "longitude": -118.37249},
{address: "8405 Pershing Dr, Los Angeles, CA 90293","latitude": 33.95709, "longitude": -118.44327},
{address: "5625 Crescent Park W, Playa Vista, CA 90094","latitude": 33.9715, "longitude": -118.42922},
{address: "13031 W Jefferson Blvd, Playa Vista, CA 90094","latitude": 33.97541, "longitude": -118.42618},
{address: "6060 Center Dr, Los Angeles, CA 90045","latitude": 33.97575, "longitude": -118.39114},
{address: "6080 Center Dr, Los Angeles, CA 90045","latitude": 33.97645, "longitude": -118.39183},
{address: "13141 Fountain Park Dr, Los Angeles, CA 90094","latitude": 33.97452, "longitude": -118.43018},
{address: "12181 W Bluff Creek Dr, Los Angeles, CA 90094","latitude": 33.97906, "longitude": -118.40606},
{address: "12777 W Jefferson Blvd, Los Angeles, CA 90066","latitude": 33.97842, "longitude": -118.41816},
{address: "12150 Millennium Dr, Playa Vista, CA 90094","latitude": 33.97933, "longitude": -118.40654},
{address: "6100 Center Dr, Los Angeles, CA 90045","latitude": 33.97704, "longitude": -118.39242},
{address: "12655 Beatrice St, Los Angeles, CA 90066","latitude": 33.97972, "longitude": -118.4189},
{address: "6701 Center Dr W, Los Angeles, CA 90045","latitude": 33.97892, "longitude": -118.39375},
{address: "6700 Center Dr W, Los Angeles, CA 90045","latitude": 33.9792, "longitude": -118.39304},
{address: "6601 Center Dr W, Los Angeles, CA 90045","latitude": 33.97996, "longitude": -118.39398},
{address: "2145 Park Pl, El Segundo, CA 90245","latitude": 33.90573, "longitude": -118.38645},
{address: "2135-2141 Park Pl, El Segundo, CA 90245","latitude": 33.90534, "longitude": -118.38712},
{address: "2120 Park Pl, El Segundo, CA 90245","latitude": 33.9045, "longitude": -118.38724},
{address: "800 Apollo St, El Segundo, CA 90245","latitude": 33.90464, "longitude": -118.38587},
{address: "12035 Waterfront Dr, Playa Vista, CA 90094","latitude": 33.98296, "longitude": -118.40472},
{address: "1180 Rosecrans Ave, El Segundo, CA 90266","latitude": 33.90157, "longitude": -118.39398},
{address: "1230 Rosecrans Ave, Manhattan Beach, CA 90266","latitude": 33.90144, "longitude": -118.39217},
{address: "777 S Aviation Blvd, Hawthorne, CA 90245","latitude": 33.90571, "longitude": -118.37878},
{address: "3160 N Sepulveda Blvd, Manhattan Beach, CA 90266","latitude": 33.89872, "longitude": -118.39436},
{address: "111 N Rengstorff Ave, Mountain View, CA 94043","latitude": 37.404, "longitude": -122.09547},
{address: "2260 W El Camino Real, Los Altos, CA 94040","latitude": 37.3971, "longitude": -122.10425},
{address: "W El Camino Real, Los Altos, CA 94040","latitude": 37.39724, "longitude": -122.10498},
{address: "4880 El Camino Real, Los Altos, CA 94022","latitude": 37.39836, "longitude": -122.10918},
{address: "750 Moffett Blvd, Mountain View, CA 94043","latitude": 37.40641, "longitude": -122.06832},
{address: "1984 W El Camino Real, Mountain View, CA 94040","latitude": 37.39482, "longitude": -122.09892},
{address: "Mountain View, CA 94043","latitude": 37.40928, "longitude": -122.06362},
{address: "101-199 Stierlin Rd, Mountain View, CA 94043","latitude": 37.39612, "longitude": -122.07788},
{address: "511 Central Ave, Mountain View, CA 94043","latitude": 37.39623, "longitude": -122.07566},
{address: "1571 W El Camino Real, Mountain View, CA 94040","latitude": 37.38899, "longitude": -122.09236},
{address: "1328 W El Camino Real, Mountain View, CA 94040","latitude": 37.38879, "longitude": -122.08942},
{address: "693 Arastradero Rd, Palo Alto, CA 94306","latitude": 37.40104, "longitude": -122.12926},
{address: "150 W Evelyn Ave, Mountain View, CA 94041","latitude": 37.3924, "longitude": -122.07184},
{address: "819 W El Camino Real, Mountain View, CA 94040","latitude": 37.38565, "longitude": -122.08492},
{address: "4005 Miranda Ave, Palo Alto, CA 94304","latitude": 37.40112, "longitude": -122.13671},
{address: "93-111 W El Camino Real, Mountain View, CA 94040-2603","latitude": 37.38201, "longitude": -122.07662},
{address: "15 Hacker Way, Menlo Park, CA 94025","latitude": 37.48392, "longitude": -122.14769},
{address: "Menlo Park, CA 94025","latitude": 37.48077, "longitude": -122.15727},
{address: "1 Facebook Way, Menlo Park, CA 94025","latitude": 37.48099, "longitude": -122.16023},
{address: "Menlo Park, CA 94025","latitude": 37.48086, "longitude": -122.16696},
{address: "180 Jefferson Dr, Menlo Park, CA 94025","latitude": 37.481, "longitude": -122.17026},
{address: "281 Jefferson Dr, Menlo Park, CA 94025","latitude": 37.48165, "longitude": -122.17055},
{address: "Menlo Park, CA 94025","latitude": 37.47954, "longitude": -122.17495},
{address: "135 Commonwealth Dr, Menlo Park, CA 94025","latitude": 37.48174, "longitude": -122.17518},
{address: "155 Constitution Dr, Menlo Park, CA 94025","latitude": 37.48492, "longitude": -122.17507},
{address: "120 Independence Dr, Menlo Park, CA 94025","latitude": 37.48297, "longitude": -122.17798},
{address: "740 Serra St, Stanford, CA 94305","latitude": 37.42732, "longitude": -122.15741},
{address: "181 Encinal Ave, Atherton, CA 94025","latitude": 37.46339, "longitude": -122.18647},
{address: "105 Constitution Dr, Menlo Park, CA 94025","latitude": 37.48528, "longitude": -122.17811},
{address: "3465 Haven Ave, Menlo Park, CA 94025","latitude": 37.48591, "longitude": -122.18238},
{address: "3465 Haven Ave, Menlo Park, CA 94025","latitude": 37.48696, "longitude": -122.18178},
{address: "3645 Haven Ave, Menlo Park, CA 94025","latitude": 37.48572, "longitude": -122.18288},
{address: "3465 Haven Ave, Menlo Park, CA 94025","latitude": 37.48591, "longitude": -122.1834},
{address: "6500 Arapahoe Rd, Boulder, CO 80303","latitude": 40.0121, "longitude": -105.20125},
{address: "4571 Broadway St, Boulder, CO 80304","latitude": 40.05766, "longitude": -105.28281},
{address: "5565 51st St, Boulder, CO 80301","latitude": 40.07318, "longitude": -105.23713},
{address: "6255 Habitat Dr, Boulder, CO 80301","latitude": 40.06099, "longitude": -105.20822},
{address: "7900 Tysons One Pl, Mc Lean, VA 22102","latitude": 38.91939, "longitude": -77.21965},
{address: "2000 Corporate Rdg, Mc Lean, VA 22102","latitude": 38.91312, "longitude": -77.21644},
{address: "8251 Greensboro Dr, Mc Lean, VA 22102","latitude": 38.92168, "longitude": -77.23003},
{address: "1750 Tysons Blvd, Mc Lean, VA 22102","latitude": 38.92367, "longitude": -77.22291},
{address: "1650 Tysons Blvd, Mc Lean, VA 22102","latitude": 38.92526, "longitude": -77.22371},
{address: "8486 Westpark Dr, Tysons, VA 22102","latitude": 38.92394, "longitude": -77.23623},
{address: "1803 Capital One Dr, Tysons, VA 22102","latitude": 38.92556, "longitude": -77.21275},
{address: "8360 Greensboro Dr, Mc Lean, VA 22102","latitude": 38.92755, "longitude": -77.23428},
{address: "8421 Broad St, Mc Lean, VA 22102","latitude": 38.92761, "longitude": -77.23811},
{address: "7596 Colshire Dr, Mc Lean, VA 22102","latitude": 38.92367, "longitude": -77.20571},
{address: "7515 Colshire Dr, Mc Lean, VA 22102","latitude": 38.9218, "longitude": -77.20415},
{address: "1575 Anderson Rd, Mc Lean, VA 22102","latitude": 38.92678, "longitude": -77.20511},
{address: "8614 Westwood Center Dr, Vienna, VA 22182","latitude": 38.92974, "longitude": -77.247},
{address: "145 District Ave, Merrifield, VA 22031","latitude": 38.87267, "longitude": -77.22958},
{address: "10700 Parkridge Blvd, Reston, VA 20191","latitude": 38.94354, "longitude": -77.31344},
{address: "101 MGM National Ave, Oxon Hill, MD 20745","latitude": 38.79512, "longitude": -77.00987},
{address: "6800 Oxon Hill Rd, Oxon Hill, MD 20745","latitude": 38.7958, "longitude": -77.00111},
{address: "2399 Jefferson Davis Hwy, Arlington, VA 22202","latitude": 38.85215, "longitude": -77.05201},
{address: "1919 S Eads St, Arlington, VA 22202","latitude": 38.8563, "longitude": -77.05277},
{address: "642 15th St S, Arlington, VA 22202","latitude": 38.85941, "longitude": -77.05812},
{address: "1400 S Joyce St, Arlington, VA 22202","latitude": 38.86187, "longitude": -77.06472},
{address: "200 12th St S, Arlington, VA 22202","latitude": 38.86233, "longitude": -77.04942},
{address: "1100 S Hayes St, Arlington, VA 22202","latitude": 38.86334, "longitude": -77.0605},
{address: "400 Army Navy Dr, Arlington, VA 22202","latitude": 38.86429, "longitude": -77.05345},
{address: "1101 S Joyce St, Arlington, VA 22202","latitude": 38.86464, "longitude": -77.06291},
{address: "69 Q St SW, Washington, DC 20024","latitude": 38.87095, "longitude": -77.01131},
{address: "79 Potomac Ave SE, Washington, DC 20003","latitude": 38.87122, "longitude": -77.00666},
{address: "2201 N Pershing Dr, Arlington, VA 22201","latitude": 38.88092, "longitude": -77.08547}]

# COMMAND ----------

for i in range(len(places)):
    
    ## Get the current place
    place = places[i]
    lat, long = place["latitude"], place["longitude"]
    
    ## Construct the request url
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{long}&radius=30&key={GOOGLE_API_KEY}"
    resp = requests.get(url) 
    
    ## Extract the places from the result
    place_types = list(set(flatten([result["types"] for result in resp.json()["results"]])))
    print(place_types)
    
    ## Append the results to the places data
    places[i]["types"] = place_types

# COMMAND ----------

resp.json()

# COMMAND ----------

places_df = spark.createDataFrame(pd.DataFrame(places))
display(places_df)

# COMMAND ----------

# DBTITLE 1,Get Place IDs by address then find type(s) (doesn't work well)
address_to_place_id = {}
if False:
    for place in places:
        input_query = place["address"]
        url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?fields=formatted_address%2Cname%2Cplace_id&input={input_query}&inputtype=textquery&key={GOOGLE_API_KEY}"
        resp = requests.get(url) 
        print(resp.json())
        address_to_place_id[address] = resp.json()["candidates"][0]["place_id"]
    #     time.sleep(0.5)

    address_to_type = {}
    for address in address_to_place_id:
        place_id = address_to_place_id[address]
        url = f"https://maps.googleapis.com/maps/api/place/details/json?fields=types&place_id={place_id}&key={GOOGLE_API_KEY}"
        resp = requests.get(url) 
        print(resp.json())
        address_to_type[address] = resp.json()["result"]["types"]
address_to_place_id