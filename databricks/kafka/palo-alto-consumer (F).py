# Databricks notebook source
# DBTITLE 1,F1
# MAGIC %run "../retrainer_clean"

# COMMAND ----------

# DBTITLE 1,F2
from kafka import KafkaConsumer

## Set the topic for this
topic = "palo-alto"
seed, six_month = palo_alto_seed, palo_alto_6m

S3 = False

stations = [
            'BRYANT',
            'WEBSTER',
            'HAMILTON',
            'HIGH',
            'TED_THOMPSON',
            'MPL',
            'RINCONADA_LIB',
            'CAMBRIDGE'
           ]

station_df_dict = {}
for station in stations:
    station_df_dict[station] = seed[seed["station"] == station]

## Delete streaming data for 
try:
    dbutils.fs.rm(f"/mnt/{mount_name}/streaming/{topic}", True)
except Exception as e:
    print(e)

consumer = KafkaConsumer(topic, bootstrap_servers='localhost:9092')

last_timestamp = None

received_data = pd.DataFrame(data = {'datetime': [],
                                     'station': [],
                                     'ports_available': []})

avg_msg_time, total_message_time, message_count = 0, 0, 0
for message in consumer:
    
    message_count += 1
    message_start = time.time()
    
    ## 1 - Gather new data & decode
    print(f"RAW: {message.key}, {message.value}")
    
    timestamp = message.key.decode().split(',')[1]
    station = message.key.decode().split(',')[0]
    ports_available = float(message.value)
    
    new_data = pd.DataFrame(data = {'datetime': [timestamp],
                                    'station': [station],
                                    'ports_available': [ports_available]})
    new_data['datetime'] = pd.to_datetime(new_data['datetime'])
    
    ## 1.5 - Check for message order
    ## TODO


    ## 2 - Append to history
    if S3:
        path = f"/mnt/{mount_name}/streaming/{topic}/{station}"
        new_data_df = spark.createDataFrame(new_data)
        (new_data_df.repartition(1)
                    .write
                    .format("parquet")
                    .mode("append")
                    .save(path))
    else:
        station_df_dict[station] = station_df_dict[station].append(new_data)
    
    ## Reread in the streaming data thus far
    if ((new_data["datetime"].max().day == 15) and 
        (new_data["datetime"].max().hour == 0) and
        (new_data["datetime"].max().minute == 0)) or (new_data["datetime"].max().minute == 0):
        
        if S3:
            station_df = spark.read.parquet(path)
            station_df = station_df.toPandas()
            print(f"LOADED FROM S3: {message.key}, {message.value}")
        else:
            station_df = station_df_dict[station]
            print(f"LOADED FROM MEMORY: {message.key}, {message.value}")
    
        ## 3 - retrain models separately and periodically
        if ((station_df["datetime"].max().day == 15) and 
            (station_df["datetime"].max().hour == 0) and
            (station_df["datetime"].max().minute == 0)):
            
            start_time = time.time()

            ## Append the history to the stream data
            retrain_df = station_df      
            retrain_df = retrain_df[retrain_df["datetime"] >= retrain_df["datetime"].max() - dt.timedelta(days=90)]
            retrain_df = retrain_df.sort_values(["datetime"])
            retrain_df = retrain_df.reset_index(drop=True)
            train_models(retrain_df)
            
            total_time = round((time.time() - start_time) / 60, 4)

            print(f"\nRetrained Model: {total_time} minutes\n")

        ## 4 - Rerun predictions
        if station_df["datetime"].max().minute == 0:
            
            start_time = time.time()
            predictions_df = make_predictions(station, 
                                              seed, 
                                              station_df, 
                                              six_month)
            total_time = round((time.time() - start_time) / 60, 4)
            
            print(f"\nMade New Predictions: {total_time} mintues\n")
    else:
        print("No New Predictions Needed, Finished Processing")
    
    total_message_time += round(((time.time() - message_start) / 60), 4)
    average_message_time = round(total_message_time / message_count, 4)
    print(f"Average message time: {average_message_time} minutes")