# Databricks notebook source
# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2

# COMMAND ----------

# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC echo """broker.id=2
# MAGIC listeners=PLAINTEXT://:9094
# MAGIC log.dirs=/tmp/kafka-logs2""" >> config/server.2.properties

# COMMAND ----------

# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC echo """broker.id=3
# MAGIC listeners=PLAINTEXT://:9095
# MAGIC log.dirs=/tmp/kafka-logs3""" >> config/server.3.properties

# COMMAND ----------

# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC mkdir /tmp/kafka-logs2
# MAGIC mkdir /tmp/kafka-logs3

# COMMAND ----------

# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC cp config/server.properties config/server.1.properties
# MAGIC cp config/server.properties config/server.2.properties
# MAGIC cp config/server.properties config/server.3.properties
# MAGIC ls config/