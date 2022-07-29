# Databricks notebook source
# DBTITLE 1,C1
# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC echo """broker.id=1
# MAGIC listeners=PLAINTEXT://:9093
# MAGIC log.dirs=/tmp/kafka-logs1""" >> config/server.1.properties

# COMMAND ----------

# DBTITLE 1,C2
# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC cp config/server.properties config/server.1.properties
# MAGIC mkdir /tmp/kafka-logs1

# COMMAND ----------

# DBTITLE 1,C3
# MAGIC %sh
# MAGIC cd kafka_2.12-2.6.2
# MAGIC bin/kafka-server-start.sh config/server.1.properties

# COMMAND ----------

