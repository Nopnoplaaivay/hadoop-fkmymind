# ========== spark-conf/spark-env.sh ==========
#!/usr/bin/env bash

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export SPARK_MASTER_HOST=spark-master
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
export SPARK_WORKER_CORES=2
export SPARK_WORKER_MEMORY=2g
export SPARK_WORKER_PORT=8881
export SPARK_WORKER_WEBUI_PORT=8081
export SPARK_LOCAL_IP=0.0.0.0
export HADOOP_CONF_DIR=/home/spark/hadoop-conf
export YARN_CONF_DIR=/home/spark/hadoop-conf
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3