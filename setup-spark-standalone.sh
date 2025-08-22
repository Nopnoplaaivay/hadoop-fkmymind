#!/bin/bash

echo "=== Setting up Spark with Hadoop Integration ==="
echo ""

docker build -t spark-hadoop:latest -f Dockerfile.spark .

# Create directories in HDFS for Spark
echo "Creating HDFS directories for Spark..."
docker exec namenode hdfs dfs -mkdir -p /spark-logs
docker exec namenode hdfs dfs -mkdir -p /spark-jars
docker exec namenode hdfs dfs -chmod 777 /spark-logs
docker exec namenode hdfs dfs -chmod 777 /spark-jars

# Start Spark cluster
echo "Starting Spark cluster..."
docker-compose -f docker-compose.spark.yml -p spark_hadoop up -d

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 20

# Verify Spark cluster
echo ""
echo "=== Spark Cluster Status ==="
docker exec spark-master /home/spark/spark/bin/spark-submit --version

echo ""
echo "=== Cluster Information ==="
echo "Spark Master UI: http://localhost:8080"
echo "Spark Worker 1 UI: http://localhost:8081"
echo "Spark Worker 2 UI: http://localhost:8082"
echo "Jupyter Notebook: http://localhost:8888"
echo "Spark History Server: http://localhost:18080"
echo ""
echo "Hadoop NameNode UI: http://localhost:9870"
echo "YARN ResourceManager UI: http://localhost:8088"
echo ""
echo "Spark cluster setup complete!"