#!/bin/bash
echo "Setting up Hadoop Multi-Node Cluster..."

# Make scripts executable
chmod +x scripts/*.sh

# Build Docker image
echo "Building Hadoop Multi-Node Docker image..."
docker build -t hadoop-multi:latest -f Dockerfile.hadoop.multi .
sleep 5


# Start the cluster
echo "Starting Hadoop cluster..."
docker-compose -f docker-compose.hadoop.yml -p spark_hadoop up -d

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 30

# Check cluster status
echo "Checking cluster status..."
docker exec namenode hdfs dfsadmin -report

echo "================================================"
echo "Hadoop Multi-Node Cluster is running!"
echo "NameNode UI: http://localhost:9870"
echo "ResourceManager UI: http://localhost:8088"
echo "================================================"
