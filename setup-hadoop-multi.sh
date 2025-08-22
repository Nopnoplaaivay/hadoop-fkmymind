#!/bin/bash
echo "Setting up Hadoop Multi-Node Cluster..."

# Create directory structure
mkdir -p config/{namenode,datanode,resourcemanager,nodemanager}
mkdir -p scripts
mkdir -p data

# Make scripts executable
chmod +x scripts/*.sh

# Build Docker image
echo "Building Hadoop Multi-Node Docker image..."
docker build -t hadoop-multi:latest -f Dockerfile.multi .

# Start the cluster
echo "Starting Hadoop cluster..."
docker-compose up -d

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
