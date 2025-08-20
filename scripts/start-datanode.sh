#!/bin/bash
sudo service ssh start

# Wait for SSH and NameNode to be ready
sleep 10

# Wait for namenode to be available
while ! nc -z namenode 9000; do
    echo "Waiting for NameNode to be available..."
    sleep 2
done

echo "Starting DataNode..."
hdfs datanode