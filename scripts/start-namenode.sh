#!/bin/bash
sudo service ssh start

# Wait for SSH to start
sleep 5

# Format namenode if not already formatted
if [ ! -d "/home/hadoop/hdfs/namenode/current" ]; then
    echo "Formatting namenode..."
    hdfs namenode -format -force
fi

echo "Starting NameNode..."
hdfs namenode