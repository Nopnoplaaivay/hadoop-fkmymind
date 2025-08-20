#!/bin/bash

# Source the Hadoop environment
# source $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Ensure environment variables are set
export HADOOP_HOME=/home/hadoop/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_LOG_DIR=$HADOOP_HOME/logs
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export YARN_LOG_DIR=$HADOOP_HOME/logs


export HADOOP_SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"


# Format the namenode (only if not already formatted)
if [ ! -d "/home/hadoop/hadoop/logs/hadoop" ]; then
    echo "Formatting namenode..."
    $HADOOP_HOME/bin/hdfs namenode -format -force
fi


$HADOOP_HOME/bin/hdfs --daemon start namenode
$HADOOP_HOME/bin/hdfs --daemon start datanode
$HADOOP_HOME/bin/yarn --daemon start resourcemanager
$HADOOP_HOME/bin/yarn --daemon start nodemanager

# Start Hadoop services
# echo "Starting HDFS services..."
# $HADOOP_HOME/sbin/start-dfs.sh

# echo "Starting YARN services..."
# $HADOOP_HOME/sbin/start-yarn.sh

# Show running services
echo "Hadoop services status:"
jps

# Keep container running
echo "All Hadoop services started successfully!"
echo "HDFS NameNode Web UI: http://localhost:9870"
echo "YARN ResourceManager Web UI: http://localhost:8088"
echo "Keeping container alive..."
tail -f /dev/null
