#!/bin/bash
echo "=== Hadoop Services Health Check ==="
echo

# Check if SSH is running
echo "1. SSH Service Status:"
sudo service ssh status | grep -E "(Active|inactive)"
echo

# Check environment variables
echo "2. Environment Variables:"
echo "JAVA_HOME: $JAVA_HOME"
echo "HADOOP_HOME: $HADOOP_HOME"
echo "PATH contains Hadoop: $(echo $PATH | grep -o 'hadoop' | head -1)"
echo

# Check Java version
echo "3. Java Version:"
java -version 2>&1 | head -1
echo

# Check running Hadoop processes
echo "4. Running Hadoop Processes:"
jps
echo

# Check HDFS status
echo "5. HDFS Health:"
if command -v hdfs &> /dev/null; then
    $HADOOP_HOME/bin/hdfs dfsadmin -report | head -10
else
    echo "HDFS command not found - check HADOOP_HOME"
fi
echo

# Check web UIs accessibility
echo "6. Web UI Status:"
echo "NameNode Web UI: http://localhost:9870"
echo "ResourceManager Web UI: http://localhost:8088"
echo

echo "=== Health Check Complete ==="
