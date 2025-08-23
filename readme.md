# setup hadoop
docker build -t hadoop-multi:latest -f Dockerfile.hadoop.multi .
docker-compose -f docker-compose.hadoop.yml -p spark_hadoop up -d
docker exec namenode hdfs dfsadmin -report
docker network ls | findstr hadoop
bash ./setup-hadoop-multi.sh


# setup spark
docker-compose -f docker-compose.spark.yml -p spark_hadoop up -d
docker-compose -f docker-compose.spark.yml -p spark_hadoop down
docker exec spark-master /home/spark/spark/bin/spark-submit --version
bash ./setup-spark-standalone.sh


# Copy to spark-master
docker cp spark-standalone/notebooks/pyspark_hadoop_examples.py spark-master:/home/spark/notebooks/
docker cp spark-standalone/notebooks/test_spark.py spark-master:/home/spark/notebooks/

docker exec -it spark-master /bin/bash
docker exec spark-master spark-submit --master spark://spark-master:7077 --deploy-mode client /home/spark/notebooks/pyspark_hadoop_examples.py
