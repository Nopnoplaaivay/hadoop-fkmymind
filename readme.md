docker build -t hadoop-multi:latest -f Dockerfile.multi .
docker-compose up -d
docker exec namenode hdfs dfsadmin -report