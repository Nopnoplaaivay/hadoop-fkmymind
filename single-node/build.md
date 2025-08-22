# Hadoop Docker Setup 

docker build -t hadoop-single .
docker run -d --name hadoop-single-node -p 9870:9870 -p 8088:8088 -p 9000:9000 --hostname localhost hadoop-single
docker exec -it hadoop-single-node bash


- **HDFS NameNode Web UI**: http://localhost:9870
- **YARN ResourceManager Web UI**: http://localhost:8088