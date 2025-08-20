echo "Cleaning up Hadoop cluster..."
docker-compose down -v
docker rmi hadoop-multi:latest
rm -rf data/*
echo "Cleanup complete."