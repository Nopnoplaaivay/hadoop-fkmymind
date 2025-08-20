#!/bin/bash
set -euo pipefail
sudo -n service ssh start

# Đợi NameNode sẵn sàng
while ! nc -z namenode 9000; do
  echo "Waiting for NameNode..."
  sleep 2
done

echo "Starting ResourceManager..."
exec yarn resourcemanager