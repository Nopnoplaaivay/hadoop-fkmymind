#!/bin/bash
set -euo pipefail
sudo service ssh start


# Wait for ResourceManager to be available
while ! nc -z resourcemanager 8032; do
    echo "Waiting for ResourceManager..."
    sleep 2
done

echo "Starting NodeManager..."
exec yarn nodemanager

