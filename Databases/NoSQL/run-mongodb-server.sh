#!/bin/sh

set -eu

sudo apt update
sudo apt install -y mongodb

for conda_env in Python2 Python3; do
    /opt/anaconda/envs/$conda_env/bin/conda install -y pymongo
done

sudo mkdir -p /data/db
sudo mongod --quiet --fork --logpath /var/log/mongod.log
