#! /bin/bash

sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install postgresql-9.5 -y
sudo service postgresql start
sudo -u postgres psql -c "create user sherlock"
sudo -u postgres psql -c "create schema input"
sudo -u postgres psql -c "grant all on schema input to sherlock"
sudo -u postgres psql -c "grant all on all tables in schema input to sherlock"
