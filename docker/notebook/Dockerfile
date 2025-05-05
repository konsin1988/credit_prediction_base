FROM jupyter/pyspark-notebook:latest

WORKDIR /notebook

COPY ./docker/notebook/requirements.txt /tmp/requirements.txt
# COPY ./.env /.env

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/* 

RUN pip3 install -r /tmp/requirements.txt