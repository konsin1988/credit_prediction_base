FROM python:3.11-slim

WORKDIR /app

COPY model_creator.py /app/model_creator.py
COPY ./requirements.txt /tmp/requirements.txt 
COPY .env .env

RUN apt-get update && apt-get install -y \
build-essential \
libpq-dev \
curl \
&& rm -rf /var/lib/apt/lists/* 

RUN pip install -r /tmp/requirements.txt

CMD ["python3", "model_creator.py"]
