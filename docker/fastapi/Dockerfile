FROM python:3.11-slim

WORKDIR /fastapi

COPY ./docker/fastapi/requirements.txt ./requirements.txt
# COPY ./.env ./.env

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* 

RUN pip3 install -r requirements.txt

CMD ["fastapi", "run", "src/app.py", "--reload", "--port", "8000"]