# FastAPI optimized
# https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM python:3.8
ENV APP_HOME /app
WORKDIR ${APP_HOME}
COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY . ./
CMD exec gunicorn --config python:src.gunicorn_config --bind :$PORT src.main:app

# need to add code for FastAPI generic
# insert code here

# Code to use if Flask is being run as main.
# FROM python:3.8
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --trusted-host pypi.python.org -r requirements.txt
# COPY . .
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app