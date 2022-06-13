# syntax=docker/dockerfile:1
FROM python:3.8
WORKDIR .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
