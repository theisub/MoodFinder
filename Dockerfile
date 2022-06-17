# syntax=docker/dockerfile:1
FROM python:3.8
WORKDIR /app
COPY requirements.txt ./app
COPY . /app
RUN pip3 install -r requirements.txt
CMD ["python3", "main.py"]