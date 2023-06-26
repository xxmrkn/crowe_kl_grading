FROM python:3.10
USER root

ENV PYTHONUNBUFFERD 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

COPY . /code/

