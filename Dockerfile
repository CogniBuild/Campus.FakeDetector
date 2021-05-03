# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8-appservice
FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureWebJobsDisableHomepage=true \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV ENV_DETECTOR_KERNEL_FILE_PATH='.kernels/weights.h5'
ENV ENV_PROBABILITY_THRESHOLD=0.5

RUN apt-get install build-essential cmake -y

COPY requirements-docker.txt /
RUN pip install -r /requirements-docker.txt

COPY . /home/site/wwwroot