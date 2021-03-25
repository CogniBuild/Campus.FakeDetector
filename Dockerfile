FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install cmake
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENV_PRODUCTION_MODE='True'
ENV ENV_APPLICATION_SECRET=''
ENV ENV_PROTOCOL_SECRET=''
ENV ENV_DETECTOR_KERNEL_FILE_PATH=''
ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["gunicorn", "-b", "127.0.0.1:5000", "web:app"]
