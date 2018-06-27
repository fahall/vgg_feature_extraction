FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update -y
RUN apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
RUN apt-get install -y wget software-properties-common

RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
RUN apt-get install -y ffmpeg

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update 
RUN apt-get install -y python3.6 python3.6-dev python3.6-tk
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

RUN rm /usr/local/bin/pip3
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
RUN ln -s /usr/local/bin/pip /usr/local/bin/pip3
RUN echo 'alias python="/usr/bin/python3.6"' >> ~/.bashrc

RUN pip3 install --upgrade pip

RUN apt-get -y update
RUN apt-get -y install wget unzip \
                       build-essential cmake git pkg-config libatlas-base-dev gfortran \
                       libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

CMD ["bash"]
