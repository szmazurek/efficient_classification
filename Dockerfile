FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update

RUN apt-get install -y wget && apt-get install -y curl &&\
    apt-get install -y software-properties-common && \
    apt-get install git -y && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && \ 
    apt-get install -y python3.11 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/szmazurek/efficient_classification.git
WORKDIR  /efficient_classification
RUN pip3.11 install -r requirements.txt
RUN mkdir checkpoints/
ENTRYPOINT ["python3.11", "src/main.py"]