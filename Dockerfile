FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.10 python3.10-distutils python3-pip timidity curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/timidity /usr/local/bin/timidity

# Install pip directly from the official source
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

RUN python3.10 -m pip install --upgrade pip \
    && pip3 install tox tox-gh-actions poetry

WORKDIR /home/runner/work/qmuvi/qmuvi

