# FROM rocm/pytorch:latest
FROM rocm/tensorflow:rocm6.4.1-py3.12-tf2.18-dev

# # Install missing deps
USER root
RUN apt update
RUN apt install libjpeg-dev python3-dev python3-pip
RUN pip3 install wheel setuptools
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/

# # Copy python package requirements
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade pipenv
COPY requirements.txt .
RUN pip install -r requirements.txt

#Finalizing installation
# CMD /bin/bash
