FROM python:3.6.8
ENV http_proxy http://proxy.infra.dgfip:8080
ENV https_proxy http://proxy.infra.dgfip:8080

ADD sf_datalake.tar.gz /pytmp/sf_datalake
ARG https_proxy=http://proxy.infra.dgfip:8080
ARG http_proxy=http://proxy.infra.dgfip:8080
SHELL ["/bin/bash", "-c"]
RUN echo "Acquire::http::Proxy \"$http_proxy\";" > /etc/apt/apt.conf.d/proxy && \
    apt-get update && \
    apt-get --assume-yes install virtualenv zip
WORKDIR /pytmp
RUN virtualenv -p /usr/local/bin/python3 sf_venv && \
    source sf_venv/bin/activate && \
    pip install -v ./sf_datalake/ && \
    venv-pack -o /python_packages.tar.gz
