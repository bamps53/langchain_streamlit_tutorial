FROM nvcr.io/nvidia/pytorch:24.02-py3

WORKDIR /workspace


ENV DEBIAN_FRONTEND=noninteractive \
    LANG=ja_JP.UTF-8 \
    TZ=Asia/Tokyo \
    apt_get_server=ftp.jaist.ac.jp/pub/Linux \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/usr/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tmux \
    zsh \
    sudo && \
    rm -rf /var/lib/apt/lists/* # Delete the apt-get lists after installing something
