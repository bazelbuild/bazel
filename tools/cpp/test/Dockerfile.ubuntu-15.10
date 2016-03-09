FROM ubuntu:15.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates \
      git pkg-config zip unzip \
      g++ gcc openjdk-8-jdk \
      zlib1g-dev libarchive-dev \
      ca-certificates-java && \
    apt-get clean
