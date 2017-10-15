FROM ubuntu:15.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates ca-certificates-java \
      git pkg-config \
      zip unzip zlib1g-dev libarchive-dev \
      g++ gcc openjdk-8-jdk python && \
    apt-get clean
