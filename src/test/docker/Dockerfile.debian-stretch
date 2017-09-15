FROM debian:stretch
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates ca-certificates-java \
      git pkg-config build-essential \
      zip unzip zlib1g-dev libarchive-dev \
      g++ gcc openjdk-8-jdk python && \
    apt-get clean
