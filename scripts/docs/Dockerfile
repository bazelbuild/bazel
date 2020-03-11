# This Dockerfile is used to build the container that then builds Bazel's
# website with Jekyll.
#
# You can test it on your local machine like this:
#
# $ docker build -t bazel-jekyll .
# $ docker run -it --rm -p 8000:8000 --entrypoint /bin/bash bazel-jekyll
# $ git clone https://bazel.googlesource.com/bazel
# $ cd bazel
# $ curl -o bazel https://releases.bazel.build/0.19.0/release/bazel-0.19.0-linux-x86_64
# $ chmod +x bazel
# $ ./bazel build //site
# $ cd bazel-bin/site/site-build
# $ python -m SimpleHTTPServer
#
# Then access the website in your browser via http://localhost:8000

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get -qqy update && \
    apt-get -qqy install build-essential curl liblzma-dev python3.7 \
      python-pygments ruby ruby-dev unzip zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/bin/python3.7 /usr/bin/python

RUN curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-linux-amd64 && \
    chown root:root /usr/local/bin/bazel && \
    chmod 0755 /usr/local/bin/bazel
RUN bazel version

COPY Gemfile .
RUN gem install -g --no-rdoc --no-ri && rm -f Gemfile
