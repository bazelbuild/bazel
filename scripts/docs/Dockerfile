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

FROM gcr.io/cloud-builders/bazel

RUN apt-get update && \
  apt-get -y install ruby ruby-dev build-essential python-pygments && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

COPY Gemfile .
RUN gem install -g --no-rdoc --no-ri && rm -f Gemfile
