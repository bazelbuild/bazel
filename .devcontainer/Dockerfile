FROM ubuntu:24.10

# Update package lists
RUN apt-get update

RUN apt-get install -y \
    # Required to build Bazel - https://bazel.build/install/compile-source#build-bazel-on-unixes
    # NOTE python3 already installed, python2 is no longer supported
    build-essential \
    openjdk-21-jdk \
    zip \
    unzip \
    # Dev utils
    git \
    curl
    
# Install bazelisk, buildifier, buildozer, unused_deps, starpls (amd64 only)
ARG TARGETARCH
RUN if [ $TARGETARCH = "arm64" ]; then \
        curl https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64 -Lo /usr/local/bin/bazel ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildifier-linux-arm64 -Lo /usr/local/bin/buildifier ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildozer-linux-arm64 -Lo /usr/local/bin/buildozer ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/unused_deps-linux-arm64 -Lo /usr/local/bin/unused_deps ; \
    fi
RUN if [ $TARGETARCH = "amd64" ]; then \
        curl https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 -Lo /usr/local/bin/bazel ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildifier-linux-amd64 -Lo /usr/local/bin/buildifier ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/buildozer-linux-amd64 -Lo /usr/local/bin/buildozer ; \
        curl https://github.com/bazelbuild/buildtools/releases/download/v7.1.2/unused_deps-linux-amd64 -Lo /usr/local/bin/unused_deps ; \
        curl https://github.com/withered-magic/starpls/releases/download/v0.1.14/starpls-linux-amd64 -Lo /usr/local/bin/starpls ; \
    fi
RUN chmod +x /usr/local/bin/bazel /usr/local/bin/buildifier /usr/local/bin/buildozer /usr/local/bin/unused_deps
RUN if [ $TARGETARCH = "amd64" ]; then \
        chmod +x /usr/local/bin/starpls ; \
    fi

CMD [ "bash"]