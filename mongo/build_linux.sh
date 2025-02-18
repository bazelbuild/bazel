#!/bin/bash

set -o errexit
set -o verbose

podman pull docker.io/redhat/ubi8:8.10-1184
podman run  --mount type=bind,source=$PWD,destination=/tmp/bazel/,rw=true --workdir /tmp/bazel redhat/ubi8:8.10-1184 /tmp/bazel/mongo/container_build.sh "$1" "$2" "$3"
"./$3" info
"./$3" --version
