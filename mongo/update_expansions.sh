#!/bin/bash

set -o errexit
set -o verbose

if [[ "$OSTYPE" == "linux"* ]]; then
  os="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  os="darwin"
else
  os="windows"
fi

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
  ARCH="arm64"
elif [[ "$ARCH" == "ppc64le" || "$ARCH" == "ppc64" || "$ARCH" == "ppc" || "$ARCH" == "ppcle" ]]; then
  ARCH="ppc64le"
elif [[ "$ARCH" == "s390x" || "$ARCH" == "s390" ]]; then
  ARCH="s390x"
else
  ARCH="x86_64"
fi

bazel_short_git=$(git rev-parse --short HEAD)

bazel_version=$1-mongo_$bazel_short_git

bazel_file_name=bazel-$bazel_version-$os-${ARCH}${2}

echo "bazel_version: $bazel_version" > bazel_expansions.yml
echo "bazel_file_name: $bazel_file_name" >> bazel_expansions.yml
