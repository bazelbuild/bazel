#!/bin/bash

set -o errexit
set -o verbose

yum install -y gcc gcc-c++ python3 zip java-21-openjdk-devel
curl -L "$1" -o bazel_bootstrap
chmod +x ./bazel_bootstrap
./bazel_bootstrap build  --compilation_mode=opt --subcommands --verbose_failures --stamp --embed_label=$2 //src:bazel
cp bazel-bin/src/bazel "$3"
sha256sum -b "$3" > "${3}.sha256"
