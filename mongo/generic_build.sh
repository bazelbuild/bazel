#!/bin/bash

set -o errexit
set -o verbose

curl -L $1 -o bootstrap_bazel$3
chmod +x "./bootstrap_bazel$3"
echo "//src:bazel$3" > target.file
"./bootstrap_bazel$3" --output_base=$PWD/bazel_output_base build --compilation_mode=opt --subcommands --verbose_failures --stamp --embed_label=$2 --target_pattern_file="target.file"
cp "bazel-bin/src/bazel$3" "mongo_bazel$3"
"./mongo_bazel$3" info
"./mongo_bazel$3" --version
shasum -b -a 256 "./mongo_bazel$3" | cut -d " " -f 1  > ./mongo_bazel${3}.sha256