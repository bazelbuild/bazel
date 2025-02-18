#!/bin/bash

set -o errexit
set -o verbose

curl -L $1 -o bootstrap_bazel$4
chmod +x "./bootstrap_bazel$4"
echo "//src:bazel$4" > target.file
"./bootstrap_bazel$4" --output_base=$PWD/bazel_output_base build --compilation_mode=opt --subcommands --verbose_failures --stamp --embed_label=$2 --target_pattern_file="target.file"
cp "bazel-bin/src/bazel$4" "$3"
"./$3" info
"./$3" --version
shasum -b -a 256 "./$3" > "./${3}.sha256"
