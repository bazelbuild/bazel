#!/bin/bash

set -ex

[[ -x "output/bazel" ]] || ./compile.sh

OUTPUT_BASE=$(mktemp -d /tmp/bazel_example_workspace.XXXXXXXX)

function delete_output_base() {
  rm -fr $OUTPUT_BASE
}

trap delete_output_base EXIT

cd example_workspace
../output/bazel --batch --output_base $OUTPUT_BASE build -k //...
rm -f bazel-*

echo "Example workspace test succeeded"
