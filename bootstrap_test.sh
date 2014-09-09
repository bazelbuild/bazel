#! /bin/bash

set -e

[ -x "output/bazel" ] || ./compile.sh

output/bazel build //src:bazel
BOOTSTRAP=$(mktemp)
cp -f bazel-genfiles/src/bazel $BOOTSTRAP
chmod +x $BOOTSTRAP

$BOOTSTRAP clean
$BOOTSTRAP build //src:bazel

bazel-genfiles/src/bazel >/dev/null  # check that execution succeeds

rm -f $BOOTSTRAP
echo "Bootstrap test succeeded"
