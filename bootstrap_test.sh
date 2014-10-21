#! /bin/bash

set -e

[ -x "output/bazel" ] || ./compile.sh

output/bazel build //src:bazel
BOOTSTRAP=$(mktemp /tmp/bootstrap.XXXXXXXXXX)
cp -f bazel-genfiles/src/bazel $BOOTSTRAP
chmod +x $BOOTSTRAP

$BOOTSTRAP clean
$BOOTSTRAP build //src:bazel

bazel-genfiles/src/bazel >/dev/null  # check that execution succeeds

$BOOTSTRAP test --test_output=errors //src:all

rm -f $BOOTSTRAP
echo "Bootstrap test succeeded"
