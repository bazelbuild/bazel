#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Jenkins is capable of executing shell scripts directly, even on Windows,
# but it uses a shell binary bundled with it and not the msys one. We don't
# want to use two different shells, so a batch file is used instead to call
# the msys shell.

# We need to execute bash with -l so that we don't get the usual environment
# variables from cmd.exe which would interfere with our operation, but that
# means that PWD will be $HOME. Thus, we need to cd to the source tree.
cd $(dirname $0)/../../..

# Find Java. Minor versions and thus the name of the directory changes quite
# often.
export JAVA_HOME=$(ls -d c:/Program\ Files/Java/jdk* 2> /dev/null | head -n 1)
if [[ "$JAVA_HOME" == "" ]]; then
  echo "JDK not found under c:\\Program Files\\Java" 1>& 2
  exit 1
fi

# These variables are temporarily needed for Bazel
export BAZEL_SH="$(cygpath --windows /bin/bash)"
export TMPDIR=${TMPDIR:-c:/bazel_ci/temp}
export PATH="${PATH}:/c/python_27_amd64/files"
mkdir -p "${TMPDIR}"  # mkdir does work with a path starting with 'c:/', wow

# Even though there are no quotes around $* in the .bat file, arguments
# containing spaces seem to be passed properly.
echo "Bootstrapping Bazel"
retCode=0
source ./scripts/ci/build.sh

# TODO(bazel-team): we should replace ./compile.sh by the same script we use
# for other platform
release_label="$(get_full_release_name)"

if [ -n "${release_label}" ]; then
  export EMBED_LABEL="${release_label}"
fi
./compile.sh "$*" || retCode=$?
if (( $retCode != 0 )); then
  echo "$retCode" > .unstable
  exit 0
fi

# Copy the resulting artifact.
mkdir -p output/ci
cp output/bazel.exe output/ci/bazel-$(get_full_release_name).exe
zip -j output/ci/bazel-$(get_full_release_name).zip output/bazel.exe

# todo(bazel-team): add more tests here.
echo "Running tests"
./output/bazel test -k --test_output=all --test_tag_filters -no_windows\
  //src/test/shell/bazel:bazel_windows_example_test \
  //src/test/java/...
retCode=$?

# Exit for failure except for test failures (exit code 3).
if (( $retCode != 0 )); then
  echo "$retCode" > .unstable
fi
