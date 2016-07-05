#!/bin/bash
# Copyright 2016 The Bazel Authors. All rights reserved.
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

set -eu

if [ "${1-}" == "help" ]; then
  cat <<EOF
Usage:
$0 [port]
     Builds docs and starts a web server serving docs on localhost:port
     Default port is 12345.
$0 <target directory> [<serving prefix>]
     Builds docs as static web pages in <target directory>.
     Replaces absolute paths in the resulting HTML with <serving prefix>,
     or, if it is not specified, with <target directory>.
EOF
  exit 0
fi

if [[ "${1-}" == [0-9]* ]]; then
  readonly PORT=$1
  readonly TARGET=''
else
  readonly PORT=${1-12345}
  readonly TARGET=${1-}
fi

readonly WORKING_DIR=$(mktemp -d)
readonly SERVING_PREFIX=${2-$TARGET}

function check {
  which $1 > /dev/null || (echo "$1 not installed. Please install $1."; exit 1)
}

function build_and_serve {
  bazel build //site:jekyll-tree.tar
  rm -rf $WORKING_DIR/*
  tar -xf bazel-genfiles/site/jekyll-tree.tar -C $WORKING_DIR

  pkill -9 jekyll || true

  if [ -z "$TARGET" ]; then
    echo "Serving bazel.io site at port $PORT"
    jekyll serve --detach --quiet --port $PORT --source $WORKING_DIR
  else
    TMP_TARGET=$(mktemp -d)
    jekyll build --source $WORKING_DIR --destination "$TMP_TARGET"
    REPLACEMENT=$(echo $SERVING_PREFIX | sed s/\\//\\\\\\//g)
    find $TMP_TARGET -name '*.html' | xargs sed -i s/href=\\\"\\//href=\"$REPLACEMENT\\//g
    find $TMP_TARGET -name '*.html' | xargs sed -i s/src=\\\"\\//src=\"$REPLACEMENT\\//g
    cp -R $TMP_TARGET/* $TARGET
    echo "Static pages copied to $TARGET"
    echo "Should be served from $SERVING_PREFIX"
  fi
}

function main {
  check jekyll

  old_version="Jekyll 0.11.2"
  if expr match "$(jekyll --version)" "$old_version"; then
    # The ancient version that apt-get has.
    echo "ERROR: Running with an old version of Jekyll, update " \
      "to 2.5.3 with \`sudo gem install jekyll -v 2.5.3\`"
    exit 1
  fi

  build_and_serve

  echo "Type q to quit, r to rebuild docs and restart jekyll"
  while true; do

    read -n 1 -s user_input
    if [ "$user_input" == "q" ]; then
      echo "Quitting"
      exit 0
    elif [ "$user_input" == "r" ]; then
      echo "Rebuilding docs and restarting jekyll"
      build_and_serve
      echo "Rebuilt docs and restarted jekyll"
    fi
  done
}

function cleanup {
  rm -rf $WORKING_DIR
  pkill -9 jekyll
}
trap cleanup EXIT

main
