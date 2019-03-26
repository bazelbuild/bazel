#!/bin/bash

set -o xtrace
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

# Use bazel to bootstrap various tools
# Configuration:
#   BAZEL: path to the bazel binary
#   EMBED_LABEL: the label to embed in tools using --embed_label (optional)
#   BAZELRC: the rc file to use

: ${BAZELRC:="/dev/null"}
: ${EMBED_LABEL:=""}
: ${SOURCE_DATE_EPOCH:=""}

EMBED_LABEL_ARG=()
if [ -n "${EMBED_LABEL}" ]; then
    EMBED_LABEL_ARG=(--stamp --embed_label "${EMBED_LABEL}")
fi

: ${JAVA_VERSION:="1.8"}

_BAZEL_ARGS="--java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain \
      --host_java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain \
      --spawn_strategy=standalone \
      --nojava_header_compilation \
      --strategy=Javac=worker --worker_quit_after_build --ignore_unsupported_sandboxing \
      --compilation_mode=opt \
      --distdir=derived/distdir \
      ${EXTRA_BAZEL_ARGS:-}"

if [ -z "${BAZEL-}" ]; then
  function _run_bootstrapping_bazel() {
    local command=$1
    shift
    run_bazel_jar $command \
        ${_BAZEL_ARGS} --verbose_failures \
        --javacopt="-g -source ${JAVA_VERSION} -target ${JAVA_VERSION}" "${@}"
  }
else
  function _run_bootstrapping_bazel() {
    local command=$1
    shift
    ${BAZEL} --bazelrc=${BAZELRC} ${BAZEL_DIR_STARTUP_OPTIONS} $command \
        ${_BAZEL_ARGS} --verbose_failures \
        --javacopt="-g -source ${JAVA_VERSION} -target ${JAVA_VERSION}" "${@}"
  }
fi

function bazel_build() {
  _run_bootstrapping_bazel build "${EMBED_LABEL_ARG[@]}" "$@"
}

function get_bazel_bin_path() {
  _run_bootstrapping_bazel info "bazel-bin" || echo "bazel-bin"
}
