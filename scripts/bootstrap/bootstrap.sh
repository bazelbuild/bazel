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

: ${JAVA_VERSION:="11"}

_BAZEL_ARGS="--spawn_strategy=standalone \
      --nojava_header_compilation \
      --strategy=Javac=worker --worker_quit_after_build --ignore_unsupported_sandboxing \
      --compilation_mode=opt \
      --distdir=derived/distdir \
      --extra_toolchains=//scripts/bootstrap:all \
      --extra_toolchains=@bazel_tools//tools/python:autodetecting_toolchain \
      --override_repository=maven="$(get_cwd)/maven" \
      ${DIST_BOOTSTRAP_ARGS:-} \
      ${EXTRA_BAZEL_ARGS:-}"

cp scripts/bootstrap/BUILD.bootstrap scripts/bootstrap/BUILD

# Remove lines containing 'install_deps' to avoid loading @bazel_pip_dev_deps,
# which requires fetching the python toolchain.
sed -i.bak '/install_deps/d' WORKSPACE && rm WORKSPACE.bak

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
