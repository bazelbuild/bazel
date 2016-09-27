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

EMBED_LABEL_ARG=()
if [ -n "${EMBED_LABEL}" ]; then
    EMBED_LABEL_ARG=(--stamp --embed_label "${EMBED_LABEL}")
fi

: ${JAVA_VERSION:="1.8"}

if [ "${JAVA_VERSION}" = "1.7" ]; then
  : ${BAZEL_ARGS:=--java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain_jdk7 \
        --host_java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain_jdk7 \
        --define JAVA_VERSION=1.7 --ignore_unsupported_sandboxing \
        --compilation_mode=opt \
        "${EXTRA_BAZEL_ARGS:-}"}
else
  : ${BAZEL_ARGS:=--java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain \
        --host_java_toolchain=//src/java_tools/buildjar:bootstrap_toolchain \
        --strategy=Javac=worker --worker_quit_after_build --ignore_unsupported_sandboxing \
        --compilation_mode=opt \
        "${EXTRA_BAZEL_ARGS:-}"}
fi

if [ -z "${BAZEL-}" ]; then
  function run_bootstrapping_bazel() {
    local command=$1
    shift
    run_bazel_jar $command \
        ${BAZEL_ARGS-} --verbose_failures \
        --javacopt="-g -source ${JAVA_VERSION} -target ${JAVA_VERSION}" "${@}"
  }
else
  function run_bootstrapping_bazel() {
    local command=$1
    shift
    ${BAZEL} --bazelrc=${BAZELRC} ${BAZEL_DIR_STARTUP_OPTIONS} $command \
        ${BAZEL_ARGS-} --verbose_failures \
        --javacopt="-g -source ${JAVA_VERSION} -target ${JAVA_VERSION}" "${@}"
  }
fi

function bazel_build() {
  run_bootstrapping_bazel build "${EMBED_LABEL_ARG[@]}" "$@"
}

function get_bazel_bin_path() {
  run_bootstrapping_bazel info "bazel-bin" || echo "bazel-bin"
}

function md5_outputs() {
  [ -n "${BAZEL_TEST_XTRACE:-}" ] && set +x  # Avoid garbage in the output
  # runfiles/MANIFEST & runfiles_manifest contain absolute path, ignore.
  # ar on OS-X is non-deterministic, ignore .a files.
  for i in $(find bazel-bin/ -type f -a \! -name MANIFEST -a \! -name '*.runfiles_manifest' -a \! -name '*.a'); do
    md5_file $i
  done
  for i in $(find bazel-genfiles/ -type f); do
    md5_file $i
  done
  [ -n "${BAZEL_TEST_XTRACE:-}" ] && set -x
}

function get_outputs_sum() {
  md5_outputs | sort -k 2
}

function bootstrap_test() {
  local BAZEL_BIN=$1
  local BAZEL_SUM=$2
  local BAZEL_TARGET=${3:-src:bazel}
  local STRATEGY="--strategy=Javac=worker --worker_quit_after_build"
  if [ "${JAVA_VERSION}" = "1.7" ]; then
    STRATEGY=
  fi
  [ -x "${BAZEL_BIN}" ] || fail "syntax: bootstrap bazel-binary"
  run ${BAZEL_BIN} --nomaster_bazelrc --bazelrc=${BAZELRC} \
      ${BAZEL_DIR_STARTUP_OPTIONS} \
      clean \
      --expunge || return $?
  run ${BAZEL_BIN} --nomaster_bazelrc --bazelrc=${BAZELRC} \
      ${BAZEL_DIR_STARTUP_OPTIONS} \
      build \
      ${EXTRA_BAZEL_ARGS-} ${STRATEGY} \
      --fetch --nostamp \
      --define "JAVA_VERSION=${JAVA_VERSION}" \
      --javacopt="-g -source ${JAVA_VERSION} -target ${JAVA_VERSION}" \
      ${BAZEL_TARGET} || return $?
  if [ -n "${BAZEL_SUM}" ]; then
    cat bazel-genfiles/src/java.version >${BAZEL_SUM}
    get_outputs_sum >> ${BAZEL_SUM} || return $?
  fi
  if [ -z "${BOOTSTRAP:-}" ]; then
    tempdir
    BOOTSTRAP=${NEW_TMPDIR}/bazel
    local FILE=bazel-bin/${BAZEL_TARGET##//}
    cp -f ${FILE/:/\/} $BOOTSTRAP
    chmod +x $BOOTSTRAP
  fi
}
