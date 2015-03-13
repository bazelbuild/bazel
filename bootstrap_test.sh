#! /bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

function usage() {
  [ -n "${1:-}" ] && echo "Invalid command(s): $1" >&2
  echo "syntax: $0 command[,command]* [BAZEL_BIN [BAZEL_SUM]]" >&2
  echo "  Available commands: bootstrap, determinism, test, all" >&2
  exit 1
}

function parse_options() {
  local keywords="(bootstrap|test|determinism|all)"
  [[ "${1:-}" =~ ^$keywords(,$keywords)*$ ]] || usage "$@"
  DO_BOOTSTRAP=
  DO_CHECKSUM=
  DO_TESTS=
  [[ "$1" =~ (bootstrap|all) ]] && DO_BOOTSTRAP=1
  [[ "$1" =~ (determinism|all) ]] && DO_CHECKSUM=1
  [[ "$1" =~ (test|all) ]] && DO_TESTS=1

  BAZEL_BIN=${2:-"bazel-bin/src/bazel"}
  BAZEL_SUM=${3:-"bazel-out/bazel_checksum"}
}

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
if [[ ${PLATFORM} == "darwin" ]]; then
  function md5_file() {
    md5 $1 | sed 's|^MD5 (\(.*\)) =|\1|'
  }
else
  function md5_file() {
    md5sum $1
  }
fi

function md5_outputs() {
  [ -n "${BAZEL_TEST_XTRACE:-}" ] && set +x  # Avoid garbage in the output
  # genproto does not strip out timestamp, let skip it for now
  # runfiles/MANIFEST & runfiles_manifest contain absolute path, ignore.
  # ar on OS-X is non-deterministic, ignore .a files.
  for i in $(find bazel-bin/ -type f -a \! -name 'libproto_*.jar' -a \! -name MANIFEST -a \! -name '*.runfiles_manifest' -a \! -name '*.a'); do
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

function fail() {
  echo $1 >&2
  exit 1
}

function bootstrap() {
  local BAZEL_BIN=$1
  local BAZEL_SUM=$2
  [ -x "${BAZEL_BIN}" ] || fail "syntax: bootstrap bazel-binary"
  ${BAZEL_BIN} --blazerc=/dev/null clean || return $?
  ${BAZEL_BIN} --blazerc=/dev/null build --nostamp //src:bazel //src:tools || return $?

  if [ -n "${BAZEL_SUM}" ]; then
    get_outputs_sum > ${BAZEL_SUM} || return $?
  fi
}

function copy_bootstrap() {
  if [ -z "${BOOTSTRAP:-}" ]; then
    BOOTSTRAP=$(mktemp /tmp/bootstrap.XXXXXXXXXX)
    trap '{ rm -f $BOOTSTRAP; }' EXIT
    cp -f $BAZEL_BIN $BOOTSTRAP
    chmod +x $BOOTSTRAP
  fi
}

function start_test() {
  echo "****${1//?/*}****"
  echo "*** $1 ***"
  echo "****${1//?/*}****"
}

function end_test() {
  echo "   ==> $1 passed"
  echo
  echo
}

parse_options "$@"

if [ -n "${DO_BOOTSTRAP}" -o ! -x ${BAZEL_BIN} ]; then
  start_test bootstrap
  [ -x "output/bazel" ] || ./compile.sh || fail "Compilation failed"
  bootstrap output/bazel ${BAZEL_SUM} || fail "Bootstrap failed"
  end_test bootstrap
fi

# check that bootstrapped binary actually runs correctly
copy_bootstrap
$BOOTSTRAP >/dev/null || fail "Bootstrapped binary is non-functional"

if [ $DO_CHECKSUM ]; then
  start_test checksum

  SUM1=$(mktemp /tmp/bootstrap-sum.XXXXXXXXXX)
  SUM2=$(mktemp /tmp/bootstrap-sum.XXXXXXXXXX)

  trap '{ rm -f $BOOTSTRAP $SUM1 $SUM2; }' EXIT
  cat $BAZEL_SUM > $SUM1

  # Second run
  bootstrap $BOOTSTRAP $SUM2 || fail "Bootstrap failed"
  (diff -U 0 $SUM1 $SUM2 >&2) || fail "Differences detected in outputs!"

  end_test checksum
fi

if [ $DO_TESTS ]; then
  start_test "test"

  $BOOTSTRAP --blazerc=/dev/null test -k --test_output=errors //src/... || fail "Tests failed"
  end_test "test"
fi

echo "Bootstrap tests succeeded (tested: $1)"
