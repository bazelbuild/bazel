#!/bin/bash
#
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
#
# loading_phase_tests.sh: miscellaneous integration tests of Bazel,
# that use only the loading or analysis phases.
#

# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

# Our tests use the static crosstool, so make it the default.
# add_to_bazelrc "build --crosstool_top=@bazel_tools//tools/cpp:default-toolchain"

output_base=$TEST_TMPDIR/out
TEST_stderr=$(dirname $TEST_log)/stderr

#### HELPER FUNCTIONS ##################################################

function set_up() {
    cd ${WORKSPACE_DIR}
}

function tear_down() {
    bazel shutdown
}

#### TESTS #############################################################

function test_query_buildfiles_with_load() {
    local -r pkg="${FUNCNAME}"
    mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
    touch WORKSPACE

    mkdir -p x || fail "mkdir x failed"
    echo "load('//y:rules.bzl', 'a')" >x/BUILD
    echo "cc_library(name='x')"   >>x/BUILD
    mkdir -p y || fail "mkdir y failed"
    touch y/BUILD
    echo "a=1" >y/rules.bzl

    bazel query --noshow_progress 'buildfiles(//x)' >$TEST_log ||
        fail "Expected success"
    expect_log //x:BUILD
    expect_log //y:BUILD
    expect_log //y:rules.bzl

    # null terminated:
    bazel query --noshow_progress --null 'buildfiles(//x)' >null.log ||
        fail "Expected null success"
    printf '//y:rules.bzl\0//y:BUILD\0//x:BUILD\0' >null.ref.log
    cmp null.ref.log null.log || fail "Expected match"

    # Missing skylark file:
    rm -f y/rules.bzl
    bazel query --noshow_progress 'buildfiles(//x)' 2>$TEST_log &&
        fail "Expected error"
    expect_log "Extension file not found. Unable to load file '//y:rules.bzl'"
}

# Regression test for:
# "Skyframe does not build targets that transitively depend on non-rule targets
# that live in packages with errors".
function test_non_error_target_in_bad_pkg() {
    local -r pkg="${FUNCNAME}"
    mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
    touch WORKSPACE

    mkdir -p a || fail "mkdir a failed"
    mkdir -p b || fail "mkdir b failed"

    echo "sh_library(name = 'a', data = ['//b'])" > a/BUILD
    echo "exports_files(['b'])" > b/BUILD
    echo "genrule(name='r1', cmd = '', outs = ['conflict'])" >> b/BUILD
    echo "genrule(name='r2', cmd = '', outs = ['conflict'])" >> b/BUILD

    bazel build --nobuild -k //a >& $TEST_log && fail "Expected failure"
    expect_log "'conflict' in rule"
    expect_not_log "Loading failed"
    expect_log "but there were loading phase errors"
    expect_not_log "Loading succeeded for only"
}

# This is a regression test to make sure that none of the bazel
# commands has an incompatible set of @Options annotations.  In the
# past, options have been declared multiple times, making a command
# unusable.
function test_options_errors() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  # Enumerate bazel commands...
  bazel help 2>/dev/null |
      grep  '^  [a-z]' |
      grep -v '^  '${PRODUCT_NAME}' ' |
      awk '{print $1}' |
  while read command; do
    bazel $command >$TEST_log 2>&1 || true
    # Mustn't crash in the options package:
    expect_not_log "Duplicate option name"
    expect_not_log "at com.google.devtools.build.lib"
    expect_not_log "lib.util.options.*Exception"
  done
}

function test_bazelrc_option() {
    local -r pkg="${FUNCNAME}"
    mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
    touch WORKSPACE

    cp "${bazelrc}" ".${PRODUCT_NAME}rc"

    echo "build --subcommands" >>".${PRODUCT_NAME}rc"    # default bazelrc
    $PATH_TO_BAZEL_BIN info --announce_rc >/dev/null 2>$TEST_log
    expect_log "Reading.*$pkg[/\\\\].${PRODUCT_NAME}rc:
.*--subcommands"

    cp .${PRODUCT_NAME}rc foo
    echo "build --nosubcommands"   >>foo         # non-default bazelrc
    $PATH_TO_BAZEL_BIN --${PRODUCT_NAME}rc=foo info --announce_rc >/dev/null \
      2>$TEST_log
    expect_log "Reading.*$pkg[/\\\\]foo:
.*--nosubcommands"
}

# This exercises the production-code assertion in AbstractCommand.java
# that all help texts mention their %{options}.
function test_all_help_topics_succeed() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  topics=($(bazel help 2>/dev/null |
              grep '^  [a-z]' |
              grep -v '^  '${PRODUCT_NAME}' ' |
              awk '{print $1}') \
          startup_options \
          target-syntax)
  for topic in "${topics[@]}"; do
    bazel help $topic >$TEST_log 2>&1 || {
       fail "help $topic failed"
       expect_not_log .  # print the log
    }
  done
  [ ${#topics[@]} -gt 15 ] || fail "Hmmm: not many topics: ${topics[*]}."
}

# Regression for "Sticky error during analysis phase when input is cyclic".
function test_regress_cycle_during_analysis_phase() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  mkdir -p cycle main
  cat >main/BUILD <<EOF
genrule(name='mygenrule', outs=['baz.h'], srcs=['//cycle:foo.h'], cmd=':')
EOF
  cat >cycle/BUILD <<EOF
genrule(name='foo.h', outs=['bar.h'], srcs=['foo.h'], cmd=':')
EOF
  bazel build --nobuild //cycle:foo.h >$TEST_log 2>&1 || true
  expect_log "in genrule rule //cycle:foo.h: .*dependency graph"
  expect_log "//cycle:foo.h.*self-edge"

  bazel build --nobuild //main:mygenrule >$TEST_log 2>&1 || true
  expect_log "in genrule rule //cycle:foo.h: .*dependency graph"
  expect_log "//cycle:foo.h.*self-edge"

  bazel build --nobuild //cycle:foo.h >$TEST_log 2>&1 || true
  expect_log "in genrule rule //cycle:foo.h: .*dependency graph"
  expect_log "//cycle:foo.h.*self-edge"
}

# glob function should not return values that are outside the package
function test_glob_with_subpackage() {
    local -r pkg="${FUNCNAME}"
    mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
    touch WORKSPACE

    mkdir -p p/subpkg || fail "mkdir p/subpkg failed"
    mkdir -p p/dir || fail "mkdir p/dir failed"

    echo "exports_files(glob(['**/*.txt']))" >p/BUILD
    echo "# Empty" >p/subpkg/BUILD

    echo "p/t1.txt" > p/t1.txt
    echo "p/dir/t2.txt" > p/dir/t2.txt
    echo "p/subpkg/t3.txt" > p/subpkg/t3.txt

    bazel query 'p:*' >$TEST_log || fail "Expected success"
    expect_log '//p:t1\.txt'
    expect_log '//p:dir/t2\.txt'
    expect_log '//p:BUILD'
    expect_not_log 't3\.txt'
    assert_equals "3" $(wc -l "$TEST_log")

    # glob returns an empty list, because t3.txt is outside the package
    echo "exports_files(glob(['subpkg/t3.txt']))" >p/BUILD
    bazel query 'p:*' -k >$TEST_log || fail "Expected success"
    expect_log '//p:BUILD'
    assert_equals "1" $(wc -l "$TEST_log")

    # same test, with a nonexisting file
    echo "exports_files(glob(['subpkg/no_glob.txt']))" >p/BUILD
    bazel query 'p:*' -k >$TEST_log || fail "Expected success"
    expect_log '//p:BUILD'
    assert_equals "1" $(wc -l "$TEST_log")

    # Non-recursive wildcard gives the same result as the recursive wildcard
    echo "exports_files(glob(['*.txt', '*/*.txt']))" >p/BUILD
    bazel query 'p:*' >$TEST_log || fail "Expected success"
    expect_log '//p:t1\.txt'
    expect_log '//p:dir/t2\.txt'
    expect_log '//p:BUILD'
    expect_not_log 't3\.txt'
    assert_equals "3" $(wc -l "$TEST_log")
}

function test_glob_with_subpackage2() {
    local -r pkg="${FUNCNAME}"
    mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
    touch WORKSPACE

    mkdir -p p/q/subpkg || fail "mkdir p/q/subpkg failed"
    mkdir -p p/q/dir || fail "mkdir p/q/dir failed"

    echo "exports_files(glob(['**/*.txt']))" >p/q/BUILD
    echo "# Empty" >p/q/subpkg/BUILD

    echo "p/q/t1.txt" > p/q/t1.txt
    echo "p/q/dir/t2.txt" > p/q/dir/t2.txt
    echo "p/q/subpkg/t3.txt" > p/q/subpkg/t3.txt

    bazel query 'p/q:*' >$TEST_log || fail "Expected success"
    expect_log '//p/q:t1\.txt'
    expect_log '//p/q:dir/t2\.txt'
    expect_log '//p/q:BUILD'
    expect_not_log 't3\.txt'
    assert_equals "3" $(wc -l "$TEST_log")
}

# Regression test for bug "ASTFileLookupFunction has an unnoted
# dependency on the PathPackageLocator".
function test_incremental_deleting_package_roots() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  local other_root=$TEST_TMPDIR/other_root/${WORKSPACE_NAME}
  mkdir -p $other_root/a
  touch $other_root/WORKSPACE
  echo 'sh_library(name="external")' > $other_root/a/BUILD
  mkdir -p a
  echo 'sh_library(name="internal")' > a/BUILD

  bazel query --package_path=$other_root:. a:all >& $TEST_log \
      || fail "Expected success"
  expect_log "//a:external"
  expect_not_log "//a:internal"
  rm -r $other_root
  bazel query --package_path=$other_root:. a:all >& $TEST_log \
      || fail "Expected success"
  expect_log "//a:internal"
  expect_not_log "//a:external"
  mkdir -p $other_root
  bazel query --package_path=$other_root:. a:all >& $TEST_log \
      || fail "Expected success"
  expect_log "//a:internal"
  expect_not_log "//a:external"
}

function test_no_package_loading_on_benign_workspace_file_changes() {
  local -r pkg="${FUNCNAME}"
  mkdir -p "$pkg" && cd "$pkg" || fail "could not create and cd \"$pkg\""
  touch WORKSPACE

  mkdir foo

  echo 'workspace(name="wsname1")' > WORKSPACE
  echo 'sh_library(name="shname1")' > foo/BUILD
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_log "Loading package: foo"
  expect_log "//foo:shname1"

  echo 'sh_library(name="shname2")' > foo/BUILD
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_log "Loading package: foo"
  expect_log "//foo:shname2"

  # Test that comment changes do not cause package reloading
  echo '#benign comment' >> WORKSPACE
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_not_log "Loading package: foo"
  expect_log "//foo:shname2"

  echo 'workspace(name="wsname2")' > WORKSPACE
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_log "Loading package: foo"
  expect_log "//foo:shname2"
}

run_suite "Integration tests of ${PRODUCT_NAME} using loading/analysis phases."
