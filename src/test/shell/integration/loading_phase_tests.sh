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

# Our tests use the static crosstool, so make it the default.
add_to_bazelrc "build --crosstool_top=@bazel_tools//tools/cpp:default-toolchain"

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

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
    mkdir -p x || fail "mkdir x failed"
    echo "load('/y/rules', 'a')" >x/BUILD
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
    cp ${bazelrc} ${new_workspace_dir}/.${PRODUCT_NAME}rc

    echo "build --cpu=armeabi-v7a" >>.${PRODUCT_NAME}rc    # default bazelrc
    $PATH_TO_BAZEL_BIN info --announce_rc >/dev/null 2>$TEST_log
    expect_log "Reading.*$(pwd)/.${PRODUCT_NAME}rc:
.*--cpu=armeabi-v7a"

    cp .${PRODUCT_NAME}rc foo
    echo "build --cpu=armeabi-v7a"   >>foo         # non-default bazelrc
    $PATH_TO_BAZEL_BIN --${PRODUCT_NAME}rc=foo info --announce_rc >/dev/null \
      2>$TEST_log
    expect_log "Reading.*$(pwd)/foo:
.*--cpu=armeabi-v7a"
}

# This exercises the production-code assertion in AbstractCommand.java
# that all help texts mention their %{options}.
function test_all_help_topics_succeed() {
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
  [ ${#topics[@]} -gt 15 ] || fail "Hmmm: not many topics: ${topics[@]}."
}

# Regression for "Sticky error during analysis phase when input is cyclic".
function test_regress_cycle_during_analysis_phase() {
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

function test_glob_with_io_error() {
  mkdir -p t/u
  touch t/u/v

  echo "filegroup(name='t', srcs=glob(['u/*']))" > t/BUILD
  chmod 000 t/u

  bazel query '//t:*' >& $TEST_log && fail "Expected failure"
  expect_log 'error globbing.*Permission denied'

  chmod 755 t/u
  bazel query '//t:*' >& $TEST_log || fail "Expected success"
  expect_not_log 'error globbing.*Permission denied'
  expect_log '//t:u'
  expect_log '//t:u/v'
}

function test_build_file_symlinks() {
  mkdir b || fail "couldn't make b"
  ln -s b a || fail "couldn't link a to b"

  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  touch b/BUILD
  bazel query a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"

  unlink a || fail "couldn't unlink a"
  ln -s c a
  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  mkdir c || fail "couldn't make c"
  ln -s foo c/BUILD || "couldn't link c/BUILD to c/foo"
  bazel query a:all >& $TEST_log && fail "Expected failure"
  expect_log "no such package 'a'"

  touch c/foo
  bazel query a:all >& $TEST_log || fail "Expected success"
  expect_log "Empty results"
}

function test_visibility_edge_causes_cycle() {
  mkdir -p a b || fail "mkdir failed"
  echo 'sh_library(name="a", visibility=["//b"])' > a/BUILD
  echo 'sh_library(name="b", deps=["//a"])' > b/BUILD
  bazel query 'deps(//a)' >& $TEST_log && fail "Expected failure"
  expect_log "cycle in dependency graph"
  expect_log "The cycle is caused by a visibility edge"
  bazel query 'deps(//b)' >& $TEST_log && fail "Expected failure"
  expect_log "cycle in dependency graph"
  expect_log "The cycle is caused by a visibility edge"
  echo 'sh_library(name="a", visibility=["//b:__pkg__"])' > a/BUILD
  bazel query 'deps(//a)' >& $TEST_log || fail "Expected success"
  expect_log "//a:a"
  expect_not_log "//b:b"
  bazel query 'deps(//b)' >& $TEST_log || fail "Expected success"
  expect_log "//a:a"
  expect_log "//b:b"
}

# Regression test for bug "ASTFileLookupFunction has an unnoted
# dependency on the PathPackageLocator".
function test_incremental_deleting_package_roots() {
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
  mkdir foo

  echo 'workspace(name="wsname1")' > WORKSPACE
  echo 'sh_library(name="shname1")' > foo/BUILD
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_log "//foo:shname1"

  echo 'sh_library(name="shname2")' > foo/BUILD
  # TODO(b/37617303): make tests UI-independent
  bazel query --noexperimental_ui //foo:all >& "$TEST_log" \
      || fail "Expected success"
  expect_log "Loading package: foo"
  expect_log "//foo:shname2"

  echo 'workspace(name="wsname1")' > WORKSPACE
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
