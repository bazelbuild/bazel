#!/bin/bash
#
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
#
# bazel_query_test.sh: integration tests for bazel query

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function set_up() {
  add_to_bazelrc "build --package_path=%workspace%"
  setup_skylib_support
}

function tear_down() {
  bazel shutdown
}

#### TESTS #############################################################

function test_does_not_fail_horribly() {
  rm -rf peach
  mkdir -p peach
  cat > peach/BUILD <<EOF
sh_library(name='brighton', deps=[':harken'])
sh_library(name='harken')
EOF

  bazel query 'deps(//peach:brighton)' > $TEST_log

  expect_log "//peach:brighton"
  expect_log "//peach:harken"
}

function test_visibility_affects_xml_output() {
  rm -rf kiwi
  mkdir -p kiwi

  cat > kiwi/BUILD <<EOF
sh_library(name='kiwi', visibility=['//visibility:private'])
EOF
  bazel query --output=xml '//kiwi:kiwi' > output_private

  cat > kiwi/BUILD <<EOF
sh_library(name='kiwi', visibility=['//visibility:public'])
EOF
  bazel query --output=xml '//kiwi:kiwi' > output_public

  cat > kiwi/BUILD <<EOF
sh_library(name='kiwi')
EOF
  bazel query --output=xml '//kiwi:kiwi' > output_none

  cmp output_private output_public && fail "visibility does not affect XML output"
  cmp output_none output_private && fail "visibility does not affect XML output"
  cmp output_none output_public && fail "visibility does not affect XML output"

  assert_contains "//kiwi:kiwi" output_private
  assert_contains "//kiwi:kiwi" output_public
  assert_contains "//kiwi:kiwi" output_none

  assert_contains "//visibility:private" output_private
  assert_contains "//visibility:public" output_public
  assert_not_contains "//visibility:private" output_none
  assert_not_contains "//visibility:public" output_none
}

function test_visibility_affects_proto_output() {
  rm -rf kiwi
  mkdir -p kiwi

  cat > kiwi/BUILD <<EOF
sh_library(name='kiwi', visibility=['//visibility:private'])
EOF
  bazel query --output=proto '//kiwi:kiwi' > output_private

  cat > kiwi/BUILD <<EOF
sh_library(name='kiwi', visibility=['//visibility:public'])
EOF
  bazel query --output=proto '//kiwi:kiwi' > output_public

  # There is no check for unspecified visibility because proto output format
  # adds every attribute to the output, regardless of whether they are specified
  # or have the default value

  cmp output_private output_public && fail "visibility does not affect proto output"

  assert_contains "//kiwi:kiwi" output_private
  assert_contains "//kiwi:kiwi" output_public

  assert_contains "//visibility:private" output_private
  assert_contains "//visibility:public" output_public
}

function make_depth_tests() {
  rm -rf depth
  rm -rf depth2
  mkdir -p depth depth2 || die "Could not create test directory"
  cat > "depth/BUILD" <<EOF
sh_binary(name = 'one', srcs = ['one.sh'], deps = [':two'])
sh_library(name = 'two', srcs = ['two.sh'],
           deps = [':div2', ':three', '//depth2:three'])
sh_library(name = 'three', srcs = ['three.sh'], deps = [':four'])
sh_library(name = 'four', srcs = ['four.sh'], deps = [':div2', ':five'])
sh_library(name = 'five', srcs = ['five.sh'])
sh_library(name = 'div2', srcs = ['two.sh'])
EOF

  echo "sh_library(name = 'three', srcs = ['three.sh'])" > depth2/BUILD

  touch depth/{one,two,three,four,five}.sh depth2/three.sh
  chmod a+x depth/*.sh depth2/*.sh
}

# Running a deps query twice should return results in the same order
# if output is sorted, otherwise just the same results.
function assert_depth_query_idempotence() {
  order_results="$1"
  if $order_results ; then
    add_to_bazelrc "query --order_output=auto"
    universe_arg=""
  else
    add_to_bazelrc "query --order_output=no"
    universe_arg=--universe_scope=//depth:*
  fi
  make_depth_tests
  last_log="$TEST_log.last"
  for run in {1..5}; do
    # Only compare the output stream with the query results.
    mv -f $TEST_log $last_log
    bazel query 'deps(//depth:one, 4)' $universe_arg > $TEST_log \
        || fail "Expected success"
    if [ $run -gt 1 ]; then
      if $order_results ; then
        diff $TEST_log $last_log || \
            fail "Lines differed between query results: $last_log"
      else
        diff <(sort $TEST_log) <(sort $last_log) || \
            fail "Lines differed between sorted query results"
      fi
    fi
  done
  rm -f $last_log || die "Could not remove $last_log"
}

function test_depth_query_idempotence_ordered() {
  assert_depth_query_idempotence true
}

function test_depth_query_idempotence_unordered() {
  assert_depth_query_idempotence false
}

function test_universe_scope_with_without_star() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't mkdir"
  echo "sh_library(name = 'foo')" > foo/BUILD || fail "Couldn't write BUILD"
  bazel query --order_output=no \
      --universe_scope=//foo/... '//foo:BUILD' >& $TEST_log ||
      fail "Expected success"
  # This is documenting current behavior, rather than enforcing a contract. For
  # performance and code simplicity, we return targets if their package was
  # loaded, so //foo:BUILD exists as a target (although its deps and rdeps are
  # unknown).
  expect_log "//foo:BUILD"
  bazel query --order_output=no \
      --universe_scope=//foo/...:* '//foo:BUILD' >& $TEST_log ||
      fail "Expected success"
  expect_log "//foo:BUILD"
}

function test_outside_universe_ok() {
  rm -rf foo
  rm -rf bar
  mkdir -p foo bar || fail "Couldn't mkdir"
  echo "sh_library(name = 'foo', deps = ['//bar:bar'])" > foo/BUILD ||
      fail "Couldn't write BUILD"
  cat <<'EOF' > bar/BUILD || fail "Couldn't write BUILD"
sh_library(name = 'bar')
sh_library(name = 'dep')
sh_library(name = 'top', deps = [':dep'])
EOF
  bazel query --order_output=no \
      --universe_scope=//foo/...:* 'allrdeps(//bar:BUILD)' >& $TEST_log ||
      fail "Expected success"
  # This is documenting current behavior, rather than enforcing a contract. See
  # corresponding comment in test_universe_scope_with_without_star.
  expect_log "//bar:BUILD"
  bazel query --order_output=no \
      --universe_scope=//foo/...:* 'allrdeps(//bar:dep)' >& $TEST_log ||
      fail "Expected success"
  # This is documenting current behavior, rather than enforcing a contract. See
  # corresponding comment in test_universe_scope_with_without_star. In this
  # case, even though we return //bar:dep, we do not see its rdep //bar:top.
  expect_log "//bar:dep"
  expect_not_log "//bar:top"
}

# Since all targets in deps(..., n) are accessible n steps away, none should
# have a minrank greater than n.
function test_minrank_le_depth_bound() {
  make_depth_tests
  for depth in {5..0}; do
    bazel query "deps(//depth:one, $depth)" --output=minrank > $TEST_log \
      || fail "Expected success"
    for rank in $(cut -d' ' -f 1 $TEST_log); do
      [ $rank -le $depth ] || fail "Expected max minrank of $depth, was $rank"
    done
  done
}

function test_skylark_dep_in_sky_query() {
  rm -rf foo
  rm -rf bar
  mkdir -p foo bar || fail "Couldn't make directories"
  echo 'load("//bar:fakerule.bzl", "const")' > foo/BUILD || fail "Couldn't write"
  touch bar/BUILD || fail "Couldn't touch bar/BUILD"
  echo 'const = 2' > bar/fakerule.bzl || fail "Couldn't write fakerule"
  bazel query --universe_scope=//foo/...:* --order_output=no \
      'rbuildfiles(bar/fakerule.bzl)' >& $TEST_log || fail "Expected success"
  expect_log_once "//foo:BUILD"
  expect_not_log "//bar:BUILD"
  expect_not_log "fakerule\.bzl"
}

function test_skylark_regular_file_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "baz" > "foo/baz.bzl" || fail "Couldn't create baz.bzl"
  echo 'sh_library(name = "foo", srcs = ["baz.bzl"])' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_symlink_source_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "moo" > "foo/moo" || fail "Couldn't create moo"
  ln -s "$PWD/foo/moo" "foo/baz.bzl" && [[ -f foo/baz.bzl ]] || fail "Couldn't create baz.bzl symlink"
  echo 'sh_library(name = "foo", srcs = ["baz.bzl"])' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_symlink_target_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "baz" > "foo/baz.bzl" || fail "Couldn't create baz.bzl"
  ln -s "$PWD/foo/baz.bzl" "foo/Moo.java" && [[ -f foo/Moo.java ]] || fail "Couldn't create Moo.java symlink"
  echo 'sh_library(name = "foo", srcs = ["Moo.java"])' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_glob_regular_file_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "baz" > "foo/baz.bzl" || fail "Couldn't create baz.bzl"
  echo 'sh_library(name = "foo", srcs = glob(["*.bzl"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_glob_symlink_source_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "moo" > "foo/moo" || fail "Couldn't create moo"
  ln -s "$PWD/foo/moo" "foo/baz.bzl" && [[ -f foo/baz.bzl ]] || fail "Couldn't create baz.bzl symlink"
  echo 'sh_library(name = "foo", srcs = glob(["*.bzl"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_glob_symlink_target_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo || fail "Couldn't make directories"
  echo "baz" > "foo/baz.bzl" || fail "Couldn't create baz.bzl"
  ln -s "$PWD/foo/baz.bzl" "foo/Moo.java" && [[ -f foo/Moo.java ]] || fail "Couldn't create Moo.java symlink"
  echo 'sh_library(name = "foo", srcs = glob(["*.java"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_recursive_glob_regular_file_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo/bar || fail "Couldn't make directories"
  echo "baz" > "foo/bar/baz.bzl" || fail "Couldn't create baz.bzl"
  echo 'sh_library(name = "foo", srcs = glob(["**/*.bzl"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/bar/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_recursive_glob_symlink_source_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo/bar || fail "Couldn't make directories"
  echo "moo" > "foo/moo" || fail "Couldn't create moo"
  ln -s "$PWD/foo/moo" "foo/bar/baz.bzl" && [[ -f foo/bar/baz.bzl ]] || fail "Couldn't create baz.bzl symlink"
  echo 'sh_library(name = "foo", srcs = glob(["**/*.bzl"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/bar/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_recursive_glob_symlink_target_not_included_in_rbuildfiles() {
  rm -rf foo
  mkdir -p foo/bar || fail "Couldn't make directories"
  echo "baz" > "foo/bar/baz.bzl" || fail "Couldn't create baz.bzl"
  ln -s "$PWD/foo/bar/baz.bzl" "foo/Moo.java" && [[ -f foo/Moo.java ]] || fail "Couldn't create Moo.java symlink"
  echo 'sh_library(name = "foo", srcs = glob(["**/*.java"]))' > foo/BUILD
  bazel query --universe_scope=//...:* --order_output=no \
    'rbuildfiles(foo/bar/baz.bzl)' >& $TEST_log || fail "Expected success"
  expect_not_log "//foo:BUILD"
  # TODO(bazel-team): Remove this once test clean-up is automated.
  # Clean up after ourselves.
  rm -rf foo
}

function test_skylark_subdir_dep_in_sky_query() {
  rm -rf foo
  mkdir -p foo bar/baz || fail "Couldn't make directories"
  echo 'load("//bar:baz/fakerule.bzl", "const")' > foo/BUILD || fail "Couldn't write"
  touch bar/BUILD || fail "Couldn't touch bar/BUILD"
  echo 'const = 2' > bar/baz/fakerule.bzl || fail "Couldn't write fakerule"
  bazel query --universe_scope=//foo/...:* --order_output=no \
      'rbuildfiles(bar/baz/fakerule.bzl)' >& $TEST_log || fail "Expected success"
  expect_log_once "//foo:BUILD"
  expect_not_log "//bar:BUILD"
  expect_not_log "fakerule\.bzl"
}

function test_parent_independent_of_child() {
  rm -rf foo
  mkdir -p foo/subdir || fail "Couldn't make directories"
  echo 'sh_library(name = "sh", data = glob(["**"]))' > foo/BUILD ||
      fail "Couldn't write"
  touch foo/subdir/BUILD || fail "Couldn't touch foo/subdir/BUILD"
  bazel query --universe_scope=//foo/...:* --order_output=no \
      'rbuildfiles(foo/subdir/BUILD)' >& $TEST_log || fail "Expected success"
  expect_log_once "//foo/subdir:BUILD"
  expect_not_log "//foo:BUILD"
}

function test_does_not_fail_horribly_with_file() {
  rm -rf peach
  mkdir -p peach
  cat > peach/BUILD <<EOF
sh_library(name='brighton', deps=[':harken'])
sh_library(name='harken')
EOF

  echo "deps(//peach:brighton)" > query_file
  bazel query --query_file=query_file > $TEST_log

  expect_log "//peach:brighton"
  expect_log "//peach:harken"
}

function test_location_output_not_allowed_with_buildfiles_or_loadfiles() {
  rm -rf foo
  mkdir -p foo
  cat > foo/bzl.bzl <<EOF
x = 2
EOF
  cat > foo/BUILD <<EOF
load('//foo:bzl.bzl', 'x')
sh_library(name='foo')
EOF

  bazel query 'buildfiles(//foo)' >& $TEST_log || fail "Expected success"
  expect_log "//foo:bzl.bzl"
  bazel query 'loadfiles(//foo)' >& $TEST_log || fail "Expected success"
  expect_log "//foo:bzl.bzl"
  bazel query --output=location '//foo' >& $TEST_log || fail "Expected success"
  expect_log "//foo:foo"

  local expected_error_msg="Query expressions involving 'buildfiles' or 'loadfiles' cannot be used with --output=location"
  local expected_exit_code=2
  for query_string in 'buildfiles(//foo)' 'loadfiles(//foo)'
  do
    bazel query --output=location "$query_string" >& $TEST_log \
        && fail "Expected failure"
    exit_code=$?
    expect_log "$expected_error_msg"
    assert_equals "$expected_exit_code" "$exit_code"
  done
}

function test_subdirectory_named_external() {
  mkdir -p foo/external foo/bar
  cat > foo/external/BUILD <<EOF
sh_library(name = 't1')
EOF
  cat > foo/bar/BUILD <<EOF
sh_library(name = 't2')
EOF

  bazel query foo/... >& $TEST_log || fail "Expected success"
  expect_log "//foo/external:t1"
  expect_log "//foo/bar:t2"
}

function test_buildfiles_with_build_bazel() {
  if [ "${PRODUCT_NAME}" != "bazel" ]; then
    return 0
  fi
  rm -rf foo
  mkdir -p foo
  cat > foo/bzl.bzl <<EOF
x = 2
EOF
  cat > foo/BUILD.bazel <<EOF
load('//foo:bzl.bzl', 'x')
sh_library(name='foo')
EOF

  bazel query 'buildfiles(//foo)' >& $TEST_log || fail "Expected success"
  expect_log "//foo:bzl.bzl$"
  expect_log "//foo:BUILD.bazel$"
  expect_not_log "//foo:BUILD$"
}

function test_buildfile_in_genquery() {
  mkdir -p papaya
  cat > papaya/BUILD <<EOF
exports_files(['papaya.bzl'])
EOF
  cat > papaya/papaya.bzl <<EOF
foo = 1
EOF
  mkdir -p honeydew
  cat > honeydew/BUILD <<EOF
load('//papaya:papaya.bzl', 'foo')
sh_library(name='honeydew', deps=[':pineapple'])
sh_library(name='pineapple')
genquery(name='q',
         scope=[':honeydew'],
         strict=0,
         expression='buildfiles(//honeydew:all)')
EOF

  bazel build //honeydew:q >& $TEST_log || fail "Expected success"
  cat bazel-bin/honeydew/q > $TEST_log
  expect_log_once "^//honeydew:BUILD$"
}

function test_genquery_bad_output_formatter() {
  mkdir -p starfruit
  cat > starfruit/BUILD <<EOF
sh_library(name = 'starfruit')
genquery(name='q',
         scope=['//starfruit'],
         expression='//starfruit',
         opts = ["--output=blargh"],)
EOF

  local expected_error_msg="in genquery rule //starfruit:q: Invalid output format 'blargh'. Valid values are: label, label_kind, build, minrank, maxrank, package, location, graph, xml, proto"
  bazel build //starfruit:q >& $TEST_log && fail "Expected failure"
  expect_log "$expected_error_msg"
}

run_suite "${PRODUCT_NAME} query tests"
