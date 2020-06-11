#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# configured_query_test.sh: integration tests for bazel configured query.
# This tests the command line ui of configured query while
# ConfiguredTargetQueryTest tests its internal functionality.

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

add_to_bazelrc "build --package_path=%workspace%"

#### TESTS #############################################################

function test_basic_query() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel cquery "deps(//$pkg:maple)" > output 2>"$TEST_log" || fail "Expected success"

 assert_contains "//$pkg:maple" output
 assert_contains "//$pkg:japanese" output
}

function test_basic_query_output_textproto() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel cquery --output=textproto "deps(//$pkg:maple)" > output 2>"$TEST_log" || fail "Expected success"

 assert_contains "name: \"//$pkg:maple\"" output
 assert_contains "name: \"//$pkg:japanese\"" output
}

function test_basic_query_output_labelkind() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='maple', data=[':japanese'])
cc_binary(name='japanese', srcs = ['japanese.cc'])
EOF

 bazel cquery --output=label_kind "deps(//$pkg:maple)" > output 2>"$TEST_log" ||
 --noimplicit_deps --nohost_deps fail "Expected success"

 assert_contains "sh_library rule //$pkg:maple" output
 assert_contains "cc_binary rule //$pkg:japanese" output
 assert_contains "source file //$pkg:japanese.cc" output
}

function test_respects_selects() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(
    name = "ash",
    deps = select({
        ":excelsior": [":foo"],
        ":americana": [":bar"],
    }),
)
sh_library(name = "foo")
sh_library(name = "bar")
config_setting(
    name = "excelsior",
    values = {"define": "species=excelsior"},
)
config_setting(
    name = "americana",
    values = {"define": "species=americana"},
)
EOF

  bazel cquery "deps(//$pkg:ash)" --define species=excelsior  > output \
    2>"$TEST_log" || fail "Excepted success"
  assert_contains "//$pkg:foo" output
  assert_not_contains "//$pkg:bar" output
}

function test_empty_results_printed() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='redwood', deps=[':sequoia',':sequoiadendron'])
sh_library(name='sequoia')
sh_library(name='sequoiadendron')
EOF

  bazel cquery "somepath(//$pkg:sequoia,//$pkg:sequoiadendron)" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "INFO: Empty query results"
  assert_not_contains "//$pkg:sequoiadendron" output
}

function test_universe_scope_specified() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  # The java_library rule has a host transition on its plugins attribute.
  bazel cquery //$pkg:dep+//$pkg:plugin --universe_scope=//$pkg:my_java \
    > output 2>"$TEST_log" || fail "Excepted success"

  # Find the lines of output for //$pkg:plugin and //$pkg:dep.
  PKG_HOST=$(grep "//$pkg:plugin" output)
  PKG_TARGET=$(grep "//$pkg:dep" output)
  # Trim to just configurations.
  HOST_CONFIG=${PKG_HOST/"//$pkg:plugin"}
  TARGET_CONFIG=${PKG_TARGET/"//$pkg:dep"}
  # Ensure they are are not equal.
  assert_not_equals $HOST_CONFIG $TARGET_CONFIG
}

function test_host_config_output() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery //$pkg:plugin --universe_scope=//$pkg:my_java \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:plugin (HOST)" output
}

function test_transitions_lite() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery "deps(//$pkg:my_java)" --transitions=lite \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:my_java" output
  assert_contains "plugins#//$pkg:plugin#HostTransition" output
}


function test_transitions_full() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery "deps(//$pkg:my_java)" --transitions=full \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:my_java" output
  assert_contains "plugins#//$pkg:plugin#HostTransition" output
}

function write_java_library_build() {
  local -r pkg=$1
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
java_library(
    name = "my_java",
    srcs = ['foo.java'],
    deps = [":dep"],
    plugins = [":plugin"]
)
java_library(name = "dep")
java_plugin(name = "plugin")
EOF
}

# TODO(gregce): --show_config_fragments and RequiredConfigFragmentsProvider
# (the native Java code that powers --show_config_fragments) were originally
# conceived as two pieces of the same functionality. But the former is just
# a provider, which means it can be consumed by other logic. Consider moving
# these tests out of cquery and into proper Java integration tests.

function test_show_transitive_config_fragments() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib",
    srcs = ["mylib.cc"],
)

cc_library(
    name = "cclib_with_py_dep",
    srcs = ["mylib2.cc"],
    data = [":pylib"],
)

py_library(
    name = "pylib",
    srcs = ["pylib.py"],
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=transitive > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib .*CppConfiguration" output
  assert_not_contains "//$pkg:cclib .*PythonConfiguration" output

  assert_contains "//$pkg:cclib_with_py_dep .*CppConfiguration" output
  assert_contains "//$pkg:cclib_with_py_dep .*PythonConfiguration" output

  assert_not_contains "//$pkg:pylib .*CppConfiguration" output
  assert_contains "//$pkg:pylib .*PythonConfiguration" output
}

function test_show_transitive_config_fragments_select() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib",
    srcs = ["mylib.cc"],
    deps = [":cclib_with_select"]
)

config_setting(
    name = "py_reading_condition",
    values = {"build_python_zip": "1"})

cc_library(
    name = "cclib_with_select",
    srcs = select({
        ":py_reading_condition": ["version1.cc"],
        "//conditions:default": ["version2.cc"],
    })
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=transitive > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib .*CppConfiguration" output
  assert_contains "//$pkg:cclib .*PythonOptions" output

  assert_contains "//$pkg:py_reading_condition .*PythonOptions" output

  assert_contains "//$pkg:cclib_with_select .*CppConfiguration" output
  assert_contains "//$pkg:cclib_with_select .*PythonOptions" output
}

function test_show_transitive_config_fragments_alias() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib_with_py_dep",
    srcs = ["mylib2.cc"],
    data = [":myalias"],
)

alias(
    name = "myalias",
    actual = ":pylib"
)

py_library(
    name = "pylib",
    srcs = ["pylib.py"],
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=transitive > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib_with_py_dep .*CppConfiguration" output
  assert_contains "//$pkg:cclib_with_py_dep .*PythonConfiguration" output
}

function test_show_transitive_config_fragments_host_deps() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib_with_py_dep",
    srcs = ["mylib2.cc"],
    data = [":g"],
)

genrule(
    name = "g",
    srcs = [],
    outs = ["g.out"],
    cmd = "echo Hello! > $@",
    tools = [":pylib"])

py_library(
    name = "pylib",
    srcs = ["pylib.py"],
)
EOF

  bazel cquery "//$pkg:cclib_with_py_dep" --show_config_fragments=transitive > \
    output 2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib_with_py_dep .*PythonConfiguration" output
}

function test_show_transitive_config_fragments_through_output_file() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib_with_py_dep",
    srcs = ["mylib2.cc"],
    data = [":g.out"],    # Output file dependency declared here.
)

genrule(
    name = "g",
    srcs = [],
    outs = ["g.out"],
    cmd = "echo Hello! > $@",
    tools = [":pylib"])

py_library(
    name = "pylib",
    srcs = ["pylib.py"],
)
EOF

  bazel cquery "//$pkg:cclib_with_py_dep" --show_config_fragments=transitive > \
    output 2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib_with_py_dep .*PythonConfiguration" output
}

function test_show_direct_config_fragments() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib",
    srcs = ["mylib.cc"],
)

cc_library(
    name = "cclib_with_py_dep",
    srcs = ["mylib2.cc"],
    data = [":pylib"],
)

py_library(
    name = "pylib",
    srcs = ["pylib.py"],
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib .*CppConfiguration" output
  assert_not_contains "//$pkg:cclib .*PythonConfiguration" output

  assert_contains "//$pkg:cclib_with_py_dep .*CppConfiguration" output
  assert_not_contains "//$pkg:cclib_with_py_dep .*PythonConfiguration" output
}

function test_show_direct_config_fragments_select() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_library(
    name = "cclib",
    srcs = ["mylib.cc"],
    deps = [":cclib_with_select"]
)

config_setting(
    name = "py_reading_condition",
    values = {
      "build_python_zip": "1",
      "shell_executable": "foo"
    }
)

cc_library(
    name = "cclib_with_select",
    srcs = select({
        ":py_reading_condition": ["version1.cc"],
        "//conditions:default": ["version2.cc"],
    })
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib .*CppConfiguration" output
  assert_not_contains "//$pkg:cclib .*PythonOptions" output

  assert_contains "//$pkg:py_reading_condition .*PythonOptions" output
  assert_contains "//$pkg:py_reading_condition .*ShellConfiguration\$Options" output

  assert_contains "//$pkg:cclib_with_select .*CppConfiguration" output
  assert_contains "//$pkg:cclib_with_select .*PythonOptions" output
  assert_contains "//$pkg:cclib_with_select .*ShellConfiguration\$Options" output
}

function test_show_config_fragments_select_on_starlark_option() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/defs.bzl <<EOF
def _string_flag_impl(ctx):
    pass

string_flag = rule(
    implementation = _string_flag_impl,
    build_setting = config.string()
)
EOF
  cat > $pkg/BUILD <<EOF
load(":defs.bzl", "string_flag")
string_flag(
    name = "my_flag",
    build_setting_default = "default_value"
)
config_setting(
    name = "is_my_flag_foo",
    flag_values = {":my_flag": "foo"}
)
cc_library(
    name = "cclib_with_select",
    srcs = select({
        ":is_my_flag_foo": ["version_foo.cc"],
        "//conditions:default": ["version_default.cc"],
    })
)
cc_library(
    name = "cclib_plain",
    srcs = ["version_plain.cc"]
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:my_flag .*//$pkg:my_flag" output
  assert_contains "//$pkg:is_my_flag_foo .*//$pkg:my_flag" output
  assert_contains "//$pkg:cclib_with_select .*//$pkg:my_flag" output

  assert_not_contains "//$pkg:cclib_plain .*//$pkg:my_flag" output
}

function test_show_config_fragments_starlark_rule_requires_starlark_option() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/defs.bzl <<EOF
def _string_flag_impl(ctx):
    pass

string_flag = rule(
    implementation = _string_flag_impl,
    build_setting = config.string()
)

def _rule_with_flag_dep_impl(ctx):
    pass

rule_with_flag_dep = rule(
    implementation = _rule_with_flag_dep_impl,
    attrs = {
        "_flagdep": attr.label(default = "//$pkg:my_flag")
    }
)
EOF
  cat > $pkg/BUILD <<EOF
load(":defs.bzl", "rule_with_flag_dep", "string_flag")
string_flag(
    name = "my_flag",
    build_setting_default = "default_value"
)
rule_with_flag_dep(
    name = "my_rule"
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:my_rule .*//$pkg:my_flag" output
}

function test_show_config_fragments_select_on_feature_flag_info_provider() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/defs.bzl <<EOF
def _feature_flag_provider_rule_impl(ctx):
    return [config_common.FeatureFlagInfo(value = "foo")]

feature_flag_provider_rule = rule(
    implementation = _feature_flag_provider_rule_impl,
)
EOF
  cat > $pkg/BUILD <<EOF
load(":defs.bzl", "feature_flag_provider_rule")
feature_flag_provider_rule(name = "foo_feature")

config_setting(
    name = "is_foo_feature",
    flag_values = {":foo_feature": "foo"},
)

cc_library(
    name = "cclib_select_on_foo_feature",
    srcs = ["hi.cc"],
    deps = select({
        ":is_foo_feature": [],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "cclib_no_select",
    srcs = ["heya.cc"]
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct > output \
    2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:is_foo_feature .*//$pkg:foo_feature" output
  assert_contains "//$pkg:cclib_select_on_foo_feature .*//$pkg:foo_feature" output
  assert_not_contains "//$pkg:cclib_no_select .*//$pkg:foo_feature" output

  # Starlark rules exposing FeatureFlagInfo aren't really part of the
  # configuration. You can't "set" them as part of the configuration like you
  # can Starlark flags or Android feature flags. All they do is provide a new
  # interface for rules to select() over the existing configuration.
  #
  # Nevertheless, since you can select() on them as this test illustrates, it's
  # reasonable to "pretend" for the purposes of --show_config_fragments. At best
  # they still show interesting dependencies. At worst they provide a bit more
  # conceptual clutter to wade through when analyzing a build graph.
  #
  # If we're going to support them, it conceptually makes sense to consider
  # //$pkg:foo_feature a config dependency on itself. That doesn't currently
  # work. The only reason is because the most natural place to encode this
  # is to have lib.analysis.RuleConfiguredTargetBuilder check if the rule
  # provides lib.rules.config.ConfigFeatureFlagProvider. But that adds an
  # unwanted dependency from lib.analysis onto lib.rules.
  #
  # Given this use case's unimportance, we just leave things as-is for the sake
  # of simplicity in the wider code base. We can always re-evaluate if needed.
  assert_not_contains "//$pkg:foo_feature .*//$pkg:foo_feature" output
}

function test_show_config_fragments_on_define() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
config_setting(
    name = "is_a_on",
    define_values = {"a": "on"}
)

cc_library(
    name = "cclib_with_select",
    srcs = select({
        ":is_a_on": ["version1.cc"],
        "//conditions:default": ["version2.cc"],
    })
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct --define a=on \
    --define b=on > output 2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:cclib_with_select .*CppConfiguration" output
  assert_contains "//$pkg:cclib_with_select .*--define:a" output
  assert_not_contains "//$pkg:cclib_with_select .*--define:b" output
}

function test_show_config_fragments_on_starlark_required_fragments() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/defs.bzl <<EOF
def _impl(ctx):
  pass

java_requiring_rule = rule(
  implementation = _impl,
  fragments = ["java"],
  attrs = {}
)
EOF
  cat > $pkg/BUILD <<EOF
load("//$pkg:defs.bzl", "java_requiring_rule")
java_requiring_rule(
  name = "buildme"
)
EOF

  bazel cquery "//$pkg:all" --show_config_fragments=direct \
    > output 2>"$TEST_log" || fail "Expected success"
  assert_contains "//$pkg:buildme .*JavaConfiguration" output
}

# Starlark aspects can't directly require configuration fragments. But they can
# have implicit label dependencies on rules that do. This test ensures that
# gets factored in.
#
# Note that cquery doesn't support queries over aspects: `deps(//foo)` won't
# list aspects or traverse their dependencies. This makes it hard to understand
# *why* a rule requires a fragment if only through an aspect. That's an argument
# for making cquery generally aspect-aware.
function test_show_config_fragments_includes_starlark_aspects() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/defs.bzl <<EOF
def _aspect_impl(target, ctx):
    return []

cc_depending_aspect = aspect(
    attrs = {
        # This creates an implicit dependency on a C++ library that
        # only comes through the aspect.
        "_extra_dep": attr.label(default = "//$pkg:cclib"),
    },
    implementation = _aspect_impl,
)

def _impl(ctx):
    pass

simple_rule = rule(
  implementation = _impl,
  attrs = {
    "deps": attr.label_list(aspects = [cc_depending_aspect])
  }
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:defs.bzl", "simple_rule")

simple_rule(
    name = "buildme",
    deps = [":simple_dep"],
)

simple_rule(
    name = "simple_dep",
)

cc_library(
    name = "cclib",
    srcs = ["cclib.cc"],
)
EOF

  # No direct requirement:
  bazel cquery "//$pkg:buildme" --show_config_fragments=direct \
    > output 2>"$TEST_log" || fail "Expected success"
  assert_not_contains "//$pkg:buildme .*CppConfiguration" output

  # But there is a transitive requirement through the aspect:
  bazel cquery "//$pkg:buildme" --show_config_fragments=transitive \
    > output 2>"$TEST_log" || fail "Expected success"
  assert_contains "//$pkg:buildme .*CppConfiguration" output
}

function test_manual_tagged_targets_always_included_for_queries() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
genrule(
  name = "always_build",
  srcs = [],
  outs = ["always_build.out"],
  cmd = "echo hi > $@")
genrule(
  name = "only_build_explicitly",
  tags = ["manual"],
  srcs = [],
  outs = ["only_build_explicitly.out"],
  cmd = "echo hi > $@")
EOF

  bazel cquery "//$pkg:all" > output 2>"$TEST_log" || fail "Expected success"
  assert_contains "//$pkg:always_build" output
  assert_contains "//$pkg:only_build_explicitly" output
}

function test_include_test_suites() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
test_suite(
  name = "my_suite",
  tests = [":my_test"])
cc_test(
  name = "my_test",
  srcs = ["my_test.cc"])
EOF

  bazel cquery "//$pkg:all" > output 2>"$TEST_log" || fail "Expected success"
  assert_contains "//$pkg:my_suite" output
  assert_contains "//$pkg:my_test" output
}

run_suite "${PRODUCT_NAME} configured query tests"
