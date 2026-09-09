#!/usr/bin/env bash
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
# A test for --discard_analysis_cache.

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

javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
jmaptool="$(rlocation "${javabase}/bin/jmap${EXE_EXT}")"

function write_hello_world_files() {
  mkdir -p hello || fail "mkdir hello failed"
  cat >hello/BUILD <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(name = 'hello',
  srcs = ['Hello.java'],
  main_class = 'hello.Hello')
EOF

  cat >hello/Hello.java <<EOF
package hello;
public class Hello {
  public static void main(String[] args) {
    System.out.println("hello!");
  }
}
EOF
}

function set_up() {
  add_rules_java MODULE.bazel
}

#### TESTS #############################################################

function test_compile_helloworld() {
  write_hello_world_files
  bazel run --experimental_ui_debug_all_events --discard_analysis_cache hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_log "Loading package: hello"
  expect_log 'hello!'

  bazel run --experimental_ui_debug_all_events --discard_analysis_cache hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_not_log "Loading package: hello"
  expect_log 'hello!'

  # Check that further incremental builds work fine.
  bazel run --experimental_ui_debug_all_events hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_not_log "Loading package: hello"
  expect_log 'hello!'
}

# Regression test for b/336514394
function test_aspect_after_cache_discard() {
  write_hello_world_files

  mkdir -p aspect
  cat > aspect/aspect.bzl << 'EOF' || fail "Couldn't write aspect.bzl"
def _simple_aspect_impl(target, ctx):
    return []

simple_aspect = aspect(
    implementation=_simple_aspect_impl,
    attr_aspects = ["*"],
)
EOF
  touch aspect/BUILD

  # Build and then discard the cache.
  bazel build \
    --discard_analysis_cache \
    --aspects=aspect/aspect.bzl%simple_aspect \
    hello:hello >&$TEST_log \
      || fail "Build failed"

  # This should rebuild the cache as needed..
  bazel build \
    --discard_analysis_cache \
    --aspects=aspect/aspect.bzl%simple_aspect \
    hello:hello >&$TEST_log \
      || fail "Second build failed"
}

function extract_histogram_count() {
  local histofile="$1"
  local item="$2"
  # We can't use + here because Macs don't recognize it as a special character by default.
  (grep "$item" "$histofile" || echo "") | sed -e 's/^ *[0-9][0-9]*: *\([0-9][0-9]*\) .*$/\1/'
}

function test_aspect_and_configured_target_cleared() {
  # NestedSetCodec can hang on to objects.
  export DONT_SANITY_CHECK_SERIALIZATION=1
  mkdir -p "foo" || fail "Couldn't make directory"
  cat > foo/simpleaspect.bzl <<'EOF' || fail "Couldn't write bzl file"
AspectInfo = provider()
def _simple_aspect_impl(target, ctx):
  result=[]
  for orig_out in target[DefaultInfo].files.to_list():
    aspect_out = ctx.actions.declare_file(orig_out.basename + ".aspect")
    ctx.actions.write(
        output=aspect_out,
        content = "Hello from aspect for %s" % orig_out.basename)
    result += [aspect_out]

  result = depset(result,
      transitive = [src[AspectInfo].aspectouts for src in ctx.rule.attr.srcs])

  return [
      OutputGroupInfo(**{"aspect-out" : result}),
      AspectInfo(aspectouts = result),
  ]

simple_aspect = aspect(implementation=_simple_aspect_impl,
                       attr_aspects = ["srcs"])

def _rule_impl(ctx):
  output = ctx.outputs.out
  ctx.actions.run_shell(
      inputs=[],
      outputs=[output],
      progress_message="Touching output %s" % output,
      command="touch %s" % output.path)

simple_rule = rule(
    implementation =_rule_impl,
    attrs = {"srcs": attr.label_list(aspects=[simple_aspect])},
    outputs={"out": "%{name}.out"}
    )
EOF

cat > foo/BUILD <<'EOF' || fail "Couldn't write BUILD file"
load("//foo:simpleaspect.bzl", "simple_rule")

simple_rule(name = "foo", srcs = [":dep"])
simple_rule(name = "dep", srcs = [])
EOF
  server_pid="$(bazel info server_pid 2>> "$TEST_log")"
  echo "server_pid is ${server_pid}" >> "$TEST_log"
  bazel build //foo:foo >> "$TEST_log" 2>&1 || fail "Expected success"
  new_server_pid="$(bazel info server_pid 2>> "$TEST_log")"
  [[ "$server_pid" == "$new_server_pid" ]] \
      || fail "unequal pids: $server_pid, $new_server_pid"
  "$jmaptool" -histo:live "$server_pid" > histo.txt
  cat histo.txt >> "$TEST_log"
  ct_count="$(extract_histogram_count histo.txt 'RuleConfiguredTarget$')"
  aspect_count="$(extract_histogram_count histo.txt 'lib.packages.Aspect$')"
  [[ "$ct_count" -ge 2 ]] \
      || fail "Too few configured targets: $ct_count. Did you move/rename the class?"
  [[ "$aspect_count" -ge 1 ]] \
      || fail "Too few aspects: $aspect_count. Did you move/rename the class?"
  bazel --batch clean >& "$TEST_log" || fail "Expected success"
  server_pid="$(bazel info server_pid 2> /dev/null)"
  bazel build --discard_analysis_cache //foo:foo >& "$TEST_log" \
      || fail "Expected success"
  "$jmaptool" -histo:live "$server_pid" > histo.txt
  #cat histo.txt >> "$TEST_log"
  ct_count="$(extract_histogram_count histo.txt 'RuleConfiguredTarget$')"
  aspect_count="$(extract_histogram_count histo.txt 'lib.packages.Aspect$')"
  # Several top-level configured targets are allowed to stick around.
  [[ "$ct_count" -le 20 ]] \
      || fail "Too many configured targets: $ct_count"
  [[ "$aspect_count" -eq 0 ]] || fail "Too many aspects: $aspect_count"
  bazel --batch clean >& "$TEST_log" || fail "Expected success"
  server_pid="$(bazel info server_pid 2> /dev/null)"
  bazel build --discard_analysis_cache \
      --aspects foo/simpleaspect.bzl%simple_aspect \
      --output_groups=aspect-out //foo:foo >& "$TEST_log" \
      || fail "Expected success"
  [[ -e "bazel-bin/foo/foo.out.aspect" ]] || fail "Aspect foo not run"
  [[ -e "bazel-bin/foo/dep.out.aspect" ]] || fail "Aspect bar not run"
  # Make sure to clear out garbage, sometimes a spare aspect hangs around.
  bazel info used-heap-size-after-gc >& /dev/null
  "$jmaptool" -histo:live "$server_pid" > histo.txt
  cat histo.txt >> "$TEST_log"
  ct_count="$(extract_histogram_count histo.txt 'RuleConfiguredTarget$')"
  aspect_count="$(extract_histogram_count histo.txt 'lib.packages.Aspect$')"
  # One top-level aspect is allowed to stick around.
  [[ "$aspect_count" -le 1 ]] || fail "Too many aspects: $aspect_count"
  [[ "$ct_count" -le 20 ]] || fail "Too many configured targets: $ct_count"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/30800.
#
# Real-world scenario:
# A C++ toolchain (or system library) is vendored in an external repository,
# and headers (e.g. Windows SDK headers) are reached dynamically via relative
# include search paths (e.g. cxx_builtin_include_directories or -isystem)
# rather than declared target inputs (srcs/hdrs/deps).
#
# On a warm server following an analysis cache discard (e.g. build option change),
# Skymeld re-instantiates IncrementalPackageRoots with an empty map. Because the
# headers are reached dynamically rather than via declared target dependency edges,
# the external package root is not re-registered in the second build's
# TopLevelTargetReadyForSymlinkPlanting event. Post-execution HeaderDiscovery
# then fails to resolve the relative path, causing spurious "undeclared inclusion(s)".
#
# Test setup:
# Setting up a full custom cc_toolchain in an external repository in a shell test
# is heavy and fragile across platforms. As noted in issue #30800, this failure
# affects any header reached via system include paths without a declared target
# dependency. Here we export system_includes via CcInfo from @other_repo//pkg_tool:tool_lib
# pointing to @other_repo//pkg_header to test this exact Skyframe state:
# 1. Build 1 loads @other_repo//pkg_header via target_a, caching PackageValue in Skyframe.
# 2. Build 2 discards analysis cache (--cxxopt change) and builds target_b.
# 3. target_b depends on tool_lib (which exports pkg_header as system_includes,
#    so CppCompileAction validates the inclusion) but does NOT declare a dependency
#    on pkg_header.
# 4. In Build 2, IncrementalPackageRoots has an empty map. With standalone spawn
#    strategy, the compiler finds the header in the execroot. HeaderDiscovery
#    requires the fallback package root lookup in IncrementalPackageRoots to
#    resolve the package root. Without the fallback, HeaderDiscovery fails.
function test_skymeld_external_repo_package_root_preserved_after_discard_analysis_cache() {
  if [[ "${PRODUCT_NAME}" != "bazel" ]]; then
    return 0
  fi

  add_rules_cc MODULE.bazel

  mkdir -p ../other_repo/pkg_header ../other_repo/pkg_tool
  touch ../other_repo/REPO.bazel
  cat > ../other_repo/pkg_header/BUILD << 'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "header_lib",
    hdrs = ["header.h"],
    visibility = ["//visibility:public"],
)
EOF
  cat > ../other_repo/pkg_header/header.h << 'EOF'
#pragma once
#define FOO_VALUE 42
EOF

  cat > ../other_repo/pkg_tool/defs.bzl << 'EOF'
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _tool_lib_impl(ctx):
    return [
        CcInfo(
            compilation_context = cc_common.create_compilation_context(
                system_includes = depset([ctx.label.workspace_root + "/pkg_header"]),
            ),
        ),
    ]

tool_lib = rule(
    implementation = _tool_lib_impl,
)
EOF

  cat > ../other_repo/pkg_tool/BUILD << 'EOF'
load(":defs.bzl", "tool_lib")
tool_lib(
    name = "tool_lib",
    visibility = ["//visibility:public"],
)
EOF

  cat >> MODULE.bazel << 'EOF'
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "other_repo",
    path = "../other_repo",
)
EOF

  mkdir -p foo
  cat > foo/BUILD << 'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "target_a",
    srcs = ["a.cc"],
    deps = ["@other_repo//pkg_header:header_lib"],
)
cc_library(
    name = "target_b",
    srcs = ["b.cc"],
    deps = ["@other_repo//pkg_tool:tool_lib"],
)
EOF

  cat > foo/a.cc << 'EOF'
#include "pkg_header/header.h"
int a() { return FOO_VALUE; }
EOF

  cat > foo/b.cc << 'EOF'
#include "header.h"
int b() { return FOO_VALUE; }
EOF

  bazel build --experimental_merged_skyframe_analysis_execution \
      --lockfile_mode=off \
      //foo:target_a >& "$TEST_log" \
      || fail "First build of target_a failed"

  # In the second build, target_b depends on @other_repo//pkg_tool:tool_lib (so
  # the @other_repo symlink is planted in the execroot), but includes header.h
  # from @other_repo//pkg_header via system_includes.
  # Following analysis cache discard (--cxxopt change), IncrementalPackageRoots
  # has an empty map. With standalone spawn strategy, the compiler accesses the
  # header from the execroot. HeaderDiscovery then requires the fallback
  # package root lookup in IncrementalPackageRoots to resolve the package root.
  bazel build --experimental_merged_skyframe_analysis_execution \
      --lockfile_mode=off \
      --spawn_strategy=standalone \
      --cxxopt=-O3 //foo:target_b >& "$TEST_log" \
      || fail "Second build of target_b failed"
}

run_suite "test for --discard_analysis_cache"

