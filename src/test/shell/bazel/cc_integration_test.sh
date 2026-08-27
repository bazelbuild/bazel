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
# Tests the behavior of C++ rules.

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_include_validation_sandbox_disabled() {
  local workspace="${FUNCNAME[0]}"
  mkdir -p "${workspace}"/lib

  setup_module_dot_bazel "${workspace}/MODULE.bazel"
  add_rules_cc "${workspace}/MODULE.bazel"
  cat >> "${workspace}/BUILD" << EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "foo",
    srcs = ["lib/foo.cc"],
    hdrs = ["lib/foo.h"],
    strip_include_prefix = "lib",
)
EOF
  cat >> "${workspace}/lib/foo.cc" << EOF
#include "foo.h"
EOF

  touch "${workspace}/lib/foo.h"

  cd "${workspace}"
  bazel build --spawn_strategy=standalone //:foo  &>"$TEST_log" \
    || fail "Build failed but should have succeeded"
}

function test_sibling_repository_layout_include_external_repo_output() {
  add_rules_java MODULE.bazel
  add_rules_cc "MODULE.bazel"
  mkdir test
  cat > test/BUILD <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
  name = "foo",
  srcs = ["foo.cc"],
  deps = ["@rules_java//toolchains:jni"],
)
EOF
  cat > test/foo.cc <<'EOF'
#include <jni.h>
#include <stdio.h>

extern "C" JNIEXPORT void JNICALL Java_foo_App_f(JNIEnv *env, jclass clazz, jint x) {
  printf("hello %d\n", x);
}
EOF
  bazel build --experimental_sibling_repository_layout //test:foo > "$TEST_log" \
    || fail "expected build success"
}

# Test writing the exposed args of CPPCompileAction to parameters file
# This is needed to avoid too long commands when the args of one of the target's
# actions are used to run a new action from the aspect. Fixes b/168634763
function test_using_compile_action_args_params_file() {
  add_rules_cc "MODULE.bazel"
  mkdir -p package

  cat > "package/lib.bzl" <<EOF
def _actions_test_impl(target, ctx):
    compile_action = None

    for action in target.actions:
      if action.mnemonic == "CppCompile":
        compile_action = action

    args = compile_action.args[0]
    aspect_out = ctx.actions.declare_file('aspect_out')

    # Passing compile_action.outputs as input to the aspect action to ensure
    # it gets the modified args value after executing the compile action.
    ctx.actions.run_shell(inputs = compile_action.outputs,
                          outputs = [aspect_out],
                          command = "for v in \$@; do echo \$v; done > " + aspect_out.path,
                          arguments = [args])
    return [OutputGroupInfo(out=[aspect_out])]

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  cat > "package/x.cc" <<EOF
#include <stdio.h>
int main() {
  printf("Hello\n");
}
EOF

  cat > "package/BUILD" <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = "x",
  srcs = ["x.cc"],
)
EOF

  # The args should not be written to a file if the experimental flag is not set
  bazel build "package:x" \
      --aspects="//package:lib.bzl%actions_test_aspect" \
      --output_groups=out

  cat "bazel-bin/package/aspect_out" | grep ".params" \
      && fail "CPPCompileAction Args should not have used a params file"

  # Copy the args to be used for validating the params file contents
  cp "bazel-bin/package/aspect_out" "package/expected_args"

  # The args should be written to a file if the experimental flag is set
  bazel build "package:x" \
      --aspects="//package:lib.bzl%actions_test_aspect" \
      --output_groups=out \
      --experimental_use_cpp_compile_action_args_params_file

  cat "bazel-bin/package/aspect_out" | grep ".params" \
      || fail "CPPCompileAction Args should have used a params file"

  # Validate the contents of the params file (with unquoting)
  assert_equals "$(sed 's/\\//g' bazel-bin/package/aspect_out-0.params)" \
      "$(cat package/expected_args)"
}

function test_include_external_genrule_header() {
  add_rules_cc "MODULE.bazel"
  REPO_PATH=$TEST_TMPDIR/repo
  mkdir -p "$REPO_PATH"
  touch "$REPO_PATH/REPO.bazel"
  mkdir "$REPO_PATH/foo"
  cat > "$REPO_PATH/foo/BUILD" <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
  name = "bar",
  srcs = [
    "bar.cc",
    "inc.h",
  ],
)

genrule(
  name = "inc_h",
  srcs = ["inc.txt"],
  outs = ["inc.h"],
  cmd = "cp $< $@",
)
EOF
  cat > "$REPO_PATH/foo/bar.cc" <<'EOF'
#include "foo/inc.h"

int main() {
  sayhello();
}
EOF
  cat > "$REPO_PATH/foo/inc.txt" <<'EOF'
#include <stdio.h>

void sayhello() {
  printf("hello\n");
}
EOF

  cat >> MODULE.bazel <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(name = 'repo', path='$REPO_PATH')
EOF

  bazel build @repo//foo:bar \
    > "$TEST_log" || fail "expected build success"
  bazel build --experimental_sibling_repository_layout @repo//foo:bar \
    > "$TEST_log" || fail "expected build success"
}

function test_reconstructing_cpp_actions_using_shadowed_action() {
  add_rules_cc "MODULE.bazel"
  local package="${FUNCNAME[0]}"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
def _actions_test_impl(target, ctx):
    compile_action = None
    archive_action = None
    link_action = None

    for action in target.actions:
      if action.mnemonic == "CppCompile":
        compile_action = action
      if action.mnemonic == "CppArchive":
        archive_action = action

    if not compile_action or not archive_action:
      fail("Couldn't find compile or archive action.")

    compile_action_outputs = compile_action.outputs.to_list()
    compile_args = ctx.actions.declare_file("compile_args")
    ctx.actions.run_shell(
        outputs = [compile_args],
        command = "echo \$@ > " + compile_args.path,
        arguments = compile_action.args,
    )

    compile_out = ctx.actions.declare_file("compile_out.o")
    ctx.actions.run_shell(
        inputs = [compile_args],
        shadowed_action = compile_action,
        mnemonic = "RecreatedCppCompile",
        outputs = [compile_out],
        command = "\$(cat %s | sed 's|%s|%s|g' | sed 's|%s|%s|g')" % (
            compile_args.path,
            # We need to replace the original output path with something else
            compile_action_outputs[0].path,
            compile_out.path,
            # We need to replace the original .d file output path with something
            # else
            compile_action_outputs[0].path.replace(".o", ".d"),
            compile_out.path + ".d",
        ),
    )

    archive_out = ctx.actions.declare_file("archive_out.a")
    ctx.actions.run_shell(
        shadowed_action = archive_action,
        mnemonic = "RecreatedCppArchive",
        outputs = [archive_out],
        command = "\$@ && cp %s %s" % (
            archive_action.outputs.to_list()[0].path,
            archive_out.path,
        ),
        arguments = archive_action.args,
    )

    return [OutputGroupInfo(out = [
        compile_args,
        compile_out,
        archive_out,
    ])]

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  echo "inline int x() { return 42; }" > "${package}/x.h"
  cat > "${package}/a.cc" <<EOF
#include "${package}/x.h"

int a() { return x(); }
EOF
  cat > "${package}/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
  name = "x",
  hdrs  = ["x.h"],
)

cc_library(
  name = "a",
  srcs = ["a.cc"],
  deps = [":x"],
)
EOF

  # Test that actions are reconstructible under default configuration
  bazel build "${package}:a" \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded"

  # Test that compile actions are reconstructible when using param files
  bazel build "${package}:a" \
      --features=compiler_param_file \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded with --features=compiler_param_file"
}

function test_include_scanning_smoketest() {
  # Make sure there are no packages containing tools/cpp/INCLUDE_HINTS to exercise that case in
  # IncludeHintsFunction.
  rm -rf BUILD tools
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_binary(
  name = 'bin',
  srcs = ['bin.cc'],
  deps = [':spurious_dep'],
)

cc_library(
  name = 'spurious_dep',
  hdrs = ['dep.h'],
)
EOF

  cat > pkg/bin.cc <<EOF
#define NASTY "dep.h"
#include NASTY
int main() { return 0; }
EOF

  touch pkg/dep.h

  bazel build --experimental_unsupported_and_brittle_include_scanning --features=cc_include_scanning //pkg:bin &>"$TEST_log" && fail 'include scanning did not (wrongly) remove dependency' || true
  expect_log "fatal error: '\?dep.h'\?"
}

function test_env_inherit_cc_test() {
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_test(
  name = 'foo_test',
  srcs = ['foo_test.cc'],
  env_inherit = ['FOO'],
)
EOF

  cat > pkg/foo_test.cc <<EOF
#include <stdlib.h>

int main() {
  auto foo = getenv("FOO");
  if (foo == nullptr) {
    return 1;
  }
  return 0;
}
EOF

  bazel test //pkg:foo_test &> "$TEST_log" && fail "Did not fail as expected. ENV leak?" || true
  FOO=1 bazel test //pkg:foo_test &> "$TEST_log" || fail "Should have inherited FOO env."
}

function test_env_attr_cc_binary() {
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = 'foo_bin_with_env',
  srcs = ['foo_test.cc'],
  env = {'FOO': 'bar'},
)

cc_binary(
  name = 'foo_bin',
  srcs = ['foo_test.cc'],
)
EOF

  cat > pkg/foo_test.cc <<EOF
#include <stdlib.h>

int main() {
  auto foo = getenv("FOO");
  if (foo == nullptr) {
    return 1;
  }
  return 0;
}
EOF

  bazel run //pkg:foo_bin &> "$TEST_log" && fail "Did not fail as expected. ENV leak?" || true
  bazel run //pkg:foo_bin_with_env &> "$TEST_log" || fail "Should have used env attr."
}

function external_cc_test_setup() {
  add_rules_cc "MODULE.bazel"
  cat >> MODULE.bazel <<'EOF'
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name = "other_repo",
  path = "other_repo",
)
EOF

  mkdir -p other_repo
  touch other_repo/REPO.bazel

  mkdir -p other_repo/lib
  cat > other_repo/lib/BUILD <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
  name = "lib",
  srcs = ["lib.cpp"],
  hdrs = ["lib.h"],
  visibility = ["//visibility:public"],
)
EOF
  cat > other_repo/lib/lib.h <<'EOF'
void print_greeting();
EOF
  cat > other_repo/lib/lib.cpp <<'EOF'
#include <cstdio>
void print_greeting() {
  printf("Hello, world!\n");
}
EOF

  mkdir -p other_repo/test
  cat > other_repo/test/BUILD <<'EOF'
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = ["//lib"],
)
EOF
  cat > other_repo/test/test.cpp <<'EOF'
#include "lib/lib.h"
int main() {
  print_greeting();
}
EOF
}

function test_external_cc_test_sandboxed() {
  if is_windows; then
    return 0
  fi

  external_cc_test_setup

  bazel test \
      --test_output=errors \
      --strategy=sandboxed \
      @other_repo//test >& $TEST_log || fail "Test should pass"
}

function test_external_cc_test_sandboxed_sibling_repository_layout() {
  if is_windows; then
    return 0
  fi

  external_cc_test_setup

  bazel test \
      --test_output=errors \
      --strategy=sandboxed \
      --experimental_sibling_repository_layout \
      @other_repo//test >& $TEST_log || fail "Test should pass"
}

function test_external_cc_test_local() {
  external_cc_test_setup

  bazel test \
      --test_output=errors \
      --strategy=local \
      @other_repo//test >& $TEST_log || fail "Test should pass"
}

function test_external_cc_test_local_sibling_repository_layout() {
  external_cc_test_setup

  bazel test \
      --test_output=errors \
      --strategy=local \
      --experimental_sibling_repository_layout \
      @other_repo//test >& $TEST_log || fail "Test should pass"

  # Test cc compile action can hit the action cache. See
  # https://github.com/bazelbuild/bazel/issues/17819
  bazel shutdown

  bazel test \
      --test_output=errors \
      --strategy=local \
      --experimental_sibling_repository_layout \
      @other_repo//test >& $TEST_log || fail "Test should pass"
  expect_log "1 process: .*1 internal"
}

function test_bazel_current_repository_define() {
  add_rules_cc "MODULE.bazel"
  cat >> MODULE.bazel <<'EOF'
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name = "other_repo",
  path = "other_repo",
)
EOF

  mkdir -p pkg
  cat > pkg/BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_library(
  name = "library",
  srcs = ["library.cpp"],
  hdrs = ["library.h"],
  implementation_deps = ["@bazel_tools//tools/cpp/runfiles"],
  visibility = ["//visibility:public"],
)

cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = [
    ":library",
    "@bazel_tools//tools/cpp/runfiles",
  ],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = [
    ":library",
    "@bazel_tools//tools/cpp/runfiles",
  ],
)
EOF

  cat > pkg/library.cpp <<'EOF'
#include "library.h"
#include <iostream>
void print_repo_name() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
}
EOF

  cat > pkg/library.h <<'EOF'
void print_repo_name();
EOF

  cat > pkg/binary.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > pkg/test.cpp <<'EOF'
#include <iostream>
#include "library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  mkdir -p other_repo
  touch other_repo/REPO.bazel

  mkdir -p other_repo/pkg
  cat > other_repo/pkg/BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_binary(
  name = "binary",
  srcs = ["binary.cpp"],
  deps = [
    "@//pkg:library",
    "@bazel_tools//tools/cpp/runfiles",
  ],
)

cc_test(
  name = "test",
  srcs = ["test.cpp"],
  deps = [
    "@//pkg:library",
    "@bazel_tools//tools/cpp/runfiles",
  ],
)
EOF

  cat > other_repo/pkg/binary.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  cat > other_repo/pkg/test.cpp <<'EOF'
#include <iostream>
#include "pkg/library.h"
int main() {
  std::cout << "in " << __FILE__ << ": '" << BAZEL_CURRENT_REPOSITORY << "'" << std::endl;
  print_repo_name();
}
EOF

  bazel run //pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in pkg/binary.cpp: ''"
  expect_log "in pkg/library.cpp: ''"

  bazel test --test_output=streamed //pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in pkg/test.cpp: ''"
  expect_log "in pkg/library.cpp: ''"

  bazel run @other_repo//pkg:binary &>"$TEST_log" || fail "Run should succeed"
  expect_log "in external/+local_repository+other_repo/pkg/binary.cpp: '+local_repository+other_repo'"
  expect_log "in pkg/library.cpp: ''"

  bazel test --test_output=streamed \
    @other_repo//pkg:test &>"$TEST_log" || fail "Test should succeed"
  expect_log "in external/+local_repository+other_repo/pkg/test.cpp: '+local_repository+other_repo'"
  expect_log "in pkg/library.cpp: ''"
}

function test_compiler_flag_gcc() {
  # The default macOS toolchain always uses XCode's clang.
  if is_darwin; then
    return 0
  fi

  type -P gcc || return 0

  add_rules_cc "MODULE.bazel"
  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
config_setting(
    name = "gcc_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "gcc"},
)

cc_binary(
  name = "main",
  srcs = select({":gcc_compiler": ["main.cc"]}),
)
EOF
  cat > main.cc <<'EOF'
int main() {}
EOF

  bazel build //:main --repo_env=CC=gcc || fail "Expected compiler flag to have value 'gcc'"
}

function test_compiler_flag_clang() {
  type -P clang || return 0

  add_rules_cc "MODULE.bazel"
  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
config_setting(
    name = "clang_compiler",
    flag_values = {"@bazel_tools//tools/cpp:compiler": "clang"},
)

cc_binary(
  name = "main",
  srcs = select({":clang_compiler": ["main.cc"]}),
)
EOF
  cat > main.cc <<'EOF'
int main() {}
EOF

  bazel build //:main --repo_env=CC=clang || fail "Expected compiler flag to have value 'clang'"
}

function test_bazel_cxxopts() {
  add_rules_cc "MODULE.bazel"
  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = "main_c",
  srcs = ["main.c"],
)
cc_binary(
  name = "main_cpp",
  srcs = ["main.cpp"],
)
EOF
  cat > main.c <<'EOF'
#include <stdlib.h>
int main() {
  exit(EXIT_CODE);
}
EOF
  cat > main.cpp <<'EOF'
#include <stdlib.h>
int main() {
  exit(EXIT_CODE);
}
EOF

  bazel build //:main_c \
    --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1 \
    --repo_env=BAZEL_CXXOPTS=-DEXIT_CODE=0 && fail "Expected C compilation to fail"
  bazel run //:main_cpp \
    --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1 \
    --repo_env=BAZEL_CXXOPTS=-DEXIT_CODE=0 || fail "Expected C++ compilation to pass"
}

function test_bazel_conlyopts() {
  add_rules_cc "MODULE.bazel"
  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = "main_c",
  srcs = ["main.c"],
)
cc_binary(
  name = "main_cpp",
  srcs = ["main.cpp"],
)
EOF
  cat > main.c <<'EOF'
#include <stdlib.h>
int main() {
  exit(EXIT_CODE);
}
EOF
  cat > main.cpp <<'EOF'
#include <stdlib.h>
int main() {
  exit(EXIT_CODE);
}
EOF

  bazel build //:main_cpp \
    --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1 \
    --repo_env=BAZEL_CONLYOPTS=-DEXIT_CODE=0 && fail "Expected C++ compilation to fail"
  bazel run //:main_c \
    --repo_env=BAZEL_USE_CPP_ONLY_TOOLCHAIN=1 \
    --repo_env=BAZEL_CONLYOPTS=-DEXIT_CODE=0 || fail "Expected C compilation to pass"
}

function test_cc_test_no_target_coverage_dep() {
  # Regression test for https://github.com/bazelbuild/bazel/issues/16961
  add_rules_cc "MODULE.bazel"
  cat >> MODULE.bazel <<'EOF'
remote_coverage_tools_extension = use_extension("@bazel_tools//tools/test:extensions.bzl", "remote_coverage_tools_extension")
use_repo(remote_coverage_tools_extension, "remote_coverage_tools")
EOF

  local package="${FUNCNAME[0]}"
  mkdir -p "${package}"

  cat > "${package}"/BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_test(
  name = "test",
  srcs = ["test.cc"],
)
EOF
  touch "${package}"/test.cc

  out=$(bazel cquery --collect_code_coverage \
   "deps(//${package}:test) intersect config(@remote_coverage_tools//:all, target)")
  if [[ -n "$out" ]]; then
    fail "Expected no dependency on lcov_merger in the target configuration, but got: $out"
  fi
}

function test_cc_test_no_coverage_tools_dep_without_coverage() {
  add_rules_cc "MODULE.bazel"

  cat >> MODULE.bazel <<'EOF'
remote_coverage_tools_extension = use_extension("@bazel_tools//tools/test:extensions.bzl", "remote_coverage_tools_extension")
use_repo(remote_coverage_tools_extension, "remote_coverage_tools")
EOF

  # Regression test for https://github.com/bazelbuild/bazel/issues/16961 and
  # https://github.com/bazelbuild/bazel/issues/15088.
  local package="${FUNCNAME[0]}"
  mkdir -p "${package}"

  cat > "${package}"/BUILD.bazel <<'EOF'
load("@rules_cc//cc:cc_test.bzl", "cc_test")
cc_test(
  name = "test",
  srcs = ["test.cc"],
)
EOF
  touch "${package}"/test.cc

  out=$(bazel cquery "somepath(//${package}:test,@remote_coverage_tools//:all)")
  if [[ -n "$out" ]]; then
    fail "Expected no dependency on remote coverage tools, but got: $out"
  fi
}

# sanitizer features are opt-in so we check if the sanitizer library is
# installed and skip the test if it isn't (e.g. centos-7-openjdk-11-gcc-10)
function __is_installed() {
  local lib="$1"

  if is_linux; then
    return $(ldconfig -p | grep -q "$lib")
  fi

  # assume installed for darwin
}

function test_cc_toolchain_asan_feature() {
  local feature=asan
  __is_installed "lib$feature" || return 0

  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = 'example',
  srcs = ['example.cc'],
  features = ['$feature'],
)
EOF

  # some versions of clang will optimize away the pointer assignment and
  # dereference without volatile
  # https://godbolt.org/z/of8cr3P8q
  cat > pkg/example.cc <<EOF
int main() {
  volatile int* p;

  {
    volatile int x = 0;
    p = &x;
  }

  return *p;
}
EOF

  bazel run //pkg:example &> "$TEST_log" && fail "Should have failed due to $feature" || true
  expect_log "ERROR: AddressSanitizer: stack-use-after-scope"
}

function test_cc_toolchain_tsan_feature() {
  local feature=tsan
  __is_installed "lib$feature" || return 0

  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = 'example',
  srcs = ['example.cc'],
  features = ['$feature'],
)
EOF

  cat > pkg/example.cc <<EOF
#include <thread>

int value = 0;

void increment() {
  ++value;
}

int main() {
  std::thread t1(increment);
  std::thread t2(increment);
  t1.join();
  t2.join();

  return value;
}
EOF

  bazel run //pkg:example &> "$TEST_log" && fail "Should have failed due to $feature" || true
  # TODO: we used to expect "WARNING: ThreadSanitizer: data race" here, but that
  # has suddenly started failing on Ubuntu on Bazel CI (see
  # https://buildkite.com/bazel/google-bazel-presubmit/builds/92979). We should
  # figure out what's going on and fix this check eventually.
  expect_log "ThreadSanitizer: "
}

function test_cc_toolchain_ubsan_feature() {
  local feature=ubsan
  __is_installed "lib$feature" || return 0

  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = 'example',
  srcs = ['example.cc'],
  features = ['$feature'],
)
EOF

  cat > pkg/example.cc <<EOF
int main() {
  int array[10];
  return array[10];
}
EOF

  bazel run //pkg:example &> "$TEST_log" && fail "Should have failed due to $feature" || true
  expect_log "runtime error: index 10 out of bounds"
}

function setup_find_optional_cpp_toolchain() {

  add_platforms "MODULE.bazel"

  mkdir -p pkg

  cat > pkg/BUILD <<'EOF'
load(":rules.bzl", "my_rule")

my_rule(
    name = "my_rule",
)

platform(
    name = "exotic_platform",
    constraint_values = [
        "@platforms//cpu:wasm64",
        "@platforms//os:windows",
    ],
)
EOF

  cat > pkg/rules.bzl <<'EOF'
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")

def _my_rule_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.name)
    toolchain = find_cpp_toolchain(ctx, mandatory = False)
    if toolchain:
        ctx.actions.write(out, "Toolchain found")
    else:
        ctx.actions.write(out, "Toolchain not found")
    return [DefaultInfo(files = depset([out]))]

my_rule = rule(
    implementation = _my_rule_impl,
    attrs = {
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:optional_current_cc_toolchain",
        ),
    },
    toolchains = use_cpp_toolchain(mandatory = False),
)
EOF
}

function test_find_optional_cpp_toolchain_present() {
  setup_find_optional_cpp_toolchain

  bazel build //pkg:my_rule &> "$TEST_log" || fail "Build failed"
  assert_contains "Toolchain found" bazel-bin/pkg/my_rule
}

function test_find_optional_cpp_toolchain_not_present() {
  setup_find_optional_cpp_toolchain

  bazel build //pkg:my_rule --platforms=//pkg:exotic_platform \
    &> "$TEST_log" || fail "Build failed"
  assert_contains "Toolchain not found" bazel-bin/pkg/my_rule
}

function test_no_cpp_stdlib_linked_to_c_library() {
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = 'example',
  srcs = ['example.c'],
)
EOF
  cat > pkg/example.c <<'EOF'
int main() {}
EOF

  bazel build //pkg:example &> "$TEST_log" || fail "Build failed"
  if is_darwin; then
    otool -L bazel-bin/pkg/example &> "$TEST_log" || fail "otool failed"
    expect_log 'libc'
    expect_not_log 'libc\+\+'
  else
    ldd bazel-bin/pkg/example &> "$TEST_log" || fail "ldd failed"
    expect_log 'libc'
    expect_not_log 'libstdc\+\+'
  fi
}

function test_parse_headers_unclean() {
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(name = "lib", hdrs = ["lib.h"])
EOF
  cat > pkg/lib.h <<'EOF'
// Missing include of cstdint, which defines uint8_t.
uint8_t foo();
EOF

  bazel build -s --process_headers_in_dependencies --features parse_headers \
    //pkg:lib &> "$TEST_log" && fail "Build should have failed due to unclean headers"
  expect_log "Compiling pkg/lib.h"
  expect_log "error:.*uint8_t"

  bazel build -s --process_headers_in_dependencies \
    //pkg:lib &> "$TEST_log" || fail "Build should have passed"
}

function test_parse_headers_clean() {
  add_rules_cc "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
package(features = ["parse_headers"])
cc_library(name = "lib", hdrs = ["lib.h"])
EOF
  cat > pkg/lib.h <<'EOF'
#include <cstdint>
uint8_t foo();
EOF

  bazel build -s --process_headers_in_dependencies \
    //pkg:lib &> "$TEST_log" || fail "Build should have passed"
  expect_log "Compiling pkg/lib.h"
}

function test_tree_artifact_sources_in_no_deps_library() {
  add_rules_shell "MODULE.bazel"
  add_rules_cc "MODULE.bazel"

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

load("generate.bzl", "generate_source")
sh_binary(
    name = "generate_tool",
    srcs = ["generate.sh"],
)

generate_source(
    name = "generated_source",
    tool = ":generate_tool",
    output_dir = "generated",
)

cc_library(
    name = "hello_world",
    srcs = [":generated_source"],
    hdrs = [":generated_source"],
)

cc_test(
    name = "testCodegen",
    srcs = ["testCodegen.cpp"],
    deps = [":hello_world"],
)
EOF
  cat > pkg/generate.bzl <<'EOF'
def _generate_source_impl(ctx):
    output_dir = ctx.attr.output_dir
    files = ctx.actions.declare_directory(output_dir)

    ctx.actions.run(
        inputs = [],
        outputs = [files],
        arguments = [files.path],
        executable = ctx.executable.tool
    )

    return [
        DefaultInfo(files = depset([files]))
    ]


generate_source = rule(
    implementation = _generate_source_impl,
    attrs = {
        "output_dir": attr.string(),
        "tool": attr.label(executable = True, cfg = "exec")
    }
)
EOF
  cat > pkg/generate.sh <<'EOF2'
#!/usr/bin/env bash

OUTPUT_DIR=$1

cat << EOF > $OUTPUT_DIR/test.hpp
#pragma once
void hello_world();
EOF


cat << EOF > $OUTPUT_DIR/test.cpp
#include "test.hpp"
#include <cstdio>
void hello_world()
{
    puts("Hello World!");
}
EOF
EOF2
  chmod +x pkg/generate.sh
  cat > pkg/testCodegen.cpp <<'EOF'
#include "pkg/generated/test.hpp"

int main() {
    hello_world();
    return 0;
}
EOF

  bazel build //pkg:testCodegen &> "$TEST_log" || fail "Build failed"
}

function test_extend_cc_binary_with_dynamic_deps() {
  add_rules_cc "MODULE.bazel"
  mkdir -p pkg
  cat >pkg/BUILD <<'EOF'
load("my_cc_binary.bzl", "my_cc_binary")

constraint_setting(name = "foo")
constraint_value(name = "never_selected", constraint_setting = ":foo")

my_cc_binary(
    name = "hello",
    srcs = ["main.cpp"],
    # Ensure that the select has no effect but can't be simplified.
    dynamic_deps = select({":never_selected": ["unused"], "//conditions:default": []}),
)
EOF

  cat >pkg/my_cc_binary.bzl << 'EOF'
load("@rules_cc//cc/private/rules_impl:cc_binary.bzl", "cc_binary")
def _my_cc_binary_impl(ctx):
  print("Hello from my_cc_binary")
  return ctx.super()

my_cc_binary = rule(
  implementation = _my_cc_binary_impl,
  parent = cc_binary,
)
EOF

  cat >pkg/main.cpp <<'EOF'
#include <iostream>

int main() {
  std::cout << "Hello from main.cpp" << std::endl;
  return 0;
}
EOF

  bazel run //pkg:hello &> $TEST_log || fail "Expected success"
  expect_log "Hello from my_cc_binary"
  expect_log "Hello from main.cpp"
}

function test_cpp20_modules_with_clang() {
  type -P clang || return 0
  # Check if clang version is less than 17
  clang_version=$(clang --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
  if [[ -n "$clang_version" ]]; then
    major_version=$(echo "$clang_version" | cut -d. -f1)
    if [[ "$major_version" -lt 17 ]]; then
      return 0
    fi
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    return 0
  fi

  add_rules_cc "MODULE.bazel"

  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

package(features = ["cpp_modules"])

cc_library(
  name = "base",
  module_interfaces = ["base.cppm"],
)
cc_library(
  name = "foo",
  module_interfaces = ["foo.cppm"],
  deps = [":base"]
)
cc_library(
  name = "bar",
  module_interfaces = ["bar.cppm"],
  deps = [":base"]
)
cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [":foo", ":bar"]
)
EOF
  cat > main.cc <<'EOF'
import foo;
import bar;

void f() {
  f_foo();
  f_bar();
}

int main() {
  f();
  return 0;
}
EOF
  cat > foo.cppm <<'EOF'
export module foo;
import base;

export void f_foo() {
  f_base();
}
EOF
  cat > bar.cppm <<'EOF'
export module bar;
import base;

export void f_bar() {
  f_base();
}
EOF
  cat > base.cppm <<'EOF'
export module base;

export void f_base() {
}
EOF

  bazel build //:main --experimental_cpp_modules --repo_env=CC=clang --copt=-std=c++20 --disk_cache=disk &> $TEST_log || fail "Expected build C++20 Modules success with compiler 'clang'"

  # Verify that the build can hit the cache without action cycles.
  bazel clean || fail "Expected clean success"
  bazel build //:main --experimental_cpp_modules --repo_env=CC=clang --copt=-std=c++20 --disk_cache=disk &> $TEST_log || fail "Expected build C++20 Modules success with compiler 'clang'"
  expect_log "17 disk cache hit"
}

function test_external_repo_lto() {
  add_rules_cc "MODULE.bazel"
  REPO_PATH=$TEST_TMPDIR/repo
  mkdir -p "$REPO_PATH"
  touch "$REPO_PATH/REPO.bazel"
  mkdir "$REPO_PATH/foo"
  cat > "$REPO_PATH/foo/BUILD" <<'EOF'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
  name = "foo",
  srcs = [
    "foo.cc",
  ],
  hdrs = [
    "foo.h",
  ],
)
EOF
  cat > "$REPO_PATH/foo/foo.cc" <<'EOF'
#include "foo.h"

int main() {
  sayhello();
}
EOF
  cat > "$REPO_PATH/foo/foo.h" <<'EOF'
#include <stdio.h>

void sayhello() {
  printf("hello\n");
}
EOF

  cat >> MODULE.bazel <<EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(name = 'repo', path='$REPO_PATH')
EOF
  cat > BUILD <<'EOF'
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = ["@repo//foo:foo"],
)
EOF
  cat > main.cc <<'EOF'
#include "foo/foo.h"
int main() {
  sayhello();
}
EOF

  bazel build --repo_env=CC=clang --features=thin_lto @repo//foo \
    > "$TEST_log" || fail "expected build success"
  bazel build --repo_env=CC=clang --features=thin_lto --experimental_sibling_repository_layout @repo//foo \
    > "$TEST_log" || fail "expected build success"
}

run_suite "cc_integration_test"
