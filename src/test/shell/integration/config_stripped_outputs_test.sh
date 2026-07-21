#!/usr/bin/env bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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


##############################################################################
# Tests stripping config prefixes from output paths for better caching.
#
# What is this?
# -------------------
# Output paths are the paths of files created in a build. For example:
# "bazel-out/x86-fastbuild/bin/myproj/my.output".
#
# The config prefix is the "/x86-fastbuild/" part. That includes the CPU and
# compilation mode, which means changing --cpu or --compilation_mode invalidates
# action cache hits even for actions that don't care about those values.
#
# This tests an experimental feature that strips those prefixes for such
# actions. So they run with "bazel-out/bin/myproject/my.output" and thus get
# better caching.

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
source "$(rlocation "io_bazel/src/test/shell/integration/config_stripped_outputs_lib.sh")" \
  || { echo "config_stripped_outputs_lib.sh not found!" >&2; exit 1; }

add_to_bazelrc "build --package_path=%workspace%"

# This is what triggers config path stripping.
add_to_bazelrc "build --experimental_output_paths=strip"

function set_up() {
  add_rules_java MODULE.bazel
}

function is_bazel() {
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"
  if [ $bazel_out == "bazel-out" ]; then
    # True for Bazel, false for Blaze/Google.
    return 0
  else
    return 1
  fi
}

if is_bazel; then
  bazel_bin=bazel-bin
  # TODO: Remove these lines when Javac actions use multiplex worker sandboxing by default in Bazel.
  add_to_bazelrc "build --strategy=Javac=worker"
  add_to_bazelrc "build --worker_sandboxing"
  add_to_bazelrc "build --noexperimental_worker_multiplex"
else
  bazel_bin=blaze-bin
fi

# Tests built-in Java support for stripping config path prefixes from
# platform-independent actions.
function test_builtin_java_support() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/BUILD" <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
load("@rules_java//java:java_library.bzl", "java_library")
java_library(
  name = "mylib",
  srcs = ["MyLib.java"],
)
java_binary(
  name = "mybin",
  srcs = ["MyBin.java"],
  deps = [":mylib"],
  main_class = "main.MyBin",
)
EOF

  cat > "$pkg/MyLib.java" <<EOF
package mylib;
public class MyLib {
  public static void runMyLib() {
    System.out.println("MyLib checking in.");
  }
}
EOF

  cat > "$pkg/MyBin.java" <<EOF
package main;
import mylib.MyLib;
public class MyBin {
  public static void main(String[] argv) {
    MyLib.runMyLib();
    System.out.println("MyBin running the main binary.");
  }
}
EOF

  # Verify the build succeeds:
  bazel clean
  bazel build -s "//$pkg:mybin" 2>"$TEST_log" || fail "Expected success"

  # Verify these output paths are stripped as expected:
  # java_library .jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/libmylib.jar"
  # java_library header jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/libmylib-hjar.jar"
  # java_binary .jar compilation:
  assert_paths_stripped "$TEST_log" "/bin/$pkg/mybin.jar"
}

function write_java_classpath_reduction_files() {
  local -r pkg="$1"
  mkdir -p "$pkg/java/hello/" || fail "Expected success"
  cat > "$pkg/java/hello/A.java" <<'EOF'
package hello;
public class A {
  public void f(B b) { b.getC().getD(); }
}
EOF
  cat > "$pkg/java/hello/B.java" <<'EOF'
package hello;
public class B {
  public C getC() { return null; }
}
EOF
  cat > "$pkg/java/hello/C.java" <<'EOF'
package hello;
public class C {
  public D getD() { return null; }
}
EOF
  cat > "$pkg/java/hello/D.java" <<'EOF'
package hello;
public class D {}
EOF
  cat > "$pkg/java/hello/BUILD" <<'EOF'
load("@rules_java//java:java_library.bzl", "java_library")
java_library(name='a', srcs=['A.java'], deps = [':b'])
java_library(name='b', srcs=['B.java'], deps = [':c'])
java_library(name='c', srcs=['C.java'], deps = [':d'])
java_library(name='d', srcs=['D.java'])
EOF
}

function test_inmemory_jdeps_support() {
  local -r pkg="${FUNCNAME[0]}"
  write_java_classpath_reduction_files "$pkg"

  bazel clean
  bazel build --experimental_output_paths=strip \
    //"$pkg"/java/hello:a -s 2>"$TEST_log" \
    || fail "Expected success"

  # java_library .jar compilation:
  assert_paths_stripped "$TEST_log" "$pkg/java/hello/liba.jar-0.params"
  # java_library header jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/java/hello/libb-hjar.jar"
  # jdeps files should contain the original paths since they are read by downstream actions that may
  # not use path mapping.
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/liba.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libb.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libb-hjar.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libc.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libc-hjar.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libd.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libd-hjar.jdeps"
}

function test_multiple_configs() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/rules"
  cat > $pkg/rules/defs.bzl <<EOF
LocationInfo = provider(fields = ["location"])

def _location_setting_impl(ctx):
    return LocationInfo(location = ctx.build_setting_value)

location_setting = rule(
    implementation = _location_setting_impl,
    build_setting = config.string(),
)

def _location_transition_impl(settings, attr):
    return {"//$pkg/rules:location": attr.location}

_location_transition = transition(
    implementation = _location_transition_impl,
    inputs = [],
    outputs = ["//$pkg/rules:location"],
)

def _bazelcon_greeting_impl(ctx):
    content = """
package com.example.{package};

import com.example.BaseLib;

public class Lib {{
  public static String getGreeting() {{
    return BaseLib.getGreeting("{location}");
  }}
}}
""".format(
        package = ctx.attr.name,
        location = ctx.attr.location,
    )
    file = ctx.actions.declare_file("Lib.java")
    ctx.actions.write(file, content)
    return [
        DefaultInfo(files = depset([file])),
    ]

bazelcon_greeting = rule(
    _bazelcon_greeting_impl,
    cfg = _location_transition,
    attrs = {
        "location": attr.string(),
    },
)
EOF
  cat > $pkg/rules/BUILD << 'EOF'
load(":defs.bzl", "location_setting")

location_setting(
    name = "location",
    build_setting_default = "",
)
EOF

  mkdir -p $pkg/java
  cat > $pkg/java/BUILD <<EOF
load("//$pkg/rules:defs.bzl", "bazelcon_greeting")
load("@rules_java//java:java_binary.bzl", "java_binary")
load("@rules_java//java:java_library.bzl", "java_library")
java_binary(
    name = "Main",
    srcs = ["Main.java"],
    deps = [":lib"],
)
java_library(
    name = "lib",
    srcs = [
        ":munich",
        ":new_york",
    ],
    deps = [":base_lib"],
)
bazelcon_greeting(
    name = "munich",
    location = "Munich",
)
bazelcon_greeting(
    name = "new_york",
    location = "New York",
)
java_library(
    name = "base_lib",
    srcs = ["BaseLib.java"],
)
EOF
  cat > $pkg/java/Main.java <<'EOF'
package com.example;
public class Main {
  public static void main(String[] args) {
    System.out.println(com.example.new_york.Lib.getGreeting());
    System.out.println(com.example.munich.Lib.getGreeting());
  }
}
EOF
  cat > $pkg/java/BaseLib.java <<'EOF'
package com.example;
public class BaseLib {
  public static String getGreeting(String location) {
    return "Hello from " + location;
  }
}
EOF

  bazel clean
  bazel build --experimental_output_paths=strip \
    //$pkg/java:Main -s 2>"$TEST_log" \
    || fail "Expected success"

  # java_binary compilation
  assert_paths_stripped "$TEST_log" "bin/$pkg/java/Main.jar"
  # base_lib .jar compilation
  assert_paths_stripped "$TEST_log" "$pkg/java/libbase_lib.jar-0.params"
  # base_lib header jar compilation
  assert_paths_stripped "$TEST_log" "bin/$pkg/java/libbase_lib-hjar.jar"
  # lib .jar compilation should not be stripped due to conflicting paths
  assert_contains "\(bazel\|blaze\)-out/[^/]\+-fastbuild/bin/$pkg/java/liblib.jar-0.params" "$TEST_log"
  # lib header jar compilation should not be stripped due to conflicting paths
  assert_contains "--output \(bazel\|blaze\)-out/[^/]\+-fastbuild/bin/$pkg/java/liblib-hjar.jar" "$TEST_log"
  # jdeps files should contain the original paths since they are read by downstream actions that may
  # not use path mapping.
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/Main.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/liblib.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/liblib-hjar.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/libbase_lib.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/libbase_lib-hjar.jdeps"
}

function test_direct_classpath() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/java/hello/" || fail "Expected success"
  # When compiling C, the direct classpath optimization in Turbine embeds information about the
  # dependency D into the header jar, which then results in the path to Ds header jar being included
  # in the jdeps file for B. The full compilation action for A requires the header jar for D  and
  # thus the path to it in the jdeps file of B has to be unmapped properly for the reduced classpath
  # created for A to contain it.
  cat > "$pkg/java/hello/A.java" <<'EOF'
package hello;
public class A extends B {}
EOF
  cat > "$pkg/java/hello/B.java" <<'EOF'
package hello;
public class B extends C {}
EOF
  cat > "$pkg/java/hello/C.java" <<'EOF'
package hello;
public class C extends D {}
EOF
  cat > "$pkg/java/hello/D.java" <<'EOF'
package hello;
public class D {}
EOF
  cat > "$pkg/java/hello/BUILD" <<'EOF'
load("@rules_java//java:java_library.bzl", "java_library")
java_library(name='a', srcs=['A.java'], deps = [':b'])
java_library(name='b', srcs=['B.java'], deps = [':c'])
java_library(name='c', srcs=['C.java'], deps = [':d'])
java_library(name='d', srcs=['D.java'])
EOF

  bazel build --experimental_output_paths=strip \
    //"$pkg"/java/hello:a -s 2>"$TEST_log" \
    || fail "Expected success"

  # java_library .jar compilation:
  assert_paths_stripped "$TEST_log" "$pkg/java/hello/liba.jar-0.params"
  # java_library header jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/java/hello/libb-hjar.jar"
  # jdeps files should contain the original paths since they are read by downstream actions that may
  # not use path mapping.
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/liba.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libb.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libb-hjar.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libc.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libc-hjar.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libd.jdeps"
  assert_contains_no_stripped_path "${bazel_bin}/$pkg/java/hello/libd-hjar.jdeps"
}

function test_builtin_cc_support() {
  add_rules_cc MODULE.bazel
  local -r pkg="third_party/${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/BUILD" <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "//$pkg/lib1",
        "//$pkg/lib2",
    ],
)
EOF
  cat > "$pkg/main.cc" <<EOF
#include <iostream>
#include "$pkg/lib1/lib1.h"
#include "lib2.h"

int main() {
  std::cout << GetLib1Greeting() << std::endl;
  std::cout << GetLib2Greeting() << std::endl;
  return 0;
}
EOF

  mkdir -p "$pkg"/lib1
  cat > "$pkg/lib1/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "lib1",
    srcs = ["lib1.cc"],
    hdrs = ["lib1.h"],
    deps = ["//$pkg/common/utils:utils"],
    visibility = ["//visibility:public"],
)
EOF
  cat > "$pkg/lib1/lib1.h" <<EOF
#ifndef LIB1_H_
#define LIB1_H_

#include <string>

std::string GetLib1Greeting();

#endif
EOF
  cat > "$pkg/lib1/lib1.cc" <<EOF
#include "lib1.h"
#include "other_dir/utils.h"

std::string GetLib1Greeting() {
  return AsGreeting("lib1");
}
EOF

  mkdir -p "$pkg"/lib2
  cat > "$pkg/lib2/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
genrule(
    name = "gen_header",
    srcs = ["lib2.h.tpl"],
    outs = ["lib2.h"],
    cmd = "cp \$< \$@",
)
genrule(
    name = "gen_source",
    srcs = ["lib2.cc.tpl"],
    outs = ["lib2.cc"],
    cmd = "cp \$< \$@",
)
cc_library(
    name = "lib2",
    srcs = ["lib2.cc"],
    hdrs = ["lib2.h"],
    includes = ["."],
    deps = ["//$pkg/common/utils:utils"],
    visibility = ["//visibility:public"],
)
EOF
  cat > "$pkg/lib2/lib2.h.tpl" <<EOF
#ifndef LIB2_H_
#define LIB2_H_

#include <string>

std::string GetLib2Greeting();

#endif
EOF
  cat > "$pkg/lib2/lib2.cc.tpl" <<EOF
#include "lib2.h"
#include "other_dir/utils.h"

std::string GetLib2Greeting() {
  return AsGreeting("lib2");
}
EOF

  mkdir -p "$pkg"/common/utils
  cat > "$pkg/common/utils/BUILD" <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
genrule(
    name = "gen_header",
    srcs = ["utils.h.tpl"],
    outs = ["dir/utils.h"],
    cmd = "cp \$< \$@",
)
genrule(
    name = "gen_source",
    srcs = ["utils.cc.tpl"],
    outs = ["dir/utils.cc"],
    cmd = "cp \$< \$@",
)
cc_library(
    name = "utils",
    srcs = ["dir/utils.cc"],
    hdrs = ["dir/utils.h"],
    include_prefix = "other_dir",
    strip_include_prefix = "dir",
    visibility = ["//visibility:public"],
)
EOF
  cat > "$pkg/common/utils/utils.h.tpl" <<EOF
#ifndef SOME_PKG_UTILS_H_
#define SOME_PKG_UTILS_H_

#include <string>

std::string AsGreeting(const std::string& name);
#endif
EOF
  cat > "$pkg/common/utils/utils.cc.tpl" <<EOF
#include "utils.h"

std::string AsGreeting(const std::string& name) {
  return "Hello, " + name + "!";
}
EOF

  # Verify the build succeeds
  bazel build -s \
    --modify_execution_info=CppCompile=+supports-path-mapping \
    "//$pkg:main" 2>"$TEST_log" || fail "Expected success"

  # Verify that all paths are stripped as expected.
  # The extension can be .pic.o or .o depending on the platform.
  assert_paths_stripped "$TEST_log" "$pkg/common/utils/_objs/utils/utils."
  assert_paths_stripped "$TEST_log" "$pkg/lib1/_objs/lib1/lib1."
  assert_paths_stripped "$TEST_log" "$pkg/lib2/_objs/lib2/lib2."
  assert_paths_stripped "$TEST_log" "$pkg/_objs/main/main."
}

##############################################################################
# Colliding-inputs tests.
#
# When path stripping maps inputs from different configurations onto the same
# root-relative path, the build must still succeed as long as the colliding
# files have identical content (verified by StrippingPathMapper.isPathStrippable).
# The tests below cover C++ headers, Java JARs (full and reduced classpath),
# and Starlark run_shell actions.
##############################################################################

# Test: C++ header collision.
# Aim: Verify that CppCompile actions use stripped paths when two generated
# headers from different configurations collide on the same root-relative path
# but have identical content.
# Case: A Starlark rule produces a header under a config transition.  Two
# instances with different setting values generate identical "shared.h" files.
# The cc_binary depending on both should compile with stripped paths.
function test_identical_colliding_inputs_are_stripped() {
  if ! is_bazel; then
    # TODO(bazel-team): figure out why Google C++ rules fails.
    return 0
  fi

  local rules_cc_prefix="@rules_cc//"
  if ! is_bazel; then
    rules_cc_prefix="@rules_cc//"
  fi
  add_rules_cc MODULE.bazel
  local -r pkg="third_party/${FUNCNAME[0]}"
  mkdir -p "$pkg/rules"
  cat > $pkg/rules/defs.bzl <<EOF
SettingInfo = provider(fields = ["value"])

def _setting_impl(ctx):
    return SettingInfo(value = ctx.build_setting_value)

my_setting = rule(
    implementation = _setting_impl,
    build_setting = config.string(),
)

def _transition_impl(settings, attr):
    return {"//$pkg/rules:my_setting": attr.setting_value}

_my_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//$pkg/rules:my_setting"],
)

def _gen_identical_header_impl(ctx):
    # Generate a header with content that does NOT depend on the config setting,
    # so that outputs from different configurations are identical.
    header = ctx.actions.declare_file("shared.h")
    ctx.actions.write(header, "#define SHARED_VALUE 42\\n")
    return [
        DefaultInfo(files = depset([header])),
    ]

gen_identical_header = rule(
    _gen_identical_header_impl,
    cfg = _my_transition,
    attrs = {
        "setting_value": attr.string(),
    },
)
EOF
  cat > $pkg/rules/BUILD << 'EOF'
load(":defs.bzl", "my_setting")

my_setting(
    name = "my_setting",
    build_setting_default = "",
)
EOF

  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
load("//$pkg/rules:defs.bzl", "gen_identical_header")
load("${rules_cc_prefix}cc:cc_binary.bzl", "cc_binary")
gen_identical_header(
    name = "header_a",
    setting_value = "a",
)
gen_identical_header(
    name = "header_b",
    setting_value = "b",
)
cc_binary(
    name = "main",
    srcs = [
        "main.cc",
        ":header_a",
        ":header_b",
    ],
    includes = ["."],
)
EOF
  cat > $pkg/main.cc <<'EOF'
#include "shared.h"
int main() {
  return SHARED_VALUE - 42;
}
EOF

  bazel clean
  bazel build -s \
    --modify_execution_info=CppCompile=+supports-path-mapping \
    "//$pkg:main" 2>"$TEST_log" \
    || fail "Expected success"

  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"

  # The two shared.h files from different configs have identical content,
  # so path mapping should be used (paths stripped) for the CppCompile action.
  # The extension can be .pic.o or .o depending on the platform.
  assert_paths_stripped "$TEST_log" "$pkg/_objs/main/main."

}

# Test: Java JAR collision (full classpath / getFullSpawn).
# Aim: Verify that JavaCompileAction uses stripped paths when two JARs from
# different configurations collide on the same root-relative path but have
# identical content.
# Case: A java_library is built under two different config transitions,
# producing identical JARs.  A Starlark rule forwards both as JavaInfo deps
# to a downstream java_binary, whose compilation should use stripped paths.
# This exercises JavaCompileAction.getFullSpawn().
function test_identical_colliding_java_inputs_are_stripped() {
  local rules_java_prefix="@rules_java//"
  if ! is_bazel; then
    rules_java_prefix="@rules_java//"
  fi
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/rules"
  cat > $pkg/rules/defs.bzl <<EOF
load("${rules_java_prefix}java/common:java_info.bzl", "JavaInfo")

SettingInfo = provider(fields = ["value"])

def _setting_impl(ctx):
    return SettingInfo(value = ctx.build_setting_value)

my_setting = rule(
    implementation = _setting_impl,
    build_setting = config.string(),
)

def _transition_impl(settings, attr):
    return {"//$pkg/rules:my_setting": attr.setting_value}

_my_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//$pkg/rules:my_setting"],
)

def _transitioned_java_dep_impl(ctx):
    # Forward the JavaInfo from a java_library built under a config transition.
    # Since the java_library doesn't depend on the custom setting, the JARs
    # are identical across configurations.
    return [ctx.attr.dep[JavaInfo]]

transitioned_java_dep = rule(
    _transitioned_java_dep_impl,
    cfg = _my_transition,
    attrs = {
        "dep": attr.label(providers = [JavaInfo]),
        "setting_value": attr.string(),
    },
    provides = [JavaInfo],
)
EOF
  cat > $pkg/rules/BUILD << 'EOF'
load(":defs.bzl", "my_setting")

my_setting(
    name = "my_setting",
    build_setting_default = "",
)
EOF

  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
load("${rules_java_prefix}java:java_binary.bzl", "java_binary")
load("${rules_java_prefix}java:java_library.bzl", "java_library")
load("//$pkg/rules:defs.bzl", "transitioned_java_dep")
java_library(
    name = "mylib",
    srcs = ["MyLib.java"],
)
transitioned_java_dep(
    name = "dep_a",
    dep = ":mylib",
    setting_value = "a",
)
transitioned_java_dep(
    name = "dep_b",
    dep = ":mylib",
    setting_value = "b",
)
java_binary(
    name = "mybin",
    srcs = ["MyBin.java"],
    main_class = "main.MyBin",
    deps = [
        ":dep_a",
        ":dep_b",
    ],
)
EOF
  cat > $pkg/MyLib.java <<'EOF'
package mylib;
public class MyLib {
  public static void runMyLib() {
    System.out.println("MyLib checking in.");
  }
}
EOF
  cat > $pkg/MyBin.java <<'EOF'
package main;
import mylib.MyLib;
public class MyBin {
  public static void main(String[] argv) {
    MyLib.runMyLib();
  }
}
EOF

  bazel clean
  bazel build --experimental_output_paths=strip \
    "//$pkg:mybin" -s 2>"$TEST_log" \
    || fail "Expected success"

  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"

  # The two libmylib.jar files from different configs have identical content,
  # so path mapping should be used (paths stripped) for the java_binary
  # compilation.
  assert_paths_stripped "$TEST_log" "bin/$pkg/mybin.jar"

  # Verify that a .params file was used (ParameterFileWriteAction) and that
  # it contains stripped paths for the classpath JAR.
  local params_file="${bazel_out:0:5}-bin/$pkg/mybin.jar-0.params"
  [[ -f "$params_file" ]] \
    || fail "Expected params file to exist: $params_file"
  grep -q "${bazel_out}/cfg/bin/$pkg/libmylib" "$params_file" \
    || fail "Expected stripped libmylib path in params file: $(cat $params_file)"
}

# Test: Java JAR collision (reduced classpath / getReducedSpawn).
# Aim: Same as test_identical_colliding_java_inputs_are_stripped, but with
# classpath reduction enabled (--experimental_java_classpath=bazel).
# Case: Identical setup to the previous test.  The difference is that this
# exercises JavaCompileAction.getReducedSpawn() instead of getFullSpawn(),
# ensuring stripped paths are also applied in the reduced-classpath code path.
function test_identical_colliding_java_inputs_are_stripped_reduced_classpath() {
  local rules_java_prefix="@rules_java//"
  if ! is_bazel; then
    rules_java_prefix="@rules_java//"
  fi
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/rules"
  cat > $pkg/rules/defs.bzl <<EOF
load("${rules_java_prefix}java/common:java_info.bzl", "JavaInfo")

SettingInfo = provider(fields = ["value"])

def _setting_impl(ctx):
    return SettingInfo(value = ctx.build_setting_value)

my_setting = rule(
    implementation = _setting_impl,
    build_setting = config.string(),
)

def _transition_impl(settings, attr):
    return {"//$pkg/rules:my_setting": attr.setting_value}

_my_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//$pkg/rules:my_setting"],
)

def _transitioned_java_dep_impl(ctx):
    return [ctx.attr.dep[JavaInfo]]

transitioned_java_dep = rule(
    _transitioned_java_dep_impl,
    cfg = _my_transition,
    attrs = {
        "dep": attr.label(providers = [JavaInfo]),
        "setting_value": attr.string(),
    },
    provides = [JavaInfo],
)
EOF
  cat > $pkg/rules/BUILD << 'EOF'
load(":defs.bzl", "my_setting")

my_setting(
    name = "my_setting",
    build_setting_default = "",
)
EOF

  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
load("${rules_java_prefix}java:java_binary.bzl", "java_binary")
load("${rules_java_prefix}java:java_library.bzl", "java_library")
load("//$pkg/rules:defs.bzl", "transitioned_java_dep")
java_library(
    name = "mylib",
    srcs = ["MyLib.java"],
)
transitioned_java_dep(
    name = "dep_a",
    dep = ":mylib",
    setting_value = "a",
)
transitioned_java_dep(
    name = "dep_b",
    dep = ":mylib",
    setting_value = "b",
)
java_binary(
    name = "mybin",
    srcs = ["MyBin.java"],
    main_class = "main.MyBin",
    deps = [
        ":dep_a",
        ":dep_b",
    ],
)
EOF
  cat > $pkg/MyLib.java <<'EOF'
package mylib;
public class MyLib {
  public static void runMyLib() {
    System.out.println("MyLib checking in.");
  }
}
EOF
  cat > $pkg/MyBin.java <<'EOF'
package main;
import mylib.MyLib;
public class MyBin {
  public static void main(String[] argv) {
    MyLib.runMyLib();
  }
}
EOF

  bazel clean
  bazel build --experimental_output_paths=strip \
    --experimental_java_classpath=bazel \
    "//$pkg:mybin" -s 2>"$TEST_log" \
    || fail "Expected success"

  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"

  # With classpath reduction, JavaCompileAction.getReducedSpawn() is used.
  # The two libmylib.jar files from different configs have identical content,
  # so path mapping should be used (paths stripped).
  assert_paths_stripped "$TEST_log" "bin/$pkg/mybin.jar"

  # Verify that a .params file was used (ParameterFileWriteAction) and that
  # it contains stripped paths for the classpath JAR.
  local params_file="${bazel_out:0:5}-bin/$pkg/mybin.jar-0.params"
  [[ -f "$params_file" ]] \
    || fail "Expected params file to exist: $params_file"
  grep -q "${bazel_out}/cfg/bin/$pkg/libmylib" "$params_file" \
    || fail "Expected stripped libmylib path in params file: $(cat $params_file)"
}

# Test: Starlark run_shell action collision.
# Aim: Verify that SpawnAction uses stripped paths when two generated files
# from different configurations collide on the same root-relative path but
# have identical content.
# Case: A Starlark rule generates an identical "data.txt" file under two
# different config transitions.  A downstream Starlark run_shell action
# concatenates both files.  This exercises SpawnAction.getSpawn() and checks
# that the actual shell command line uses /cfg/ stripped paths.
function test_identical_colliding_starlark_action_inputs_are_stripped() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/rules"
  cat > $pkg/rules/defs.bzl <<EOF
SettingInfo = provider(fields = ["value"])

def _setting_impl(ctx):
    return SettingInfo(value = ctx.build_setting_value)

my_setting = rule(
    implementation = _setting_impl,
    build_setting = config.string(),
)

def _transition_impl(settings, attr):
    return {"//$pkg/rules:my_setting": attr.setting_value}

_my_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = ["//$pkg/rules:my_setting"],
)

def _gen_identical_file_impl(ctx):
    f = ctx.actions.declare_file("data.txt")
    ctx.actions.write(f, "identical content\\n")
    return [DefaultInfo(files = depset([f]))]

gen_identical_file = rule(
    _gen_identical_file_impl,
    cfg = _my_transition,
    attrs = {
        "setting_value": attr.string(),
    },
)

def _cat_impl(ctx):
    inputs = ctx.files.srcs
    output = ctx.actions.declare_file(ctx.attr.name + ".out")
    out_args = ctx.actions.args()
    out_args.add(output)
    in_args = ctx.actions.args()
    in_args.add_all(inputs)
    ctx.actions.run_shell(
        inputs = inputs,
        outputs = [output],
        arguments = [out_args, in_args],
        command = "out=\$1; shift; cat \$@ > \$out",
        execution_requirements = {"supports-path-mapping": "1"},
    )
    return [DefaultInfo(files = depset([output]))]

cat_rule = rule(
    _cat_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
)
EOF
  cat > $pkg/rules/BUILD << 'EOF'
load(":defs.bzl", "my_setting")

my_setting(
    name = "my_setting",
    build_setting_default = "",
)
EOF

  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
load("//$pkg/rules:defs.bzl", "gen_identical_file", "cat_rule")
gen_identical_file(
    name = "file_a",
    setting_value = "a",
)
gen_identical_file(
    name = "file_b",
    setting_value = "b",
)
cat_rule(
    name = "combined",
    srcs = [
        ":file_a",
        ":file_b",
    ],
)
EOF

  bazel clean
  bazel build --experimental_output_paths=strip \
    "//$pkg:combined" -s 2>"$TEST_log" \
    || fail "Expected success"

  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"

  # The two data.txt files from different configs have identical content,
  # so path mapping should be used (paths stripped) for the Starlark
  # run_shell action (SpawnAction.getSpawn).
  # Note: assert_paths_stripped uses xargs which chokes on single quotes
  # in run_shell commands, so we check directly.
  # Grep for the actual bash command line (not the SUBCOMMAND header).
  local action_cmd
  action_cmd=$(grep "bash.*combined.out" "$TEST_log" || true)
  [[ -n "$action_cmd" ]] || fail "No bash command found for combined.out in $TEST_log"
  echo "$action_cmd" | grep -q "${bazel_out}/cfg/bin/$pkg/combined.out" \
    || fail "Expected combined.out with /cfg/ path in action: $action_cmd"
  echo "$action_cmd" | grep -q "${bazel_out}/cfg/bin/$pkg/data.txt" \
    || fail "Expected data.txt with /cfg/ path in action: $action_cmd"
  # Verify no non-stripped config paths appear for our package's artifacts.
  echo "$action_cmd" | grep -oE "${bazel_out}/[^ ')]+" | grep "$pkg" | while read -r path; do
    echo "$path" | grep -q "/cfg/" \
      || fail "Found non-stripped path for $pkg artifact: $path"
  done
}

run_suite "Tests stripping config prefixes from output paths for better action caching"
