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
# Tests the examples provided in Bazel
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_hello_library_files() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(name = 'main',
    deps = ['//java/hello_library'],
    srcs = ['Main.java'],
    main_class = 'main.Main')
EOF

  cat >java/main/Main.java <<EOF
package main;
import hello_library.HelloLibrary;
public class Main {
  public static void main(String[] args) {
    HelloLibrary.funcHelloLibrary();
    System.out.println("Hello, World!");
  }
}
EOF

  mkdir -p java/hello_library
  cat >java/hello_library/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'hello_library',
             srcs = ['HelloLibrary.java']);
EOF

  cat >java/hello_library/HelloLibrary.java <<EOF
package hello_library;
public class HelloLibrary {
  public static void funcHelloLibrary() {
    System.out.print("Hello, Library!;");
  }
}
EOF
}

function write_files_for_java_provider_in_attr() {
  mkdir -p java/com/google/sandwich
  cd java/com/google/sandwich

  touch BUILD A.java B.java Main.java java_custom_library.bzl

  rule_type="$1" # java_library / java_import
  attribute_name="$2" # exports / runtime_deps
  srcs_attribute_row="srcs = ['A.java']"
  if [ "$rule_type" = "java_import" ]; then
    srcs_attribute_row="jars = []"
  fi

  cat > BUILD <<EOF
load(':java_custom_library.bzl', 'java_custom_library')

java_binary(
  name = "Main",
EOF

  if [ "$attribute_name" = "runtime_deps" ]; then
    cat >> BUILD <<EOF
  main_class = "com.google.sandwich.Main",
  runtime_deps = [":top"]
)

EOF
  else
    cat >> BUILD <<EOF
  srcs = ["Main.java"],
  deps = [":top"]
)

EOF
  fi

  echo "$rule_type(" >> BUILD

  cat >> BUILD <<EOF
  name = "top",
EOF

  echo "  $srcs_attribute_row," >> BUILD
  echo "  $attribute_name = [':middle']" >> BUILD

  cat >> BUILD <<EOF
)

java_custom_library(
  name = "middle",
EOF

  if [ "$attribute_name" = "runtime_deps" ]; then
    cat >> BUILD <<EOF
  srcs = ["B.java", "Main.java"],
)
EOF
  else
    cat >> BUILD <<EOF
  srcs = ["B.java"],
)
EOF
  fi

  cat > B.java <<EOF
package com.google.sandwich;
class B {
  public void printB() {
    System.out.println("Message from B");
  }
}
EOF

if [ "$rule_type" = "java_library" ]; then
  cat > A.java <<EOF
package com.google.sandwich;
class A {
  public void printA() {
    System.out.println("Message from A");
  }
}
EOF
fi

cat > Main.java <<EOF
package com.google.sandwich;
class Main {
  public static void main(String[] args) {
EOF

if [[ "$rule_type" = "java_library" && "$attribute_name" = "exports" ]]; then
  cat >> Main.java <<EOF
    A myObjectA = new A();
    myObjectA.printA();
EOF
fi

cat >> Main.java <<EOF
    B myObjectB = new B();
    myObjectB.printB();
  }
}
EOF
}

function write_java_custom_rule() {
  cat > java_custom_library.bzl << EOF
def _impl(ctx):
  deps = []
  for dep in ctx.attr.deps:
    if java_common.provider in dep:
      deps.append(dep[java_common.provider])
  deps_provider = java_common.merge(deps)

  output_jar = ctx.actions.declare_file("lib" + ctx.label.name + ".jar")

  compilation_provider = java_common.compile(
    ctx,
    source_files = ctx.files.srcs,
    output = output_jar,
    javac_opts = java_common.default_javac_opts(ctx, java_toolchain_attr = "_java_toolchain"),
    deps = deps,
    resources = ctx.files.resources,
    strict_deps = "ERROR",
    java_toolchain = ctx.attr._java_toolchain,
    host_javabase = ctx.attr._host_javabase
  )
  result = java_common.merge([deps_provider, compilation_provider])
  return struct(
    files = set([output_jar]),
    providers = [result]
  )

java_custom_library = rule(
  implementation = _impl,
  attrs = {
    "srcs": attr.label_list(allow_files=True),
    "deps": attr.label_list(),
    "resources": attr.label_list(allow_files=True),
    "_java_toolchain": attr.label(default = Label("@bazel_tools//tools/jdk:toolchain")),
    "_host_javabase": attr.label(default = Label("//tools/defaults:jdk"))
  },
  fragments = ["java"]
)
EOF
}

function test_build_hello_world() {
  write_hello_library_files

  bazel build //java/main:main &> $TEST_log || fail "build failed"
}

function test_build_with_sourcepath() {
  mkdir -p g
  cat >g/A.java <<'EOF'
package g;
public class A {
   public A() {
      new B();
   }
}
EOF

  cat >g/B.java <<'EOF'
package g;
public class B {
   public B() {
   }
}
EOF

  cat >g/BUILD <<'EOF'
genrule(
  name = "stub",
  srcs = ["B.java"],
  outs = ["B.jar"],
  cmd = "zip $@ $(SRCS)",
)
java_library(
  name = "test",
  srcs = ["A.java"],
  javacopts = ["-sourcepath $(GENDIR)/$(location :stub)", "-implicit:none"],
  deps = [":stub"]
)
EOF
  bazel build //g:test >$TEST_log || fail "Failed to build //g:test"
}

 function test_java_common_compile_sourcepath() {
   # TODO(bazel-team): Enable this for Java 7 when VanillaJavaBuilder supports --sourcepath.
   JAVA_VERSION="1.$(bazel query  --output=build '@bazel_tools//tools/jdk:toolchain' | grep source_version | cut -d '"' -f 2)"
   if [ "${JAVA_VERSION}" = "1.7" ]; then
     return 0
   fi
   mkdir -p g
   cat >g/A.java <<'EOF'
package g;
public class A {
   public A() {
      new B();
   }
}
EOF

  cat >g/B.java <<'EOF'
package g;
public class B {
   public B() {
   }
}
EOF

   cat >g/BUILD <<'EOF'
load(':java_custom_library.bzl', 'java_custom_library')
genrule(
  name = "stub",
  srcs = ["B.java"],
  outs = ["B.jar"],
  cmd = "zip $@ $(SRCS)",
)

java_custom_library(
  name = "test",
  srcs = ["A.java"],
  sourcepath = [":stub"]
)
EOF

  cat >g/java_custom_library.bzl <<'EOF'
def _impl(ctx):
  output_jar = ctx.actions.declare_file("lib" + ctx.label.name + ".jar")

  compilation_provider = java_common.compile(
    ctx,
    source_files = ctx.files.srcs,
    output = output_jar,
    javac_opts = java_common.default_javac_opts(ctx, java_toolchain_attr = "_java_toolchain"),
    deps = [],
    sourcepath = ctx.files.sourcepath,
    strict_deps = "ERROR",
    java_toolchain = ctx.attr._java_toolchain,
    host_javabase = ctx.attr._host_javabase
  )
  return struct(
    files = set([output_jar]),
    providers = [compilation_provider]
  )

java_custom_library = rule(
  implementation = _impl,
  attrs = {
    "srcs": attr.label_list(allow_files=True),
    "sourcepath": attr.label_list(),
    "_java_toolchain": attr.label(default = Label("@bazel_tools//tools/jdk:toolchain")),
    "_host_javabase": attr.label(default = Label("//tools/defaults:jdk"))
  },
  fragments = ["java"]
)
EOF
   bazel build //g:test &> $TEST_log || fail "Failed to build //g:test"
   jar tf bazel-bin/g/libtest.jar >> $TEST_log || fail "Failed to jar tf bazel-bin/g/libtest.jar"
   expect_log "g/A.class"
   expect_not_log "g/B.class"
 }

function test_java_common_compile_sourcepath_with_implicit_class() {
   # TODO(bazel-team): Enable this for Java 7 when VanillaJavaBuilder supports --sourcepath.
   JAVA_VERSION="1.$(bazel query  --output=build '@bazel_tools//tools/jdk:toolchain' | grep source_version | cut -d '"' -f 2)"
   if [ "${JAVA_VERSION}" = "1.7" ]; then
     return 0
   fi
   mkdir -p g
   cat >g/A.java <<'EOF'
package g;
public class A {
   public A() {
      new B();
   }
}
EOF

  cat >g/B.java <<'EOF'
package g;
public class B {
   public B() {
   }
}
EOF

   cat >g/BUILD <<'EOF'
load(':java_custom_library.bzl', 'java_custom_library')
genrule(
  name = "stub",
  srcs = ["B.java"],
  outs = ["B.jar"],
  cmd = "zip $@ $(SRCS)",
)

java_custom_library(
  name = "test",
  srcs = ["A.java"],
  sourcepath = [":stub"]
)
EOF

  cat >g/java_custom_library.bzl <<'EOF'
def _impl(ctx):
  output_jar = ctx.actions.declare_file("lib" + ctx.label.name + ".jar")

  compilation_provider = java_common.compile(
    ctx,
    source_files = ctx.files.srcs,
    output = output_jar,
    javac_opts = java_common.default_javac_opts(ctx, java_toolchain_attr = "_java_toolchain") + ["-implicit:class"],
    deps = [],
    sourcepath = ctx.files.sourcepath,
    strict_deps = "ERROR",
    java_toolchain = ctx.attr._java_toolchain,
    host_javabase = ctx.attr._host_javabase
  )
  return struct(
    files = set([output_jar]),
    providers = [compilation_provider]
  )

java_custom_library = rule(
  implementation = _impl,
  attrs = {
    "srcs": attr.label_list(allow_files=True),
    "sourcepath": attr.label_list(),
    "_java_toolchain": attr.label(default = Label("@bazel_tools//tools/jdk:toolchain")),
    "_host_javabase": attr.label(default = Label("//tools/defaults:jdk"))
  },
  fragments = ["java"]
)
EOF
   bazel build //g:test &> $TEST_log || fail "Failed to build //g:test"
   jar tf bazel-bin/g/libtest.jar >> $TEST_log || fail "Failed to jar tf bazel-bin/g/libtest.jar"
   expect_log "g/A.class"
   expect_log "g/B.class"
 }

# Runfiles is disabled by default on Windows, but we can test it on Unix by
# adding flag --experimental_enable_runfiles=0
function test_build_and_run_hello_world_without_runfiles() {
  write_hello_library_files

  bazel run --experimental_enable_runfiles=0 //java/main:main &> $TEST_log || fail "build failed"
  expect_log "Hello, Library!;Hello, World!"
}

function test_errorprone_error_fails_build_by_default() {
  JAVA_VERSION="1.$(bazel query  --output=build '@bazel_tools//tools/jdk:toolchain' | grep source_version | cut -d '"' -f 2)"
  if [ "${JAVA_VERSION}" = "1.7" ]; then
    return 0
  fi

  write_hello_library_files
  # Trigger an error-prone error by comparing two arrays via #equals().
  cat >java/hello_library/HelloLibrary.java <<EOF
package hello_library;
public class HelloLibrary {
  public static boolean funcHelloLibrary() {
    int[] arr1 = {1, 2, 3};
    int[] arr2 = {1, 2, 3};
    return arr1.equals(arr2);
  }
}
EOF

  bazel build //java/main:main &> $TEST_log && fail "build should have failed" || true
  expect_log "error: \[ArrayEquals\] Reference equality used to compare arrays"
}

function test_extrachecks_off_disables_errorprone() {
  JAVA_VERSION="1.$(bazel query  --output=build '@bazel_tools//tools/jdk:toolchain' | grep source_version | cut -d '"' -f 2)"
  if [ "${JAVA_VERSION}" = "1.7" ]; then
    return 0
  fi

  write_hello_library_files
  # Trigger an error-prone error by comparing two arrays via #equals().
  cat >java/hello_library/HelloLibrary.java <<EOF
package hello_library;
public class HelloLibrary {
  public static boolean funcHelloLibrary() {
    int[] arr1 = {1, 2, 3};
    int[] arr2 = {1, 2, 3};
    return arr1.equals(arr2);
  }
}
EOF
  # Disable error-prone for this target, though.
  cat >java/hello_library/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'hello_library',
             srcs = ['HelloLibrary.java'],
             javacopts = ['-extra_checks:off'],);
EOF

  bazel build //java/main:main &> $TEST_log || fail "build failed"
  expect_not_log "error: \[ArrayEquals\] Reference equality used to compare arrays"
}

function test_java_test_main_class() {
  setup_javatest_support
  mkdir -p java/testrunners || fail "mkdir failed"
  cat > java/testrunners/TestRunner.java <<EOF
package testrunners;

import com.google.testing.junit.runner.BazelTestRunner;

public class TestRunner {
  public static void main(String[] argv) {
    System.out.println("Custom test runner was run");
    BazelTestRunner.main(argv);
  }
}
EOF

  cat > java/testrunners/Tests.java <<EOF
package testrunners;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testTest() {
    System.out.println("testTest was run");
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_library(name = "test_runner",
             srcs = ['TestRunner.java'],
             deps = ['@bazel_tools//tools/jdk:TestRunner_deploy.jar'],
)

java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = ['//third_party:junit4'],
          main_class = "testrunners.TestRunner",
          runtime_deps = [':test_runner']
)
EOF
  bazel test --test_output=streamed //java/testrunners:Tests &> "$TEST_log"
  expect_log "Custom test runner was run"
  expect_log "testTest was run"
}

function test_basic_java_sandwich() {
  mkdir -p java/com/google/sandwich
  cd java/com/google/sandwich

  touch BUILD A.java B.java C.java Main.java java_custom_library.bzl

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')

java_binary(
  name = "Main",
  srcs = ["Main.java"],
  deps = [":top"]
)

java_library(
  name = "top",
  srcs = ["A.java"],
  deps = [":middle"]
)

java_custom_library(
  name = "middle",
  srcs = ["B.java"],
  deps = [":bottom"]
)

java_library(
  name = "bottom",
  srcs = ["C.java"]
)
EOF

  cat > C.java << EOF
package com.google.sandwich;
class C {
  public void printC() {
    System.out.println("Message from C");
  }
}
EOF

  cat > B.java << EOF
package com.google.sandwich;
class B {
  C myObject;
  public void printB() {
    System.out.println("Message from B");
    myObject = new C();
    myObject.printC();
  }
}
EOF

  cat > A.java << EOF
package com.google.sandwich;
class A {
  B myObject;
  public void printA() {
    System.out.println("Message from A");
    myObject = new B();
    myObject.printB();
  }
}
EOF

  cat > Main.java << EOF
package com.google.sandwich;
class Main {
  public static void main(String[] args) {
    A myObject = new A();
    myObject.printA();
  }
}
EOF

  write_java_custom_rule

  bazel run :Main > $TEST_log || fail "Java sandwich build failed"
  expect_log "Message from A"
  expect_log "Message from B"
  expect_log "Message from C"
}

function test_java_library_exports_java_sandwich() {
  write_files_for_java_provider_in_attr "java_library" "exports"
  write_java_custom_rule

  bazel run :Main > $TEST_log || fail "Java sandwich build failed"
  expect_log "Message from A"
  expect_log "Message from B"
}

function test_java_library_runtime_deps_java_sandwich() {
  write_files_for_java_provider_in_attr "java_library" "runtime_deps"
  write_java_custom_rule

  bazel run :Main > $TEST_log || fail "Java sandwich build failed"
  expect_log "Message from B"
}

function test_java_import_exports_java_sandwich() {
  write_files_for_java_provider_in_attr "java_import" "exports"
  write_java_custom_rule

  bazel run :Main > $TEST_log || fail "Java sandwich build failed"
  expect_log "Message from B"
}

function test_java_import_runtime_deps_java_sandwich() {
  write_files_for_java_provider_in_attr "java_import" "runtime_deps"
  write_java_custom_rule

  bazel run :Main > $TEST_log || fail "Java sandwich build failed"
  expect_log "Message from B"
}

function test_java_binary_deps_java_sandwich() {
  mkdir -p java/com/google/sandwich
  cd java/com/google/sandwich

  touch BUILD A.java B.java Main.java java_custom_library.bzl

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')

java_binary(
  name = "Main",
  srcs = ["Main.java"],
  deps = [":custom"]
)

java_custom_library(
  name = "custom",
  srcs = ["A.java"],
  deps = [":bottom"]
)

java_library(
  name = "bottom",
  srcs = ["B.java"]
)
EOF

  cat > B.java << EOF
package com.google.sandwich;
class B {
  public void print() {
    System.out.println("Message from B");
  }
}
EOF

  cat > A.java << EOF
package com.google.sandwich;
class A {
  B myObject;
  public void print() {
    System.out.println("Message from A");
    myObject = new B();
    myObject.print();
  }
}
EOF

  cat > Main.java << EOF
package com.google.sandwich;
class Main {
  public static void main(String[] args) {
    A myObject = new A();
    myObject.print();
  }
}
EOF

  write_java_custom_rule

  bazel run :Main > "$TEST_log" || fail "Java sandwich build failed"
  expect_log "Message from A"
  expect_log "Message from B"
}

function test_java_binary_runtime_deps_java_sandwich() {
  mkdir -p java/com/google/sandwich
  cd java/com/google/sandwich

  touch BUILD A.java Main.java java_custom_library.bzl

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')

java_binary(
  name = "Main",
  main_class = "com.google.sandwich.Main",
  runtime_deps = [":custom"]
)

java_custom_library(
  name = "custom",
  srcs = ["Main.java"],
  deps = [":bottom"]
)

java_library(
  name = "bottom",
  srcs = ["A.java"]
)
EOF

  cat > A.java << EOF
package com.google.sandwich;
class A {
  public void print() {
    System.out.println("Message from A");
  }
}
EOF

  cat > Main.java << EOF
package com.google.sandwich;
class Main {
  public static void main(String[] args) {
    System.out.println("Message from Main");
    A myObject = new A();
    myObject.print();
  }
}
EOF

  write_java_custom_rule

  bazel run :Main > "$TEST_log" || fail "Java sandwich build failed"
  expect_log "Message from Main"
  expect_log "Message from A"
}

function test_java_test_java_sandwich() {
  setup_javatest_support
  mkdir -p java/com/google/sandwich
  cd java/com/google/sandwich

  touch BUILD A.java B.java MainTest.java java_custom_library.bzl

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')

java_test(
  name = "MainTest",
  size = "small",
  srcs = ["MainTest.java"],
  deps = [
      ":custom",
      "//third_party:junit4",
  ],
)

java_custom_library(
  name = "custom",
  srcs = ["A.java"],
  deps = [":bottom"]
)

java_library(
  name = "bottom",
  srcs = ["B.java"]
)
EOF

  cat > A.java << EOF
package com.google.sandwich;
class A {
  B myObj = new B();
  public boolean returnsTrue() {
    System.out.println("Message from A");
    return myObj.returnsTrue();
  }
}
EOF

  cat > B.java << EOF
package com.google.sandwich;
class B {
  public boolean returnsTrue() {
    System.out.println("Message from B");
    return true;
  }
}
EOF

  cat > MainTest.java << EOF
package com.google.sandwich;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MainTest {
  @Test
  public void testReturnsTrue() {
    A myObj = new A();
    assertTrue(myObj.returnsTrue());
    System.out.println("Test message");
  }
}
EOF

  write_java_custom_rule

  bazel test :MainTest --test_output=streamed > "$TEST_log" || fail "Java sandwich for java_test failed"
  expect_log "Message from A"
  expect_log "Message from B"
  expect_log "Test message"
}

function test_explicit_java_test_deps_flag() {
  setup_javatest_support
  mkdir -p java/testrunners || fail "mkdir failed"

  cat > java/testrunners/Tests.java <<EOF
package testrunners;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testTest() {
    System.out.println("testTest was run");
  }
}
EOF

  # With explicit_java_test_deps, we fail without explicitly specifying the JUnit deps.
  cat > java/testrunners/BUILD <<EOF
java_test(name = "Tests",
          srcs = ['Tests.java'],
)
EOF
  bazel test --test_output=streamed --explicit_java_test_deps //java/testrunners:Tests \
      &> "$TEST_log" && fail "Expected Failure" || true
  expect_log "cannot find symbol"

  # We start passing again with explicit_java_test_deps once we explicitly specify the deps.
  cat > java/testrunners/BUILD <<EOF
java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = ['//third_party:junit4'],
)
EOF
  bazel test --test_output=streamed --explicit_java_test_deps //java/testrunners:Tests \
      &> "$TEST_log" || fail "Expected success"
  expect_log "testTest was run"
}

function test_java_sandwich_resources_file() {
  mkdir -p java/com/google/sandwich
  workspace_dir="$PWD"
  cd java/com/google/sandwich

  touch BUILD A.java java_custom_library.bzl my_precious_resource.txt

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')
java_custom_library(
  name = "custom",
  srcs = ["A.java"],
  resources = ["my_precious_resource.txt"]
)
EOF
  cat > A.java << EOF
package com.google.sandwich;
class A { }
EOF
  write_java_custom_rule
  cd "${workspace_dir}"
  bazel build java/com/google/sandwich:custom > "$TEST_log" || fail "Java sandwich build failed"
  unzip -l bazel-bin/java/com/google/sandwich/libcustom.jar  > "$TEST_log"
  expect_log "my_precious_resource.txt"
}

function test_java_sandwich_resources_filegroup() {
  mkdir -p java/com/google/sandwich
  workspace_dir="$PWD"
  cd java/com/google/sandwich

  touch BUILD A.java java_custom_library.bzl my_precious_resource.txt my_other_precious_resource.txt

  cat > BUILD << EOF
load(':java_custom_library.bzl', 'java_custom_library')
filegroup(
  name = "resources_group",
  srcs = ["my_precious_resource.txt", "my_other_precious_resource.txt"]
)
java_custom_library(
  name = "custom",
  srcs = ["A.java"],
  resources = [":resources_group"]
)
EOF
  cat > A.java << EOF
package com.google.sandwich;
class A { }
EOF
  write_java_custom_rule
  cd "${workspace_dir}"
  bazel build java/com/google/sandwich:custom > "$TEST_log" || fail "Java sandwich build failed"
  unzip -l bazel-bin/java/com/google/sandwich/libcustom.jar  > "$TEST_log"
  expect_log "my_precious_resource.txt"
  expect_log "my_other_precious_resource.txt"
}

run_suite "Java integration tests"

