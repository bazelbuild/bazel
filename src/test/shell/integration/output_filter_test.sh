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
# output_filter_test.sh: a couple of end to end tests for the warning
# filter functionality.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_output_filter_cc() {
  # "test warning filter for C compilation"

  mkdir -p cc/main
  cat > cc/main/BUILD <<EOF
cc_library(name='cc',
           srcs=['main.c'],
           nocopts='-Werror')
EOF

  cat >cc/main/main.c <<EOF
#include <stdio.h>

int main(void)
{
#warning("Through me you pass into the city of woe:")
#warning("Through me you pass into eternal pain:")
#warning("Through me among the people lost for aye.")
  printf("%s", "Hello, World!\n");
}
EOF

  bazel clean
  bazel build cc/main:cc 2> stderr.txt
  cat stderr.txt
  grep "into eternal pain" stderr.txt || \
      fail "No warning from C compilation"

  bazel clean
  bazel build --output_filter="dummy" cc/main:cc &> stderr.txt
  grep "into eternal pain" stderr.txt && \
      fail "Warning given by C compilation although they are disabled"

  true
}

function test_output_filter_java() {
  # "test warning filter for Java compilation"

  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(name = 'main',
    deps = ['//java/hello_library'],
    srcs = ['Main.java'],
    javacopts = ['-Xlint:deprecation'],
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
             srcs = ['HelloLibrary.java'],
             javacopts = ['-Xlint:deprecation']);
EOF

  cat >java/hello_library/HelloLibrary.java <<EOF
package hello_library;
public class HelloLibrary {
  /** @deprecated */
  @Deprecated
  public static void funcHelloLibrary() {
    System.out.print("Hello, Library!;");
  }
}
EOF

  bazel clean
  # check that we do get a deprecation warning
  bazel build //java/main:main 2>stderr.txt || fail "build failed"
  grep -q "has been deprecated" stderr.txt || fail "no deprecation warning"
  # check that we do get a deprecation warning if we select the target
  bazel clean
  bazel build --output_filter=java/main //java/main:main 2>stderr.txt || fail "build failed"
  grep -q "has been deprecated" stderr.txt || fail "no deprecation warning"

  # check that we do not get a deprecation warning if we select another target
  bazel clean
  bazel build --output_filter=java/hello_library //java/main:main 2>stderr.txt || fail "build failed"
  grep -q "has been deprecated" stderr.txt && fail "deprecation warning"

  true
}

function test_test_output_printed() {
  # "test that test output is printed if warnings are disabled"

  mkdir -p foo/bar
  cat >foo/bar/BUILD <<EOF
sh_test(name='test',
        srcs=['test.sh'])
EOF

  cat >foo/bar/test.sh <<EOF
#!/bin/bash
exit 0
EOF

  chmod +x foo/bar/test.sh

  bazel test --output_filter="dummy" foo/bar:test 2> stderr.txt
  grep "PASS: //foo/bar:test" stderr.txt || fail "no PASSED message"
}

function test_output_filter_build() {
  # "test output filter for BUILD files"

  mkdir -p foo/bar
  cat >foo/bar/BUILD <<EOF
# Trigger sh_binary in deps of sh_binary warning.
sh_binary(name='red',
          srcs=['tomato.skin'])
sh_binary(name='tomato',
          srcs=['tomato.pulp'],
          deps=[':red'])
EOF

  touch foo/bar/tomato.{skin,pulp}
  chmod +x foo/bar/tomato.{skin,pulp}

  bazel clean
  # check that we do get a deprecation warning
  bazel build //foo/bar:tomato 2>stderr.txt || fail "build failed"
  grep -q "is unexpected here" stderr.txt \
    || fail "no warning"
  # check that we do get a deprecation warning if we select the target
  bazel clean
  bazel build --output_filter=foo/bar:tomato //foo/bar:tomato 2>stderr.txt \
    || fail "build failed"
  grep -q "is unexpected here" stderr.txt \
    || fail "no warning"

  # check that we do not get a deprecation warning if we select another target
  bazel clean
  bazel build --output_filter=foo/bar/:red //foo/bar:tomato 2>stderr.txt \
    || fail "build failed"
  grep -q "is unexpected here" stderr.txt \
    && fail "warning"

  true
}

function test_output_filter_build_hostattribute() {
  # "test that output filter also applies to host attributes"

  # What do you get in bars?
  mkdir -p bar

  cat >bar/BUILD <<EOF
# Trigger sh_binary in deps of sh_binary warning.
sh_binary(name='red',
          srcs=['tomato.skin'])
sh_binary(name='tomato',
          srcs=['tomato.pulp'],
          deps=[':red'])

# Booze, obviously.
genrule(name='bloody_mary',
        srcs=['vodka'],
        outs=['fun'],
        tools=[':tomato'],
        cmd='cp \$< \$@')
EOF

  touch bar/tomato.{skin,pulp}
  chmod +x bar/tomato.{skin,pulp}
  echo Moskowskaya > bar/vodka

  # Check that we do get a deprecation warning
  bazel clean
  bazel build //bar:bloody_mary 2>stderr1.txt || fail "build failed"
  grep -q "is unexpected here" stderr1.txt \
      || fail "no warning"

  # Check that the warning is disabled if we do not want to see it
  bazel clean
  bazel build //bar:bloody_mary --output_filter='nothing' 2>stderr2.txt \
    || fail "build failed"
  grep -q "is unexpected here" stderr2.txt \
      && fail "warning is not disabled"

  true
}

function test_output_filter_does_not_apply_to_test_output() {
  mkdir -p geflugel
  cat >geflugel/BUILD <<EOF
sh_test(name='mockingbird', srcs=['mockingbird.sh'])
sh_test(name='hummingbird', srcs=['hummingbird.sh'])
EOF

  cat >geflugel/mockingbird.sh <<EOF
#!/bin/bash
echo "To kill -9 a mockingbird"
exit 1
EOF

  cat >geflugel/hummingbird.sh <<EOF
#!/bin/bash
echo "To kill -9 a hummingbird"
exit 1
EOF

  chmod +x geflugel/*.sh

  bazel clean
  bazel test //geflugel:all --test_output=errors --output_filter=mocking &> $TEST_log \
    && fail "expected tests to fail"

  expect_log "To kill -9 a mockingbird"
  expect_log "To kill -9 a hummingbird"
}

# TODO(mstaib): enable test after deprecation warnings work in bazel
function disabled_test_filters_deprecated_targets() {
  init_test "test that deprecated target warnings are filtered"

  mkdir -p relativity ether
  cat > relativity/BUILD <<EOF
cc_binary(name = 'relativity', srcs = ['relativity.cc'], deps = ['//ether'])
EOF

  cat > ether/BUILD <<EOF
cc_library(name = 'ether', srcs = ['ether.cc'], deprecation = 'Disproven',
           visibility = ['//visibility:public'])
EOF

  bazel build --nobuild //relativity &> $TEST_log || fail "Expected success"
  expect_log_once "WARNING:.*target '//relativity:relativity' depends on \
deprecated target '//ether:ether': Disproven."

  bazel build --nobuild --output_filter="^//pizza" \
      //relativity &> $TEST_log || fail "Expected success"
  expect_not_log "WARNING:.*target '//relativity:relativity' depends on \
deprecated target '//ether:ether': Disproven."
}

run_suite "Warning Filter tests"
