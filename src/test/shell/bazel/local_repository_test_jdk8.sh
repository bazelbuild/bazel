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
# Test parts of the local_repository binding which are broken with jdk7
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# Creates an indirect dependency on X from A and make sure the error message
# refers to the correct label, both in an external repository and not.
function test_indirect_dep_message() {
  local external_dir=$TEST_TMPDIR/ext-dir
  mkdir -p a b $external_dir/x
  cat > a/A.java <<EOF
package a;

import x.X;

public class A {
  public static void main(String args[]) {
    X.print();
  }
}
EOF
  cat > a/BUILD <<EOF
java_binary(
    name = "a",
    main_class = "a.A",
    srcs = ["A.java"],
    deps = ["//b"],
)
EOF


  cat > b/B.java <<EOF
package b;

public class B {
  public static void print() {
     System.out.println("B");
  }
}
EOF
  cat > b/BUILD <<EOF
java_library(
    name = "b",
    srcs = ["B.java"],
    deps = ["@x_repo//x"],
    visibility = ["//visibility:public"],
)
EOF

  cp -r a b $external_dir

  touch $external_dir/WORKSPACE
  cat > $external_dir/x/X.java <<EOF
package x;

public class X {
  public static void print() {
    System.out.println("X");
  }
}
EOF
  cat > $external_dir/x/BUILD <<EOF
java_library(
    name = "x",
    srcs = ["X.java"],
    visibility = ["//visibility:public"],
)
EOF

  cat > WORKSPACE <<EOF
local_repository(
    name = "x_repo",
    path = "$external_dir",
)
EOF

  bazel build @x_repo//a >& $TEST_log && fail "Building @x_repo//a should error out"
  expect_log "** Please add the following dependencies:"
  expect_log "@x_repo//x  to @x_repo//a"
}

run_suite "local repository tests for jdk8 only"
