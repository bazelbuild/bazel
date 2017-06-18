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
# Correctness tests for using a Persistent TestRunner.
#

if is_windows; then
  echo "Persistent test runner functionality not ready for windows" >&2
  exit 0
fi

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_simple_scenario() {
  setup_javatest_support
  mkdir -p java/testrunners || fail "mkdir failed"

  cat > java/testrunners/TestsPass.java <<EOF
package testrunners;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class TestsPass {

  @Test
  public void testPass() {
    // This passes
  }
}
EOF

  cat > java/testrunners/TestsFail.java <<EOF
package testrunners;
import static org.junit.Assert.fail;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class TestsFail {

  @Test
  public void testFail() {
    fail("Test is supposed to fail");
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_test(name = "TestsPass",
          srcs = ['TestsPass.java'],
          deps = ['//third_party:junit4'],
)

java_test(name = "TestsFail",
          srcs = ['TestsFail.java'],
          deps = ['//third_party:junit4'],
)
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      //java/testrunners:TestsPass || fail "Test fails unexpectedly"

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      --test_output=all //java/testrunners:TestsFail &> $TEST_log \
      && fail "Test passes unexpectedly" || true
  expect_log "Test is supposed to fail"
}

#TODO(b/37304748): Re-enable once we fix its flakiness.
function DISABLED_test_reload_modified_classes() {
  setup_javatest_support
  mkdir -p java/testrunners || fail "mkdir failed"

  # Create a passing test.
  cat > java/testrunners/Tests.java <<EOF
package testrunners;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testPass() {
    // This passes
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = ['//third_party:junit4'],
)
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      //java/testrunners:Tests &> $TEST_log || fail "Test fails unexpectedly"

  # Now get the test to fail.
  cat > java/testrunners/Tests.java <<EOF
package testrunners;
import static org.junit.Assert.fail;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testPass() {
    fail("Test is supposed to fail now");
  }
}
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      --test_output=all --nocache_test_results //java/testrunners:Tests &> $TEST_log \
      && fail "Test passes unexpectedly" || true
  expect_log "Test is supposed to fail now"
}

function test_reload_modified_classpaths() {
  setup_javatest_support
  mkdir -p java/testrunners || fail "mkdir failed"

  # Create a passing test.
  cat > java/testrunners/Tests.java <<EOF
package testrunners;
import static org.junit.Assert.fail;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testPass() {
    // This passes
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = ['//third_party:junit4'],
)
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      //java/testrunners:Tests &> $TEST_log || fail "Test fails unexpectedly"

  # Create a library to add a dep.
  cat > java/testrunners/TrueVal.java <<EOF
package testrunners;

public class TrueVal {
  public static final boolean VAL = true;
}
EOF

  # Now get the test to fail depending on the library
  cat > java/testrunners/Tests.java <<EOF
package testrunners;
import static org.junit.Assert.fail;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class Tests {

  @Test
  public void testFail() {
    if (TrueVal.VAL) {
      fail("Supposed to fail now.");
    }
  }
}
EOF

  # Add an additional library to the classpath.
  cat > java/testrunners/BUILD <<EOF
java_library(name = "trueval",
             srcs = ["TrueVal.java"],
)

java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = [
                   ':trueval',
                   '//third_party:junit4'
                 ],
)
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --test_strategy=experimental_worker \
      --test_output=all --nocache_test_results //java/testrunners:Tests &> $TEST_log \
       && fail "Test passes unexpectedly" || true
  expect_log "Supposed to fail now."
}

function test_fail_without_testrunner() {
  mkdir -p java/testrunners || fail "mkdir failed"

  cat > java/testrunners/TestWithoutRunner.java <<EOF
package testrunners;
public class TestWithoutRunner {
  public static void main(String[] args) {
    // Empty main. Silently pass.
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_test(name = "TestWithoutRunner",
          srcs = ['TestWithoutRunner.java'],
          use_testrunner = 0,
          main_class = "testrunners.TestWithoutRunner"
)
EOF

  bazel test --explicit_java_test_deps --experimental_testrunner --nocache_test_results \
      //java/testrunners:TestWithoutRunner >& $TEST_log || fail "Normal test execution should pass."

  bazel test --explicit_java_test_deps --experimental_testrunner --nocache_test_results \
      --test_strategy=experimental_worker >& $TEST_log //java/testrunners:TestWithoutRunner \
      && fail "Test should have failed when running with an experimental runner." || true

  expect_log \
      "Tests that do not use the experimental test runner are incompatible with the persistent worker"
}

function test_fail_without_experimental_testrunner() {
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
  public void testPass() {
    // This passes
  }
}
EOF

  cat > java/testrunners/BUILD <<EOF
java_test(name = "Tests",
          srcs = ['Tests.java'],
          deps = ['//third_party:junit4'],
)
EOF

  bazel test --nocache_test_results //java/testrunners:Tests >& $TEST_log \
      || fail "Normal test execution should pass."

  bazel test --nocache_test_results --test_strategy=experimental_worker >& $TEST_log \
      //java/testrunners:Tests \
      && fail "Test should have failed when running with an experimental runner." \
      || true

  expect_log "Build configuration not compatible with experimental_worker"
}

run_suite "Persistent Test Runner tests"
