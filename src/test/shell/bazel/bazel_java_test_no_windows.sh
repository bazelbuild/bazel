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
# Java integration tests that don't pass on Windows.

# Load the test setup defined in the parent directory

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
    || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_java_timeout_test() {
  setup_javatest_support
  mkdir -p javatests/com/google/timeout
  touch javatests/com/google/timeout/{BUILD,TimeoutTests.java}

  cat > javatests/com/google/timeout/TimeoutTests.java << EOF
package com.google.timeout;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.Test;

@RunWith(JUnit4.class)
public class TimeoutTests {

  @Test
  public void testPasses() throws InterruptedException { }

  @Test
  public void testTimesOut() throws InterruptedException {
    // sleep more than 1 min
    Thread.sleep(Long.MAX_VALUE);
  }
}
EOF

  cat > javatests/com/google/timeout/BUILD <<EOF
java_test(
  name = "TimeoutTests",
  srcs = ["TimeoutTests.java"],
  deps = ['//third_party:junit4'],
  timeout = "short", # 1 min
)
EOF
}

function test_java_test_timeout() {
  write_java_timeout_test
  bazel test javatests/com/google/timeout:TimeoutTests --test_timeout=5 >& "$TEST_log" \
      && fail "Unexpected success"
  xml_log=bazel-testlogs/javatests/com/google/timeout/TimeoutTests/test.xml
  [[ -s $xml_log ]] || fail "$xml_log was not present after test"
  cat "$xml_log" > "$TEST_log"
  expect_log "failures='2'"
  expect_log "<failure message='Test cancelled' type='java.lang.Exception'>java.lang.Exception: Test cancelled"
  expect_log "<failure message='Test interrupted' type='java.lang.Exception'>java.lang.Exception: Test interrupted"
}

function test_wrapper_resolves_runfiles_to_subsuming_tree() {
    setup_clean_workspace
    mkdir -p java/com/google/runfiles/
    cat <<'EOF' > java/com/google/runfiles/EchoRunfiles.java
package com.google.runfiles;

public class EchoRunfiles {
   public static void main(String[] argv) {
       System.out.println(System.getenv("JAVA_RUNFILES"));
   }
}
EOF
    cat <<'EOF' > java/com/google/runfiles/BUILD
java_binary(
    name = 'EchoRunfiles',
    srcs = ['EchoRunfiles.java'],
    visibility = ['//visibility:public'],
)
EOF
    # The workspace name is initialized in testenv.sh; use that var rather than
    # hardcoding it here. The extra sed pass is so we can selectively expand
    # that one var while keeping the rest of the heredoc literal.
    cat | sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" > check_runfiles.sh << 'EOF'
#!/bin/sh -eu
unset JAVA_RUNFILES # Force the wrapper script to recompute it.
subrunfiles=`$TEST_SRCDIR/{{WORKSPACE_NAME}}/java/com/google/runfiles/EchoRunfiles`
if [ $subrunfiles != $TEST_SRCDIR ]; then
  echo $subrunfiles
  echo "DOES NOT MATCH"
  echo $TEST_SRCDIR
  exit 1
fi
EOF
    chmod u+x check_runfiles.sh
    cat <<'EOF' > BUILD
sh_test(
    name = 'check_runfiles',
    srcs = ['check_runfiles.sh'],
    data = ['//java/com/google/runfiles:EchoRunfiles'],
)
EOF

    # Create a runfiles tree for EchoRunfiles.
    bazel build //java/com/google/runfiles:EchoRunfiles
    # We're testing a formerly non-hermetic interaction, so disable the sandbox.
    bazel test --spawn_strategy=standalone --test_output=errors :check_runfiles
}
