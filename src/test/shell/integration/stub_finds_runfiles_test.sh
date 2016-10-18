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
# Test that {java,py}_{binary,test} rules can find their own runfiles
# and assemble a classpath when invoked in a variety of ways.

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

test_strategy="standalone"
genrule_strategy="local"
if [ $# -ge 1 ]; then
  test_strategy=$1
  genrule_strategy=$1
  shift
fi

#### TESTS #############################################################

mkdir pkg pkg/java
cat > pkg/BUILD << 'EOF'
java_binary(name = "javabin",
            main_class = "ExitZero",
            srcs = [ "java/ExitZero.java", ])
java_test(name = "javatest",
          main_class = "ExitZero",
          use_testrunner = 0,
          srcs = [ "java/ExitZero.java", ])
py_binary(name = "pybin",
          srcs = [ "pybin.py", ])
py_test(name = "pytest",
        srcs = [ "pytest.py", ])
sh_binary(name = "sh_runs_javabin",
          srcs = [ "sh_runs_javabin.sh", ],
          data = [ ":javabin", ])
sh_test(name = "sh_runs_javatest",
        srcs = [ "sh_runs_javatest.sh", ],
        data = [ ":javatest", ])
sh_binary(name = "sh_runs_pybin",
          srcs = [ "sh_runs_pybin.sh", ],
          data = [ ":pybin", ])
sh_test(name = "sh_runs_pytest",
        srcs = [ "sh_runs_pytest.sh", ],
        data = [ ":pytest", ])
genrule(name = "genrule_runs_javabin",
        tools = [ ":javabin", ":sh_runs_javabin", ],
        outs = [ "dummy1", ],
        cmd = "$(location :javabin) && $(location :sh_runs_javabin) && >$@")
genrule(name = "genrule_runs_pybin",
        tools = [ ":pybin", ":sh_runs_pybin", ],
        outs = [ "dummy2", ],
        cmd = "$(location :pybin) && $(location :sh_runs_pybin) && >$@")
EOF
cat > pkg/java/ExitZero.java << 'EOF'
public class ExitZero {
  public static void main(String[] args) { }
}
EOF
touch pkg/pybin.py
touch pkg/pytest.py
cat > pkg/sh_runs_javabin.sh << 'EOF'
#!/bin/bash
exec $0.runfiles/*/pkg/javabin
EOF
cat > pkg/sh_runs_javatest.sh << 'EOF'
#!/bin/bash
exec $TEST_SRCDIR/*/pkg/javatest
EOF
cat > pkg/sh_runs_pybin.sh << 'EOF'
#!/bin/bash
exec $0.runfiles/*/pkg/pybin
EOF
cat > pkg/sh_runs_pytest.sh << 'EOF'
#!/bin/bash
exec $TEST_SRCDIR/*/pkg/pytest
EOF
chmod +x pkg/*.sh

# Drop environment variable (set by the test runner for this test)
# which will interfere with directly invoking the tests we have just built.
unset TEST_SRCDIR

bazel build //pkg:javabin
${PRODUCT_NAME}-bin/pkg/javabin
bazel run //pkg:javabin
${PRODUCT_NAME}-bin/pkg/javabin.runfiles/*/pkg/javabin

bazel build //pkg:javatest
${PRODUCT_NAME}-bin/pkg/javatest
bazel run //pkg:javatest
${PRODUCT_NAME}-bin/pkg/javatest.runfiles/*/pkg/javatest
bazel test --test_strategy="$test_strategy" //pkg:javatest

bazel build //pkg:pybin
${PRODUCT_NAME}-bin/pkg/pybin
bazel run //pkg:pybin
${PRODUCT_NAME}-bin/pkg/pybin.runfiles/*/pkg/pybin

bazel build //pkg:pytest
${PRODUCT_NAME}-bin/pkg/pytest
bazel run //pkg:pytest
${PRODUCT_NAME}-bin/pkg/pytest.runfiles/*/pkg/pytest
bazel test --test_strategy="$test_strategy" //pkg:pytest

bazel build //pkg:sh_runs_javabin
${PRODUCT_NAME}-bin/pkg/sh_runs_javabin
bazel run //pkg:sh_runs_javabin

bazel build //pkg:sh_runs_javatest
bazel test --test_strategy="$test_strategy" //pkg:sh_runs_javatest

bazel build //pkg:sh_runs_pybin
${PRODUCT_NAME}-bin/pkg/sh_runs_pybin
bazel run //pkg:sh_runs_pybin

bazel build //pkg:sh_runs_pytest
bazel test --test_strategy="$test_strategy" //pkg:sh_runs_pytest

bazel clean
bazel build --genrule_strategy="$genrule_strategy" //pkg:genrule_runs_pybin

bazel clean
bazel build --genrule_strategy="$genrule_strategy" //pkg:genrule_runs_javabin

echo PASS
