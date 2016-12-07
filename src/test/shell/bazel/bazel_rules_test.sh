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
# Test rules provided in Bazel not tested by examples
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_sh_test() {
  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
sh_test(
name = 'success_test',
srcs = [ 'success_test.sh' ],
)

sh_test(
name = 'fail_test',
srcs = [ 'fail_test.sh' ],
)

EOF

  cat > a/success_test.sh <<EOF
#!/bin/sh
echo success-marker
exit 0
EOF

  cat > a/fail_test.sh <<EOF
#!/bin/sh
echo failure-message
exit 1
EOF

  chmod +x a/*.sh

  assert_test_ok //a:success_test
  assert_test_fails //a:fail_test
  expect_log 'failure-message'
}

function test_extra_action() {
  mkdir -p mypkg
  # Make a program to run on each action that just prints the path to the extra
  # action file. This file is a proto, but I don't want to bother implementing
  # a program that parses the proto here.
  cat > mypkg/echoer.sh <<EOF
#!/bin/bash
if [[ ! -e \$0.runfiles/__main__/mypkg/runfile ]]; then
  echo "Runfile not found" >&2
  exit 1
fi
echo EXTRA ACTION FILE: \$1
EOF
  chmod +x mypkg/echoer.sh

  cat > mypkg/Hello.java <<EOF
public class Hello {
    public void sayHi() {
        System.out.println("hi");
    }
}
EOF

  touch mypkg/runfile

  cat > mypkg/BUILD <<EOF
package(default_visibility = ["//visibility:public"])

extra_action(
    name = "echo-filename",
    cmd = "\$(location :echoer) \$(EXTRA_ACTION_FILE)",
    tools = [":echoer"],
)

action_listener(
    name = "al",
    extra_actions = [":echo-filename"],
    mnemonics = ["Javac"],
)

sh_binary(
    name = "echoer",
    srcs = ["echoer.sh"],
    data = ["runfile"],
)

java_library(
    name = "hello",
    srcs = ["Hello.java"],
)
EOF

    bazel build --experimental_action_listener=//mypkg:al //mypkg:hello >& $TEST_log \
      || fail "Building with action listener failed"
    expect_log "EXTRA ACTION FILE"
}

function test_with_arguments() {
  mkdir -p mypkg
  cat > mypkg/BUILD <<EOF
sh_test(
    name = "expected_arg_test",
    srcs = ["check_expected_argument.sh"],
    args = ["expected_value"],
)

sh_test(
    name = "unexpected_arg_test",
    srcs = ["check_expected_argument.sh"],
    args = ["unexpected_value"],
)
EOF
  cat > mypkg/check_expected_argument.sh <<EOF
#!/bin/sh
[ "expected_value" = "\$1" ] || exit 1
EOF

  chmod +x mypkg/check_expected_argument.sh

  assert_test_ok //mypkg:expected_arg_test
  assert_test_fails //mypkg:unexpected_arg_test
}

function test_top_level_test() {
  cat > BUILD <<EOF
sh_test(
    name = "trivial_test",
    srcs = ["true.sh"],
)
EOF
  cat > true.sh <<EOF
#!/bin/sh
exit 0
EOF

  chmod +x true.sh

  assert_test_ok //:trivial_test
}

# Regression test for https://github.com/bazelbuild/bazel/issues/67
# C++ library depedending on C++ library fails to compile on Darwin
function test_cpp_libdeps() {
  mkdir -p pkg
  cat <<'EOF' >pkg/BUILD
cc_library(
  name = "a",
  srcs = ["a.cc"],
)

cc_library(
  name = "b",
  srcs = ["b.cc"],
  deps = [":a"],
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [":b"],
)
EOF

  cat <<'EOF' >pkg/a.cc
#include <string>

std::string get_hello(std::string world) {
  return "Hello, " + world + "!";
}
EOF

  cat <<'EOF' >pkg/b.cc
#include <string>
#include <iostream>

std::string get_hello(std::string);

void print_hello(std::string world) {
  std::cout << get_hello(world) << std::endl;
}
EOF

  cat <<'EOF' >pkg/main.cc
#include <string>
void print_hello(std::string);

int main() {
   print_hello(std::string("World"));
}
EOF

  bazel build //pkg:a >& $TEST_log \
    || fail "Failed to build //pkg:a"
  bazel build //pkg:b >& $TEST_log \
    || fail "Failed to build //pkg:b"
  bazel run //pkg:main >& $TEST_log \
    || fail "Failed to run //pkg:main"
  expect_log "Hello, World!"
  ./bazel-bin/pkg/main >& $TEST_log \
    || fail "Failed to run //pkg:main"
  expect_log "Hello, World!"
}


function test_genrule_default_env() {
  mkdir -p pkg
  cat <<'EOF' >pkg/BUILD
genrule(
  name = "test",
  outs = ["test.out"],
  cmd = "(echo \"PATH=$$PATH\"; echo \"TMPDIR=$$TMPDIR\") > $@",
)
EOF
  local old_path="${PATH}"
  local old_tmpdir="${TMPDIR-}"
  local new_tmpdir="$(mktemp -d "${TEST_TMPDIR}/newfancytmpdirXXXXXX")"
  [ -d "${new_tmpdir}" ] || \
    fail "Could not create new temporary directory ${new_tmpdir}"
  export PATH="$PATH_TO_BAZEL_WRAPPER:/bin:/usr/bin:/random/path"
  export TMPDIR="${new_tmpdir}"
  # batch mode to force reload of the environment
  bazel --batch build //pkg:test || fail "Failed to build //pkg:test"
  assert_contains "PATH=$PATH_TO_BAZEL_WRAPPER:/bin:/usr/bin:/random/path" \
    bazel-genfiles/pkg/test.out
  assert_contains "TMPDIR=.*newfancytmpdir" \
    bazel-genfiles/pkg/test.out
  if [ -n "${old_tmpdir}" ]
  then
    export TMPDIR="${old_tmpdir}"
  else
    unset TMPDIR
  fi
  export PATH="${old_path}"
}

function test_genrule_remote() {
  cat > WORKSPACE <<EOF
local_repository(
    name = "r",
    path = __workspace_dir__,
)
EOF
  mkdir package
  cat > package/BUILD <<EOF
genrule(
    name = "abs_dep",
    srcs = ["//package:in"],
    outs = ["abs_dep.out"],
    cmd = "echo '\$(locations //package:in)' > \$@",
)

sh_binary(
    name = "in",
    srcs = ["in.sh"],
)
EOF

  cat > package/in.sh << EOF
#!/bin/bash
echo "Hi"
EOF
  chmod +x package/in.sh

  bazel build @r//package:abs_dep >$TEST_log 2>&1 || fail "Should build"
}

function test_genrule_remote_d() {
  cat > WORKSPACE <<EOF
local_repository(
    name = "r",
    path = __workspace_dir__,
)
EOF
  mkdir package
  cat > package/BUILD <<'EOF'
genrule(
    name = "hi",
    outs = [
        "a/b",
        "c/d"
    ],
    cmd = "echo 'hi' | tee $(@D)/a/b $(@D)/c/d",
)
EOF

  bazel build @r//package:hi >$TEST_log 2>&1 || fail "Should build"
  expect_log bazel-genfiles/external/r/package/a/b
  expect_log bazel-genfiles/external/r/package/c/d
}

function test_python_with_workspace_name() {

 create_new_workspace
 cd ${new_workspace_dir}
 mkdir -p {module_a,module_b}
 local remote_path="${new_workspace_dir}"

 cat > module_a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "foo", srcs=["foo.py"])
EOF

 cat > module_b/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "bar", deps = ["//module_a:foo"], srcs=["bar.py"],)
py_binary(name = "bar2", deps = ["//module_a:foo"], srcs=["bar2.py"],)
EOF

 cat > module_a/foo.py <<EOF
def GetNumber():
  return 42
EOF

 cat > module_b/bar.py <<EOF
from module_a import foo
def PrintNumber():
  print "Print the number %d" % foo.GetNumber()
EOF

 cat > module_b/bar2.py <<EOF
from module_a import foo
print "The number is %d" % foo.GetNumber()
EOF

 cd ${WORKSPACE_DIR}
 mkdir -p {module1,module2}
 cat > WORKSPACE <<EOF
workspace(name = "foobar")
local_repository(name="remote", path="${remote_path}")
EOF
 cat > module1/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
py_library(name = "fib", srcs=["fib.py"],)
EOF
 cat > module2/BUILD <<EOF
py_binary(name = "bez",
  deps = ["@remote//module_a:foo", "@remote//module_b:bar", "//module1:fib"],
  srcs = ["bez.py"],)
EOF

cat > module1/fib.py <<EOF
def Fib(n):
  if n == 0 or n == 1:
    return 1
  else:
    return Fib(n-1) + Fib(n-2)
EOF

 cat > module2/bez.py <<EOF
from remote.module_a import foo
from remote.module_b import bar
from module1 import fib

print "The number is %d" % foo.GetNumber()
bar.PrintNumber()
print "Fib(10) is %d" % fib.Fib(10)
EOF
 bazel run //module2:bez >$TEST_log
 expect_log "The number is 42"
 expect_log "Print the number 42"
 expect_log "Fib(10) is 89"
 bazel run @remote//module_b:bar2 >$TEST_log
 expect_log "The number is 42"
}

function test_build_with_aliased_input_file() {
  mkdir -p a
  cat > a/BUILD <<EOF
exports_files(['f'])
alias(name='a', actual=':f')
EOF

  touch a/f
  bazel build //a:a || fail "build failed"
}

function test_visibility() {
  mkdir visibility
  cat > visibility/BUILD <<EOF
cc_library(
  name = "foo",
  visibility = [
    "//foo/bar:__pkg__",
    "//visibility:public",
  ],
)
EOF

  bazel build //visibility:foo &> $TEST_log && fail "Expected failure" || true
  expect_log "Public or private visibility labels (e.g. //visibility:public or //visibility:private) cannot be used in combination with other labels"
}

run_suite "rules test"
