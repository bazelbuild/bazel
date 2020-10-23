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

set -euo pipefail
# --- begin runfiles.bash initialization ---
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

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
  # The workspace name is initialized in testenv.sh; use that var rather than
  # hardcoding it here. The extra sed pass is so we can selectively expand that
  # one var while keeping the rest of the heredoc literal.
  sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" > mypkg/echoer.sh << 'EOF'
#!/bin/bash
set -euo pipefail
# --- begin runfiles.bash initialization ---
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

if [[ ! -e "$(rlocation {{WORKSPACE_NAME}}/mypkg/runfile)" ]]; then
  echo "ERROR: Runfile not found" >&2
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
    data = [
        "runfile",
        "@bazel_tools//tools/bash/runfiles",
    ],
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
  cat >pkg/BUILD  <<'EOF'
genrule(
    name = "test",
    outs = ["test.out"],
    cmd = "env > $@",
)
EOF

  local new_tmpdir="$(mktemp -d "${TEST_TMPDIR}/newfancytmpdirXXXXXX")"
  if $is_windows; then
    PATH="/random/path:$PATH" TMP="${new_tmpdir}" \
      bazel build //pkg:test --spawn_strategy=standalone --action_env=PATH \
      &> $TEST_log || fail "Failed to build //pkg:test"

    # Test that Bazel respects the client environment's TMP.
    # new_tmpdir is based on $TEST_TMPDIR which is not Unix-style -- convert it.
    assert_contains "TMP=$(cygpath -u "${new_tmpdir}")" bazel-bin/pkg/test.out
  else
    PATH="/random/path:$PATH" TMPDIR="${new_tmpdir}" \
      bazel build //pkg:test --spawn_strategy=standalone --action_env=PATH \
      &> $TEST_log || fail "Failed to build //pkg:test"

    # Test that Bazel respects the client environment's TMPDIR.
    assert_contains "TMPDIR=${new_tmpdir}" bazel-bin/pkg/test.out
  fi

  # Test that Bazel passed through the PATH from --action_env.
  assert_contains "PATH=/random/path" bazel-bin/pkg/test.out
}

function test_genrule_remote() {
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
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
#!/bin/sh
echo "Hi"
EOF
  chmod +x package/in.sh

  bazel build @r//package:abs_dep >$TEST_log 2>&1 || fail "Should build"
}

function test_genrule_remote_d() {
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
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
  expect_log "bazel-.*bin/external/r/package/a/b"
  expect_log "bazel-.*bin/external/r/package/c/d"
}

function test_genrule_toolchain_dependency {
  mkdir -p t
  cat > t/BUILD <<EOF
genrule(
    name = "toolchain_check",
    outs = ["version"],
    toolchains = ['@bazel_tools//tools/jdk:current_host_java_runtime'],
    cmd = "ls -al \$(JAVABASE) > \$@",
)
EOF
  bazel build //t:toolchain_check >$TEST_log 2>&1 || fail "Should build"
  expect_log "bazel-.*bin/t/version"
  expect_not_log "ls: cannot access"
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
  print("Print the number %d" % foo.GetNumber())
EOF

 cat > module_b/bar2.py <<EOF
from module_a import foo
print("The number is %d" % foo.GetNumber())
EOF

 cd ${WORKSPACE_DIR}
 mkdir -p {module1,module2}
 rm WORKSPACE
 cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
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
  if n < 2:
    return 1
  else:
    a = 1
    b = 1
    i = 2
    while i <= n:
      c = a + b
      a = b
      b = c
      i += 1
    return b
EOF

 cat > module2/bez.py <<EOF
from remote.module_a import foo
from remote.module_b import bar
from module1 import fib

print("The number is %d" % foo.GetNumber())
bar.PrintNumber()
print("Fib(10) is %d" % fib.Fib(10))
EOF
 bazel run //module2:bez >$TEST_log
 expect_log "The number is 42"
 expect_log "Print the number 42"
 expect_log "Fib(10) is 89"
 bazel run @remote//module_b:bar2 >$TEST_log
 expect_log "The number is 42"
}

function test_build_python_zip_with_middleman() {
  mkdir py
  touch py/data.txt
  cat > py/BUILD <<EOF
py_binary(name = "bin", srcs = ["bin.py"], data = ["data.txt"])
py_binary(name = "bin2", srcs = ["bin2.py"], data = [":bin"])
EOF
  cat > py/bin.py <<EOF
print("hello")
EOF
  cat > py/bin2.py <<EOF
print("world")
EOF
  bazel build --build_python_zip //py:bin2 || fail "build failed"
  # `unzip` prints the right output but exits with non-zero, because the zip
  # file starts with a shebang line. Capture the output and swallow this benign
  # error, and only assert the output.
  local found=$(unzip -l ./bazel-bin/py/bin2 | grep "data.txt" || echo "")
  [[ -n "$found" ]] || fail "failed to zip data file"
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

function test_executable_without_default_files() {
  mkdir pkg
  cat >pkg/BUILD <<'EOF'
load(":rules.bzl", "bin_rule", "out_rule")
bin_rule(name = "hello_bin")
out_rule(name = "hello_out")

genrule(
    name = "hello_gen",
    tools = [":hello_bin"],
    outs = ["hello_gen.txt"],
    cmd = "$(location :hello_bin) $@",
)
EOF

  # On Windows this file needs to be acceptable by CreateProcessW(), rather
  # than a Bourne script.
  if "$is_windows"; then
    cat >pkg/rules.bzl <<'EOF'
_SCRIPT_EXT = ".bat"
_SCRIPT_CONTENT = "@ECHO OFF\necho hello world > %1"
EOF
  else
    cat >pkg/rules.bzl <<'EOF'
_SCRIPT_EXT = ".sh"
_SCRIPT_CONTENT = "#!/bin/sh\necho 'hello world' > $@"
EOF
  fi

  cat >>pkg/rules.bzl <<'EOF'
def _bin_rule(ctx):
    out_sh = ctx.actions.declare_file(ctx.attr.name + _SCRIPT_EXT)
    ctx.actions.write(
        output = out_sh,
        content = _SCRIPT_CONTENT,
        is_executable = True,
    )
    return DefaultInfo(
        files = depset(direct = []),
        executable = out_sh,
    )

def _out_rule(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".txt")
    ctx.actions.run(
        executable = ctx.executable._hello_bin,
        outputs = [out],
        arguments = [out.path],
        mnemonic = "HelloOut",
    )
    return DefaultInfo(
        files = depset(direct = [out]),
    )

bin_rule = rule(_bin_rule, executable = True)
out_rule = rule(_out_rule, attrs = {
    "_hello_bin": attr.label(
        default = ":hello_bin",
        executable = True,
        cfg = "host",
    ),
})
EOF

  bazel build //pkg:hello_out //pkg:hello_gen >$TEST_log 2>&1 || fail "Should build"
  assert_contains "hello world" bazel-bin/pkg/hello_out.txt
  assert_contains "hello world" bazel-bin/pkg/hello_gen.txt
}

run_suite "rules test"
