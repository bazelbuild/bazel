#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# modify_execution_info_test.sh: tests of the --modify_execution_info flag.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### HELPER FUNCTIONS ##################################################

if ! type try_with_timeout >&/dev/null; then
  # Bazel's testenv.sh defines try_with_timeout but the Google-internal version
  # uses a different testenv.sh.
  function try_with_timeout() { $* ; }
fi

function set_up() {
    cd ${WORKSPACE_DIR}
}

function tear_down() {
  try_with_timeout bazel shutdown
}

#### TESTS #############################################################

function test_aquery_respects_modify_execution_info_changes {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(name = "bar", outs = ["bar_out.txt"], cmd = "touch $(OUTS)")
EOF
  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=+requires-x  \
    > output1 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: ''}" output1

  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=+requires-y \
    > output2 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-y: ''}" output2
}

function test_modify_execution_info_multiple {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(
    name = "bar",
    outs = ["bar_out.txt"],
    cmd = "touch $(OUTS)",
    tags = ["requires-x"],
)
cc_binary(name="zero", srcs=["zero.cc"])
EOF
  echo "int main(void) {}" > "$pkg/zero.cc"

  # multiple elements in the value list that match the same mnemonic.
  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=+requires-y,Genrule=+requires-z \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: '', requires-y: '', "\
"requires-z: ''}" output

  # multiple elements in the value list, the first of which adds an
  # ExecutionInfo and the second of which removes it.
  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=+requires-z,.*=-requires-z \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: ''}" output

  # multiple elements in the value list, the first of which removes an
  # ExecutionInfo (previously absent) and the second of which adds it.
  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=-requires-z,.*=+requires-z \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: '', requires-z: ''}" output

  # multiple elements in the value list, the first of which removes an
  # ExecutionInfo (previously present) and the second of which adds it back.
  bazel aquery --output=text "//$pkg:bar" \
   --modify_execution_info=Genrule=-requires-x,.*=+requires-x \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: ''}" output

  # multiple elements with multiple values
  bazel aquery --output=text "//$pkg:all" \
   --modify_execution_info=Genrule=-requires-x,Genrule=+requires-z,\
Genrule=+requires-a,CppCompile=+requires-b,CppCompile=+requires-c \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-a: '', requires-z: ''}" output
  assert_contains "ExecutionInfo: {requires-b: '', requires-c: ''" output

  # negative lookahead
  bazel aquery --output=text "//$pkg:all" \
   --modify_execution_info='(?!Genrule).*=+requires-a,(?!CppCompile).*=+requires-z' \
    > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "ExecutionInfo: {requires-x: '', requires-z: ''}" output
  assert_contains "ExecutionInfo: {requires-a: ''" output
}

# TODO(davido): Find a way to re-use the same protobuf's BUILD file
function test_modify_execution_info_various_types() {
  if [[ "$PRODUCT_NAME" = "bazel" ]]; then
    # proto_library requires this external workspace.
    cat >> WORKSPACE << EOF
workspace(name = "io_bazel")
new_local_repository(
    name = "com_google_protobuf",
    path = "$(dirname $(rlocation io_bazel/third_party/protobuf/3.7.1/BUILD_without_zlib_dep.bazel))",
    build_file = "$(rlocation io_bazel/third_party/protobuf/3.7.1/BUILD_without_zlib_dep.bazel)",
)
new_local_repository(
    name = "zlib",
    path = "$(dirname $(rlocation io_bazel/third_party/zlib/BUILD))",
    build_file = "$(rlocation io_bazel/third_party/zlib/BUILD)",
)
EOF
  fi
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  echo "load('//$pkg:shell.bzl', 'skylark_shell')" > "$pkg/BUILD"
  cat >> "$pkg/BUILD" <<'EOF'
skylark_shell(
  name = "shelly",
  output = "ok.txt",
)

cc_binary(name="zero", srcs=["zero.cc"])

sh_test(name="test_a", srcs=["a.sh"])

java_library(name = "javalib", srcs = ["HelloWorld.java"])

action_listener(
  name = "al",
  extra_actions = [":echo-filename"],
  mnemonics = ["Javac"],
  visibility = ["//visibility:public"],
)

extra_action(name = "echo-filename", cmd = "echo Hi \$(EXTRA_ACTION_FILE)")

py_binary(name = "pybar", srcs=["pybar.py"],)

proto_library(name = "proto", srcs=["foo.proto"])
EOF
  cat > "$pkg/shell.bzl" <<'EOF'
def _impl(ctx):
  ctx.actions.run_shell(
    outputs = [ ctx.outputs.output ],
    command = "touch %s" % ctx.outputs.output.path,
  )

skylark_shell = rule(
  _impl,
  attrs = {
    "output": attr.output(mandatory=True),
  }
)
EOF
  cat > "$pkg/a.sh" <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod 755 "$pkg/a.sh"
  echo "int main(void) {}" > "$pkg/zero.cc"
  echo "public class HelloWorld {}" > "$pkg/HelloWorld.java"
  echo 'print("Hi")' > "$pkg/pybar.py"
  echo 'syntax="proto2"; package foo;' > "$pkg/foo.proto"

  bazel aquery --output=text "//$pkg:all" \
   --experimental_action_listener=$pkg:al \
   --modify_execution_info=\
echo.*=+requires-extra-action,\
.*Proto.*=+requires-proto,\
CppCompile=+requires-cpp-compile,\
CppLink=+requires-cpp-link,\
TestRunner=+requires-test-runner,\
Turbine=+requires-turbine,\
JavaSourceJar=+requires-java-source-jar,\
Javac=+requires-javac,\
PyTinypar=+requires-py-tinypar,\
SkylarkAction=+requires-skylark-action \
   > output 2> "$TEST_log" || fail "Expected success"

  # There are sometimes other elements in ExecutionInfo, e.g. requires-darwin
  # for obj-c, supports-workers for java.  Since testing for these combinations
  # would be brittle, irrelevant to the operation of the flag, and in some
  # cases platform-dependent, we just search for the key itself, not the whole
  # ExecutionInfo: {...} line.
  assert_contains "requires-skylark-action: ''" output
  assert_contains "requires-cpp-compile: ''" output
  assert_contains "requires-cpp-link: ''" output
  assert_contains "requires-extra-action: ''" output
  assert_contains "requires-test-runner: ''" output
  assert_contains "requires-javac: ''" output
  assert_contains "requires-turbine: ''" output
  assert_contains "requires-java-source-jar: ''" output
  assert_contains "requires-proto: ''" output  # GenProtoDescriptorSet should match
  if [[ "$PRODUCT_NAME" != "bazel" ]]; then
    # Python rules generate some cpp actions and local actions, but py-tinypar
    # is the main unique-to-python rule which runs remotely for a py_binary.
    assert_contains "requires-py-tinypar: ''" output
  fi
}

run_suite "Integration tests of the --modify_execution_info option."
