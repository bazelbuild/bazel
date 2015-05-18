#!/bin/sh
#
# Copyright 2015 Google Inc. All rights reserved.
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
# Test sandboxing spawn strategy
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# namespaces which are used by the sandbox were introduced in 3.8, so
# test won't run on earlier kernels
function check_kernel_version {
  if [ "${PLATFORM-}" = "darwin" ]; then
    echo "Test will skip: sandbox is not yet supported on Darwin."
    exit 0
  fi
  MAJOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\1/')
  MINOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\2/')
  if [ $MAJOR -lt 3 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
  if [ $MAJOR -eq 3 ] && [ $MINOR -lt 8 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
}

function set_up {
   mkdir -p examples/genrule
   cat << 'EOF' > examples/genrule/a.txt
foo bar bz
EOF
   cat << 'EOF' > examples/genrule/b.txt
apples oranges bananas
EOF
   cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "works",
  srcs = [ "a.txt" ],
  outs = [ "works.txt" ],
  cmd = "wc a.txt > $@",
)

sh_binary(
    name = "tool",
    srcs = ["tool.sh"],
    data = ["datafile"],
)

genrule(
    name = "tools_work",
    srcs = [],
    outs = ["tools.txt"],
    cmd = "$(location :tool) $@",
    tools = [":tool"],
)

genrule(
   name = "tooldir",
   srcs = [],
   outs = ["tooldir.txt"],
   cmd = "ls -l tools/genrule | tee $@ >&2; cat tools/genrule/genrule-setup.sh >&2",
)

genrule(
  name = "breaks1",
  srcs = [ "a.txt" ],
  outs = [ "breaks1.txt" ],
  cmd = "wc b.txt a.txt > $@",
)

genrule(
  name = "breaks2",
  srcs = [ "a.txt" ],
  outs = [ "breaks2.txt" ],
  # the point of this test is to attempt to read something from filesystem
  # that resides outside sandbox by using an absolute path to that file
  #
  # /home is an arbitrary choice of directory (we doesn't mount it in sandbox
  # and it should exist on every linux) which could be changed in case it turns
  # out it's necessary to put it in sandbox
  #
  cmd = "ls /home > $@",
)
EOF
  cat << 'EOF' >> examples/genrule/datafile
this is a datafile
EOF
  cat << 'EOF' >> examples/genrule/tool.sh
#!/bin/sh

set -e
cp examples/genrule/datafile $1
echo "Tools work!"
EOF
chmod +x examples/genrule/tool.sh
}

function test_sandboxed_genrule() {
  bazel build --genrule_strategy=sandboxed --verbose_failures \
    examples/genrule:works \
    || fail "Hermetic genrule failed: examples/genrule:works"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/works.txt" ] \
    || fail "Genrule didn't produce output: examples/genrule:works"
}

function test_sandboxed_tooldir() {
  bazel build --genrule_strategy=sandboxed --verbose_failures \
    examples/genrule:tooldir \
    || fail "Hermetic genrule failed: examples/genrule:tooldir"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/tooldir.txt" ] \
    || fail "Genrule didn't produce output: examples/genrule:works"
  cat "${BAZEL_GENFILES_DIR}/examples/genrule/tooldir.txt" > $TEST_log
  expect_log "genrule-setup.sh"
}

function test_sandboxed_genrule_with_tools() {
  bazel build --genrule_strategy=sandboxed --verbose_failures \
    examples/genrule:tools_work \
    || fail "Hermetic genrule failed: examples/genrule:tools_work"
  [ -f "${BAZEL_GENFILES_DIR}/examples/genrule/tools.txt" ] \
    || fail "Genrule didn't produce output: examples/genrule:tools_work"
}

function test_sandbox_undeclared_deps() {
  bazel build --genrule_strategy=sandboxed --verbose_failures \
    examples/genrule:breaks1 \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks1" || true
  [ ! -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks1.txt" ] || {
    output=$(cat "${BAZEL_GENFILES_DIR}/examples/genrule/breaks1.txt")
    fail "Non-hermetic genrule breaks1 suceeded with following output: $(output)"
  }
}

function test_sandbox_block_filesystem() {
  bazel build --genrule_strategy=sandboxed --verbose_failures \
    examples/genrule:breaks2 \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks2" || true
  [ ! -f "${BAZEL_GENFILES_DIR}/examples/genrule/breaks2.txt" ] || {
    output=$(cat "${BAZEL_GENFILES_DIR}/examples/genrule/breaks2.txt")
    fail "Non-hermetic genrule suceeded with following output: $(output)"
  }
}

check_kernel_version
run_suite "sandbox"
