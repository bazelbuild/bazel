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
# Test sandboxing spawn strategy
#

# Load test environment
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/../sandboxing_test_utils.sh \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }
function set_up {
  add_to_bazelrc "build --spawn_strategy=sandboxed"
  add_to_bazelrc "build --genrule_strategy=sandboxed"
}

function test_sandboxed_tooldir() {
  mkdir -p examples/genrule

  cat << 'EOF' > examples/genrule/BUILD
genrule(
   name = "tooldir",
   srcs = [],
   outs = ["tooldir.txt"],
   cmd = "ls -l external/bazel_tools/tools/genrule | tee $@ >&2; " +
       "cat external/bazel_tools/tools/genrule/genrule-setup.sh >&2",
)
EOF

  bazel build examples/genrule:tooldir &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tooldir"
  [ -f "bazel-genfiles/examples/genrule/tooldir.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:works"
  cat "bazel-genfiles/examples/genrule/tooldir.txt" > $TEST_log
  expect_log "genrule-setup.sh"
}

function test_sandbox_block_filesystem() {
  # The point of this test is to attempt to read something from the filesystem
  # that is blocked via --sandbox_block_path= and thus shouldn't be accessible.
  #
  # /var/log is an arbitrary choice of directory that should exist on all
  # Unix-like systems.
  local block_path
  case "$(uname -s)" in
    Darwin)
      # TODO(jmmv): sandbox-exec does not resolve symlinks, so attempting
      # to block /var/log does not work. Unsure if we should make this work
      # by resolving symlinks or documenting the expected behavior.
      block_path=/private/var/log
      ;;
    *)
      block_path=/var/log
      ;;
  esac

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
  name = "breaks",
  srcs = [ "a.txt" ],
  outs = [ "breaks.txt" ],
  cmd = "ls ${block_path} &> \$@",
)
EOF
  touch pkg/a.txt

  local output_file="bazel-genfiles/pkg/breaks.txt"

  bazel build --sandbox_block_path="${block_path}" pkg:breaks \
    &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"
  cat "${output_file}" >$TEST_log

  if [ "$(wc -l $output_file | awk '{print $1}')" -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  grep -E "(Operation not permitted|Permission denied)" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

# TODO(philwo) - this doesn't work on Ubuntu 14.04 due to "unshare" being too
# old and not understanding the --user flag.
function DISABLED_test_sandbox_different_nobody_uid() {
  cat /etc/passwd | sed 's/\(^nobody:[^:]*:\)[0-9]*:[0-9]*/\15000:16000/g' > \
      "${TEST_TMPDIR}/passwd"
  unshare --user --mount --map-root-user -- bash - \
      << EOF || fail "Hermetic genrule with different UID for nobody failed" \
set -e
set -u

mount --bind ${TEST_TMPDIR}/passwd /etc/passwd
bazel build examples/genrule:works &> ${TEST_log}
EOF
}

# Tests that a pseudoterminal can be opened in linux when --sandbox_explicit_pseudoterminal is active
function test_can_enable_pseudoterminals() {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: flag intended for linux systems"
    return 0
  fi

  cat > test.py <<'EOF'
import pty
pty.openpty()
EOF
  cat > BUILD <<'EOF'
py_test(
  name = "test",
  srcs = ["test.py"],
)
EOF
  bazel test --sandbox_explicit_pseudoterminal --verbose_failures --sandbox_debug :test || fail "test did not pass"
}

function test_sandbox_debug() {
  if [[ "$PLATFORM" == "darwin" ]]; then
    # The process wrapper sandbox used in MacOS doesn't emit debug output.
    return 0
  fi

  cat > BUILD <<'EOF'
genrule(
  name = "broken",
  outs = ["bla.txt"],
  cmd = "exit 1",
)
EOF
  bazel build --verbose_failures :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging"
  expect_log "Executing genrule //:broken failed"

  bazel build --verbose_failures --sandbox_debug :broken &> $TEST_log \
    && fail "build should have failed" || true
  expect_log "Executing genrule //:broken failed"
  expect_not_log "Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging"
  expect_log "child exited normally with code 1"

  bazel build --verbose_failures --sandbox_debug --experimental_use_hermetic_linux_sandbox :broken &> $TEST_log \
    && fail "build should have failed with hermetic sandbox" || true
  expect_log "child exited normally with code 1"

  bazel build --verbose_failures --sandbox_debug --incompatible_sandbox_hermetic_tmp :broken &> $TEST_log \
    && fail "build should have failed with hermetic sandbox /tmp" || true
  expect_log "child exited normally with code 1"
}

function test_sandbox_expands_tree_artifacts_in_runfiles_tree() {
  create_workspace_with_default_repos WORKSPACE

  cat > def.bzl <<'EOF'
def _mkdata_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name + ".d")
    script = "mkdir -p {out}; touch {out}/file; ln -s file {out}/link".format(out = out.path)
    ctx.actions.run_shell(
        outputs = [out],
        command = script,
    )
    runfiles = ctx.runfiles(files = [out])
    return [DefaultInfo(
        files = depset([out]),
        runfiles = runfiles,
    )]

mkdata = rule(
    _mkdata_impl,
)
EOF

  cat > mkdata_test.sh <<'EOF'
#!/bin/bash

set -euo pipefail

test_dir="$1"
cd "$test_dir"
ls -l | cut -f1,9 -d' ' >&2

if [ ! -f file -o -L file ]; then
  echo "'file' is not a regular file" >&2
  exit 1
fi
EOF
  chmod +x mkdata_test.sh

  cat > BUILD <<'EOF'
load("//:def.bzl", "mkdata")

mkdata(name = "mkdata")

sh_test(
    name = "mkdata_test",
    srcs = ["mkdata_test.sh"],
    args = ["$(location :mkdata)"],
    data = [":mkdata"],
)
EOF

  bazel test --test_output=streamed //:mkdata_test &>$TEST_log && fail "expected test to fail" || true
  expect_log "'file' is not a regular file"
}

# regression test for https://github.com/bazelbuild/bazel/issues/6262
function test_create_tree_artifact_outputs() {
  create_workspace_with_default_repos WORKSPACE

  cat > def.bzl <<'EOF'
def _r(ctx):
    d = ctx.actions.declare_directory("%s_dir" % ctx.label.name)
    ctx.actions.run_shell(
        outputs = [d],
        command = "cd %s && pwd" % d.path,
    )
    return [DefaultInfo(files = depset([d]))]

r = rule(implementation = _r)
EOF

cat > BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build --test_output=streamed :a &>$TEST_log || fail "expected build to succeed"
}

function test_empty_tree_artifact_as_inputs() {
  # Test that when an empty tree artifact is the input, an empty directory is
  # created in the sandbox for action to read.
  create_workspace_with_default_repos WORKSPACE

  mkdir -p pkg

  cat > pkg/def.bzl <<'EOF'
def _r(ctx):
    empty_d = ctx.actions.declare_directory("%s/empty_dir" % ctx.label.name)
    ctx.actions.run_shell(
        outputs = [empty_d],
        command = "mkdir -p %s" % empty_d.path,
    )
    f = ctx.actions.declare_file("%s/file" % ctx.label.name)
    ctx.actions.run_shell(
        inputs = [empty_d],
        outputs = [f],
        command = "touch %s && cd %s && pwd" % (f.path, empty_d.path),
    )
    return [DefaultInfo(files = depset([f]))]

r = rule(implementation = _r)
EOF

cat > pkg/BUILD <<'EOF'
load(":def.bzl", "r")

r(name = "a")
EOF

  bazel build //pkg:a &>$TEST_log || fail "expected build to succeed"
}

# The test shouldn't fail if the environment doesn't support running it.
check_sandbox_allowed || exit 0

run_suite "sandbox"
