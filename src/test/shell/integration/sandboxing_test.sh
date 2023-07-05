#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  add_to_bazelrc "build --spawn_strategy=sandboxed"
  add_to_bazelrc "build --genrule_strategy=sandboxed"
}

function tear_down() {
  bazel clean --expunge
  bazel shutdown
  rm -rf pkg
}

function do_sandbox_base_wiped_only_on_startup_test {
  local extra_args=( "${@}" )

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "echo >\$@")
EOF

  local output_base="$(bazel info output_base)"

  do_build() {
    bazel build --sandbox_debug "${extra_args[@]}" //pkg
  }

  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  find "${output_base}" >>"${TEST_log}" 2>&1 || true

  local sandbox_dir="$(echo "${output_base}/sandbox"/*-sandbox)"
  [[ -d "${sandbox_dir}" ]] \
    || fail "${sandbox_dir} is missing; prematurely deleted?"

  local garbage="${output_base}/sandbox/garbage"
  mkdir -p "${garbage}/some/nested/contents"
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_not_log "Deleting stale sandbox"
  [[ -d "${garbage}" ]] \
    || fail "Spurious contents deleted from sandbox base too early"

  bazel shutdown
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_log "Deleting stale sandbox"
  [[ ! -d "${garbage}" ]] \
    || fail "sandbox base was not cleaned on restart"
}

function test_sandbox_base_wiped_only_on_startup_with_sync_deletions() {
  do_sandbox_base_wiped_only_on_startup_test \
    --experimental_sandbox_async_tree_delete_idle_threads=0
}

function test_sandbox_base_wiped_only_on_startup_with_async_deletions() {
  do_sandbox_base_wiped_only_on_startup_test \
    --experimental_sandbox_async_tree_delete_idle_threads=HOST_CPUS
}

function do_succeed_when_executor_not_initialized_test() {
  local extra_args=( "${@}" )

  mkdir pkg
  mkfifo pkg/BUILD

  bazel build --nobuild "${@}" //pkg:all \
    >"${TEST_log}" 2>&1 &
  local pid="${!}"

  echo "Waiting for Blaze to finish initializing all modules"
  while ! grep "currently loading: pkg" "${TEST_log}"; do
    sleep 1
  done

  echo "Interrupting Blaze before it gets to init the executor"
  kill "${pid}"

  echo "And now giving Blaze a chance to finalize all modules"
  echo "unblock fifo" >pkg/BUILD
  wait "${pid}" || true

  expect_log "Build did NOT complete successfully"
  # Disallow some common messages we might see during a crash.
  expect_not_log "Internal error"
  expect_not_log "stack trace"
  expect_not_log "NullPointerException"
}

function test_succeed_when_executor_not_initialized_with_defaults() {
  # Pass a no-op flag to the test to workaround a bug in macOS's default
  # and ancient bash version which causes it to error out on an empty
  # argument list when $@ is consumed and set -u is enabled.
  local noop=( --nobuild )

  do_succeed_when_executor_not_initialized_test "${noop[@]}"
}

function test_succeed_when_executor_not_initialized_with_async_deletions() {
  do_succeed_when_executor_not_initialized_test \
    --experimental_sandbox_async_tree_delete_idle_threads=auto
}

function test_sandbox_base_can_be_rm_rfed() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "echo >\$@")
EOF

  local output_base="$(bazel info output_base)"

  do_build() {
    bazel build --sandbox_debug //pkg
  }

  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  find "${output_base}" >>"${TEST_log}" 2>&1 || true

  local sandbox_base="${output_base}/sandbox"
  [[ -d "${sandbox_base}" ]] \
    || fail "${sandbox_base} is missing; build did not use sandboxing?"

  # Ensure the sandbox base does not contain protected files that would prevent
  # a simple "rm -rf" from working under an unprivileged user.
  rm -rf "${sandbox_base}" || fail "Cannot clean sandbox base"

  # And now ensure Bazel reconstructs the sandbox base on a second build.
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
}

function test_sandbox_not_used_with_legacy_fallback() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build \
    --incompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"

  expect_not_log "${output_base}.*/sandbox/"
  expect_log "implicit fallback from sandbox to local"
}

function test_sandbox_local_not_used_without_legacy_fallback() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build \
    --noincompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
  # Still warning in this case even when the flag is flipped
  expect_log "implicit fallback from sandbox to local"
}

function test_sandbox_local_used_with_proper_strategy() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "pwd; echo >\$@",
  tags = ["no-sandbox"])
EOF

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"
  rm -rf ${sandbox_base}

  bazel build --genrule_strategy=sandboxed,standalone \
    --noincompatible_legacy_local_fallback //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"

  expect_not_log "${output_base}.*/sandbox/"
  expect_not_log "implicit fallback from sandbox to local"
}

function test_sandbox_base_top_is_removed() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(name = "pkg", outs = ["pkg.out"], cmd = "echo >\$@")
EOF

  local output_base="$(bazel info output_base)"

  do_build() {
    bazel build //pkg
  }

  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  find "${output_base}" >>"${TEST_log}" 2>&1 || true

  local sandbox_base="${output_base}/sandbox"
  [[ ! -d "${sandbox_base}" ]] \
    || fail "${sandbox_base} left behind unnecessarily"

  # Restart Bazel and check we don't print spurious "Deleting stale sandbox"
  # warnings.
  bazel shutdown
  do_build >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  expect_not_log "Deleting stale sandbox"
}

function test_sandbox_old_contents_not_reused_in_consecutive_builds() {
  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
    name = "pkg",
    srcs = ["pkg.in"],
    outs = ["pkg.out"],
    cmd = "cp \$(location :pkg.in) \$@",
)
EOF
  touch pkg/pkg.in

  for i in $(seq 5); do
    # Ensure that, even if we don't clean up the sandbox at all (with
    # --sandbox_debug), consecutive builds don't step on each other by trying to
    # reuse previous spawn identifiers.
    bazel build --sandbox_debug //pkg \
      >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
    echo foo >>pkg/pkg.in
  done
}

function test_sandbox_hardening_with_cgroups_v1() {
  if ! grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts; then
    echo "No cgroup v1 memory controller mounted, skipping test"
    return 0
  fi
  memmount=$(grep -E '^cgroup +[^ ]+ +cgroup +.*memory.*' /proc/mounts | cut -d' ' -f2)
  if ! grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup &>/dev/null; then
    echo "Does not use cgroups v1, skipping test"
    return 0
  fi
  memsubdir=$(grep -E '^[0-9]*:[^:]*memory[^:]*:' /proc/self/cgroup | cut -d: -f3)
  memdir="$memmount$memsubdir"
  if [[ ! -w "$memdir" ]]; then
    echo "Cgroups v1 directory not writable, skipping test"
    return 0
  fi

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
  name = "pkg",
  outs = ["pkg.out"],
  cmd = "pwd; hexdump -C -n 250000 < /dev/random | sort > /dev/null 2>\$@"
)
EOF
  local genfiles_base="$(bazel info ${PRODUCT_NAME}-genfiles)"

  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit_mb=1000 //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  rm -f ${genfiles_base}/pkg/pkg.out
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit_mb=1 //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
}

function test_sandbox_hardening_with_cgroups_v2() {
  if ! grep -E '^cgroup2 +[^ ]+ +cgroup2 ' /proc/mounts; then
    echo "No cgroup2 mounted, skipping test"
    return 0
  fi
  if ! grep -E '^0::' /proc/self/cgroup &>/dev/null; then
    echo "Does not use cgroups v2, skipping test"
    return 0
  fi
  if ! XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope true; then
    echo "Not able to use systemd, skipping test"
    return 0
  fi

  mkdir pkg
  cat >pkg/BUILD <<EOF
genrule(
  name = "pkg",
  outs = ["pkg.out"],
  cmd = "pwd; hexdump -C -n 250000 < /dev/random | sort > /dev/null 2>\$@"
)
EOF
  local genfiles_base="$(bazel info ${PRODUCT_NAME}-genfiles)"
  # Need to make sure the bazel server runs under systemd, too.
  bazel shutdown

  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit_mb=1000 //pkg \
    >"${TEST_log}" 2>&1 || fail "Expected build to succeed"
  rm -f ${genfiles_base}/pkg/pkg.out

  bazel shutdown
  XDG_RUNTIME_DIR=/run/user/$( id -u ) systemd-run --user --scope \
  bazel build --genrule_strategy=linux-sandbox \
    --experimental_sandbox_memory_limit_mb=1 //pkg \
    >"${TEST_log}" 2>&1 && fail "Expected build to fail" || true
}

function test_sandboxed_genrule() {
  mkdir -p examples/genrule
  cat << 'EOF' > examples/genrule/a.txt
foo bar bz
EOF
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "works",
  srcs = [ "a.txt" ],
  outs = [ "works.txt" ],
  cmd = "wc $(location :a.txt) > $@",
)
EOF

  bazel build examples/genrule:works &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:works"
  [ -f "bazel-genfiles/examples/genrule/works.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:works"
}

function test_sandboxed_genrule_with_tools() {
  mkdir -p examples/genrule

  cat << 'EOF' > examples/genrule/BUILD
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
EOF
  cat << 'EOF' >> examples/genrule/datafile
this is a datafile
EOF
  # The workspace name is initialized in testenv.sh; use that var rather than
  # hardcoding it here. The extra sed pass is so we can selectively expand that
  # one var while keeping the rest of the heredoc literal.
  cat | sed "s/{{WORKSPACE_NAME}}/$WORKSPACE_NAME/" >> examples/genrule/tool.sh << 'EOF'
#!/bin/sh

set -e
cp $(dirname $0)/tool.runfiles/{{WORKSPACE_NAME}}/examples/genrule/datafile $1
echo "Tools work!"
EOF
  chmod +x examples/genrule/tool.sh

  bazel build examples/genrule:tools_work &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:tools_work"
  [ -f "bazel-genfiles/examples/genrule/tools.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:tools_work"
}

# Test for #400: Linux sandboxing and relative symbolic links.
#
# let A = examples/genrule/symlinks/a/b/x.txt -> ../x.txt
# where   examples/genrule/symlinks/a/b -> examples/genrule/symlinks/ok/sub
# thus the realpath of A is example/genrule/symlinks/ok/x.txt
# but if the code doesn't correctly resolve intermediate symlinks and instead
# uses string operations to handle ".." parts, it will arrive at:
# examples/genrule/symlinks/a/x.txt, which is wrong.
#
function test_sandbox_relative_symlink_in_inputs() {
  mkdir -p examples/genrule

  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "relative_symlinks",
  srcs = [ "symlinks/a/b/x.txt" ],
  outs = [ "relative_symlinks.txt" ],
  cmd = "cat $(location :symlinks/a/b/x.txt) > $@",
)
EOF

    mkdir -p examples/genrule/symlinks/{a,ok/sub}
    echo OK > examples/genrule/symlinks/ok/x.txt
    ln -s $PWD/examples/genrule/symlinks/ok/sub examples/genrule/symlinks/a/b
    ln -s ../x.txt examples/genrule/symlinks/a/b/x.txt

  bazel build examples/genrule:relative_symlinks &> $TEST_log \
    || fail "Hermetic genrule failed: examples/genrule:relative_symlinks"
  [ -f "bazel-genfiles/examples/genrule/relative_symlinks.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:relative_symlinks"
}

function test_sandbox_undeclared_deps() {
  output_file="bazel-genfiles/examples/genrule/breaks1.txt"

  mkdir -p examples/genrule
  cat << 'EOF' > examples/genrule/a.txt
foo bar bz
EOF
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "breaks1",
  srcs = [ "a.txt" ],
  outs = [ "breaks1.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt &> $@",
)
EOF

  bazel build examples/genrule:breaks1 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:breaks1" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "No such file or directory" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

function test_sandbox_undeclared_deps_with_local() {
  mkdir -p examples/genrule
  echo "foo bar bz" > examples/genrule/a.txt
  echo "apples oranges bananas" > examples/genrule/b.txt
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "breaks1_works_with_local",
  srcs = [ "a.txt" ],
  outs = [ "breaks1_works_with_local.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt > $@",
  local = 1,
)
EOF
  bazel build --incompatible_legacy_local_fallback \
    examples/genrule:breaks1_works_with_local &> $TEST_log \
    || fail "Non-hermetic genrule failed even though local=1: examples/genrule:breaks1_works_with_local"
  [ -f "bazel-genfiles/examples/genrule/breaks1_works_with_local.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks1_works_with_local"
}

function test_sandbox_undeclared_deps_with_local_tag() {
  mkdir -p examples/genrule
  echo "foo bar bz" > examples/genrule/a.txt
  echo "apples oranges bananas" > examples/genrule/b.txt
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "breaks1_works_with_local_tag",
  srcs = [ "a.txt" ],
  outs = [ "breaks1_works_with_local_tag.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt > $@",
  tags = [ "local" ],
)
EOF
  bazel build --incompatible_legacy_local_fallback \
    examples/genrule:breaks1_works_with_local_tag &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['local']: examples/genrule:breaks1_works_with_local_tag"
  [ -f "bazel-genfiles/examples/genrule/breaks1_works_with_local_tag.txt" ] \
    || fail "Genrule did not produce output: examples/genrule:breaks1_works_with_local_tag"
}

function test_sandbox_undeclared_deps_with_local_tag_no_fallback() {
  mkdir -p examples/genrule
  echo "foo bar bz" > examples/genrule/a.txt
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "breaks1_works_with_local_tag",
  srcs = [ "a.txt" ],
  outs = [ "breaks1_works_with_local_tag.txt" ],
  cmd = "wc $(location :a.txt) `dirname $(location :a.txt)`/b.txt > $@",
  tags = [ "local" ],
)
EOF
  bazel build examples/genrule:breaks1_works_with_local_tag &> $TEST_log \
    && fail "Non-hermetic genrule suucceeded even though tags=['local']: examples/genrule:breaks1_works_with_local_tag" \
    || true
}

function write_starlark_breaks1() {
  cat << 'EOF' >> examples/genrule/starlark.bzl
def _starlark_breaks1_impl(ctx):
  print(ctx.outputs.output.path)
  ctx.actions.run_shell(
    inputs = [ ctx.file.input ],
    outputs = [ ctx.outputs.output ],
    command = "wc %s `dirname %s`/b.txt &> %s" % (ctx.file.input.path,
                                                 ctx.file.input.path,
                                                 ctx.outputs.output.path),
    execution_requirements = { tag: '' for tag in ctx.attr.action_tags },
  )

starlark_breaks1 = rule(
  _starlark_breaks1_impl,
  attrs = {
    "input": attr.label(mandatory=True, allow_single_file=True),
    "output": attr.output(mandatory=True),
    "action_tags": attr.string_list(),
  },
)
EOF
}

function test_sandbox_undeclared_deps_starlark() {
  mkdir -p examples/genrule
  echo "foo bar bz" > examples/genrule/a.txt
  echo "apples oranges bananas" > examples/genrule/b.txt
  cat << 'EOF' > examples/genrule/BUILD
load('//examples/genrule:starlark.bzl', 'starlark_breaks1')

starlark_breaks1(
  name = "starlark_breaks1",
  input = "a.txt",
  output = "starlark_breaks1.txt",
)
EOF
  write_starlark_breaks1
  output_file="bazel-bin/examples/genrule/starlark_breaks1.txt"
  bazel build examples/genrule:starlark_breaks1 &> $TEST_log \
    && fail "Non-hermetic genrule succeeded: examples/genrule:starlark_breaks1" || true

  [ -f "$output_file" ] ||
    fail "Action did not produce output: $output_file"

  if [ $(wc -l $output_file) -gt 1 ]; then
    fail "Output contained more than one line: $output_file"
  fi

  fgrep "No such file or directory" $output_file ||
    fail "Output did not contain expected error message: $output_file"
}

function test_sandbox_undeclared_deps_starlark_with_local_tag() {
  mkdir -p examples/genrule
  echo "foo bar bz" > examples/genrule/a.txt
  echo "apples oranges bananas" > examples/genrule/b.txt
  cat << 'EOF' > examples/genrule/BUILD
load('//examples/genrule:starlark.bzl', 'starlark_breaks1')

starlark_breaks1(
  name = "starlark_breaks1",
  input = "a.txt",
  output = "starlark_breaks1.txt",
)

starlark_breaks1(
  name = "starlark_breaks1_works_with_local_tag",
  input = "a.txt",
  output = "starlark_breaks1_works_with_local_tag.txt",
  action_tags = [ "local" ],
)
EOF
  write_starlark_breaks1

  bazel build --incompatible_legacy_local_fallback \
    examples/genrule:starlark_breaks1_works_with_local_tag &> $TEST_log \
    || fail "Non-hermetic genrule failed even though tags=['local']: examples/genrule:starlark_breaks1_works_with_local_tag"
  [ -f "bazel-bin/examples/genrule/starlark_breaks1_works_with_local_tag.txt" ] \
    || fail "Action did not produce output: examples/genrule:starlark_breaks1_works_with_local_tag"
}

function test_sandbox_cyclic_symlink_in_inputs() {
  mkdir -p examples/genrule
  # Create cyclic symbolic links to check whether the strategy catches that.
  ln -sf cyclic2 examples/genrule/cyclic1
  ln -sf cyclic1 examples/genrule/cyclic2
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "breaks3",
  srcs = [ "cyclic1", "cyclic2" ],
  outs = [ "breaks3.txt" ],
  cmd = "wc $(location :cyclic1) > $@",
)
EOF
  bazel build examples/genrule:breaks3 &> $TEST_log \
    && fail "Genrule with cyclic symlinks succeeded: examples/genrule:breaks3" || true
  [ ! -f "bazel-genfiles/examples/genrule/breaks3.txt" ] || {
    output=$(cat "bazel-genfiles/examples/genrule/breaks3.txt")
    fail "Genrule with cyclic symlinks breaks3 succeeded with following output: $output"
  }
}


function test_requires_root() {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: fake usernames not supported in this system" 1>&2
    return 0
  fi

  cat > test.sh <<'EOF'
#!/bin/sh
([ $(id -u) = "0" ] && [ $(id -g) = "0" ]) || exit 1
EOF
  chmod +x test.sh
  cat > BUILD <<'EOF'
sh_test(
  name = "test",
  srcs = ["test.sh"],
  tags = ["requires-fakeroot"],
)
EOF
  bazel test --test_output=errors :test || fail "test did not pass"
  bazel test --nocache_test_results --sandbox_fake_username --test_output=errors :test || fail "test did not pass"
}

# Tests that /proc/self == /proc/$$. This should always be true unless the PID namespace is active without /proc being remounted correctly.
function test_sandbox_proc_self() {
  if [[ ! -d /proc/self ]]; then
    echo "Skipping tests: requires /proc" 1>&2
    return 0
  fi
  mkdir -p examples/genrule
  cat << 'EOF' > examples/genrule/BUILD
genrule(
  name = "check_proc_works",
  outs = [ "check_proc_works.txt" ],
  cmd = "sh -c 'cd /proc/self && echo $$$$ && exec cat stat | sed \"s/\\([^ ]*\\) .*/\\1/g\"' > $@",
)
EOF

  bazel build examples/genrule:check_proc_works >& $TEST_log || fail "build should have succeeded"

  (
    # Catch the head and tail commands failing.
    set -e
    if [[ "$(head -n1 "bazel-genfiles/examples/genrule/check_proc_works.txt")" \
          != "$(tail -n1 "bazel-genfiles/examples/genrule/check_proc_works.txt")" ]] ; then
      fail "Reading PID from /proc/self/stat should have worked, instead have these: $(cat "bazel-genfiles/examples/genrule/check_proc_works.txt")"
    fi
  )
}

function test_succeeding_action_with_ioexception_while_copying_outputs_throws_correct_exception() {
  cat > BUILD <<'EOF'
genrule(
  name = "test",
  outs = ["readonlydir/output.txt"],
  cmd = "touch $(location readonlydir/output.txt); chmod 0 $(location readonlydir/output.txt); chmod 0500 `dirname $(location readonlydir/output.txt)`",
)
EOF
  bazel build :test &> $TEST_log \
    && fail "build should have failed" || true

  # This is the generic "we caught an IOException" log message used by the
  # SandboxedStrategy. We don't want to see this in this case, because we have
  # special handling that prints a better error message and then lets the
  # sandbox code throw the actual ExecException.
  expect_not_log "I/O error during sandboxed execution"

  # There was no ExecException during sandboxed execution, because the action
  # returned an exit code of 0.
  expect_not_log "Executing genrule //:test failed: linux-sandbox failed: error executing command"

  # This is the error message telling us that some output artifacts couldn't be copied.
  expect_log "Could not move output artifacts from sandboxed execution"

  # The build fails, because the action didn't generate its output artifact.
  expect_log "ERROR:.*Executing genrule //:test failed"
}

function test_failing_action_with_ioexception_while_copying_outputs_throws_correct_exception() {
  cat > BUILD <<'EOF'
genrule(
  name = "test",
  outs = ["readonlydir/output.txt"],
  cmd = "touch $(location readonlydir/output.txt); chmod 0 $(location readonlydir/output.txt); chmod 0500 `dirname $(location readonlydir/output.txt)`; exit 1",
)
EOF
  bazel build :test &> $TEST_log \
    && fail "build should have failed" || true

  # This is the generic "we caught an IOException" log message used by the
  # SandboxedStrategy. We don't want to see this in this case, because we have
  # special handling that prints a better error message and then lets the
  # sandbox code throw the actual ExecException.
  expect_not_log "I/O error during sandboxed execution"

  # This is the error message printed by the EventHandler telling us that some
  # output artifacts couldn't be copied.
  expect_log "Could not move output artifacts from sandboxed execution"

  # This is the UserExecException telling us that the build failed.
  expect_log "Executing genrule //:test failed:"
}

function test_read_non_hermetic_tmp {
  sed -i.bak '/sandbox_tmpfs_path/d' "$bazelrc"
  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
[[ -f "${temp_dir}/file" ]]
EOF
  chmod +x pkg/tmp_test.sh

  touch "${temp_dir}/file"
  bazel test //pkg:tmp_test \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
}

function test_read_hermetic_tmp {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: --incompatible_sandbox_hermetic_tmp is only supported in Linux" 1>&2
    return 0
  fi

  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
[[ ! -f "${temp_dir}/file" ]]
EOF
  chmod +x pkg/tmp_test.sh

  touch "${temp_dir}/file"
  bazel test //pkg:tmp_test --incompatible_sandbox_hermetic_tmp \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
}

function test_read_hermetic_tmp_user_override {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: --incompatible_sandbox_hermetic_tmp is only supported in Linux" 1>&2
    return 0
  fi
  sed -i.bak '/sandbox_tmpfs_path/d' "$bazelrc"

  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
[[ -f "${temp_dir}/file" ]]
EOF
  chmod +x pkg/tmp_test.sh

  touch "${temp_dir}/file"
  bazel test //pkg:tmp_test --incompatible_sandbox_hermetic_tmp --sandbox_add_mount_pair=/tmp \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
}

function test_write_non_hermetic_tmp {
  sed -i.bak '/sandbox_tmpfs_path/d' "$bazelrc"
  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
touch "${temp_dir}/file"
EOF
  chmod +x pkg/tmp_test.sh

  bazel test //pkg:tmp_test \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
  [[ -f "${temp_dir}/file" ]] || fail "Expected ${temp_dir}/file to exist"
}

function test_write_hermetic_tmp {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: --incompatible_sandbox_hermetic_tmp is only supported in Linux" 1>&2
    return 0
  fi

  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
mkdir -p "${temp_dir}"
touch "${temp_dir}/file"
EOF
  chmod +x pkg/tmp_test.sh

  bazel test //pkg:tmp_test --incompatible_sandbox_hermetic_tmp \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
  [[ ! -f "${temp_dir}" ]] || fail "Expected ${temp_dir} to not exit"
}

function test_write_hermetic_tmp_user_override {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: --incompatible_sandbox_hermetic_tmp is only supported in Linux" 1>&2
    return 0
  fi
  sed -i.bak '/sandbox_tmpfs_path/d' "$bazelrc"

  temp_dir=$(mktemp -d /tmp/test.XXXXXX)
  trap 'rm -rf ${temp_dir}' EXIT

  mkdir -p pkg
  cat > pkg/BUILD <<'EOF'
sh_test(
  name = "tmp_test",
  srcs = ["tmp_test.sh"],
)
EOF

  cat > pkg/tmp_test.sh <<EOF
touch "${temp_dir}/file"
EOF
  chmod +x pkg/tmp_test.sh

  bazel test //pkg:tmp_test --incompatible_sandbox_hermetic_tmp --sandbox_add_mount_pair=/tmp \
    --test_output=errors &>$TEST_log || fail "Expected test to pass"
  [[ -f "${temp_dir}/file" ]] || fail "Expected ${temp_dir}/file to exist"
}

function test_sandbox_reuse_stashes_sandbox() {
  mkdir pkg
  cat >pkg/BUILD <<'EOF'
genrule(
  name = "a",
  srcs = [ "a.txt" ],
  outs = [ "aout.txt" ],
  cmd = "wc $(location :a.txt) > $@",
)
genrule(
  name = "b",
  srcs = [ "b.txt" ],
  outs = [ "bout.txt" ],
  cmd = "wc $(location :b.txt) > $@",
)
EOF
  echo A > pkg/a.txt
  echo BB > pkg/b.txt
  local output_base="$(bazel info output_base)"
  local execroot="$(bazel info execution_root)"
  local execroot_reldir="${execroot#$output_base}"

  bazel build --reuse_sandbox_directories //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"

  local sandbox_stash="${output_base}/sandbox_stash"
  [[ -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} not present"
  [[ -d "${sandbox_stash}/_NoMnemonic_/3" ]] \
    || fail "${sandbox_stash} did not stash anything"
  [[ -L "${sandbox_stash}/_NoMnemonic_/3/$execroot_reldir/pkg/a.txt" ]] \
    || fail "${sandbox_stash} did not have a link to a.txt"

  bazel build --reuse_sandbox_directories //pkg:b >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"
  ls -R "${sandbox_stash}/_NoMnemonic_/"
  [[ ! -L "${sandbox_stash}/_NoMnemonic_/6/$execroot_reldir/pkg/a.txt" ]] \
    || fail "${sandbox_stash} should no longer have a link to a.txt"
  [[ -L "${sandbox_stash}/_NoMnemonic_/6/$execroot_reldir/pkg/b.txt" ]] \
    || fail "${sandbox_stash} should now have a link to b.txt"

  bazel clean
  [[ ! -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} present after clean"

  bazel build --reuse_sandbox_directories //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"
}

function test_sandbox_reuse_stashes_sandbox_with_changing_hermetic_tmp() {
  mkdir pkg
  cat >pkg/BUILD <<'EOF'
genrule(
  name = "a",
  srcs = [ "a.txt" ],
  outs = [ "aout.txt" ],
  cmd = "wc $(location :a.txt) > $@",
)
genrule(
  name = "b",
  srcs = [ "b.txt" ],
  outs = [ "bout.txt" ],
  cmd = "wc $(location :b.txt) > $@",
)
EOF
  echo A > pkg/a.txt
  echo BB > pkg/b.txt
  local output_base="$(bazel info output_base)"
  local execroot="$(bazel info execution_root)"
  local execroot_reldir="${execroot#$output_base}"

  bazel build --reuse_sandbox_directories \
    //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"

  local sandbox_stash="${output_base}/sandbox_stash"
  [[ -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} not present"
  [[ -d "${sandbox_stash}/_NoMnemonic_/3" ]] \
    || fail "${sandbox_stash} did not stash anything"
  [[ -L "${sandbox_stash}/_NoMnemonic_/3/$execroot_reldir/pkg/a.txt" ]] \
    || fail "${sandbox_stash} did not have a link to a.txt"

  bazel build --reuse_sandbox_directories --incompatible_sandbox_hermetic_tmp \
    //pkg:b >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"
  ls -R "${sandbox_stash}/_NoMnemonic_/"
  [[ ! -L "${sandbox_stash}/_NoMnemonic_/6/$execroot_reldir/pkg/a.txt" ]] \
    || fail "${sandbox_stash} should no longer have a link to a.txt"
  [[ -L "${sandbox_stash}/_NoMnemonic_/6/$execroot_reldir/pkg/b.txt" ]] \
    || fail "${sandbox_stash} should now have a link to b.txt"

  bazel clean
  [[ ! -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} present after clean"

  bazel build --reuse_sandbox_directories --incompatible_sandbox_hermetic_tmp \
    //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"
}

function test_sandbox_reuse_clean() {
  mkdir pkg
  cat >pkg/BUILD <<'EOF'
genrule(
  name = "a",
  srcs = [ "a.txt" ],
  outs = [ "aout.txt" ],
  cmd = "wc $(location :a.txt) > $@",
)
EOF
  echo A > pkg/a.txt
  local output_base="$(bazel info output_base)"

  bazel build --reuse_sandbox_directories //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"

  local sandbox_stash="${output_base}/sandbox_stash"
  [[ -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} not present"
  [[ -d "${sandbox_stash}/_NoMnemonic_/3" ]] \
    || fail "${sandbox_stash} did not stash anything"

  bazel clean --reuse_sandbox_directories
  [[ ! -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} present after clean"

  bazel build --experimental_sandbox_async_tree_delete_idle_threads=2 \
    --reuse_sandbox_directories //pkg:a >"${TEST_log}" 2>&1 \
    || fail "Expected build to succeed"
  [[ -d "${sandbox_stash}/_NoMnemonic_/6" ]] \
    || fail "${sandbox_stash} did not stash anything"

  bazel clean
  [[ ! -d "${sandbox_stash}" ]] \
    || fail "${sandbox_stash} present after non-reusing clean"
}

run_suite "sandboxing"
