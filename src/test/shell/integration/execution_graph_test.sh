#!/usr/bin/env bash
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
# execution_graph_test.sh: integration tests for the executed graph log.
#

# --- begin runfiles.bash initialization ---
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

add_to_bazelrc "test --test_strategy=standalone"
add_to_bazelrc "test --strategy=TestRunner=local"

# Location of the proto to text helper binary
EXECUTION_GRAPH_MAIN="$(rlocation "io_bazel/src/test/java/com/google/devtools/build/lib/runtime/execution_graph_proto_to_text${EXE_EXT}")"

OUTPUT_PATH="$TEST_TMPDIR/execution_graph_dump.proto.zst"

function set_up() {
  cd "${WORKSPACE_DIR}"
  rm -rf foo subdir
  add_rules_shell "MODULE.bazel"
}

function dump_execution_graph() {
  if is_windows; then
    "$EXECUTION_GRAPH_MAIN" "$OUTPUT_PATH" > "$TEST_log" \
        || fail "Text conversion failed"
  else
    JAVA_RUNFILES="$TEST_SRCDIR" "$EXECUTION_GRAPH_MAIN" "$OUTPUT_PATH" \
        > "$TEST_log" || fail "Text conversion failed"
  fi
}

function test_simple_sh_test() {
  if is_windows; then
    # Windows uses launcher executables and disables runfile symlink trees by default.
    return 0
  fi
  mkdir -p foo || fail "mkdir foo failed"
  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "shell_test", srcs = ["shell_test.sh"])
EOF
  cat > foo/shell_test.sh <<'EOF'
#!/bin/bash
exit 0
EOF
  chmod a+x foo/shell_test.sh

  rm -f "$OUTPUT_PATH"
  bazel clean || fail "Cleaning failed"
  bazel test --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="$OUTPUT_PATH" //foo:shell_test \
      > "$TEST_log" || fail "Build failed"
  dump_execution_graph
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    # Includes RepoMappingManifest under Bzlmod.
    expect_log_n "metrics {" 8
  else
    expect_log_n "metrics {" 7
  fi
  expect_not_log "index"
}

function test_sh_test_with_genrule_dep() {
  if is_windows; then
    # Windows uses launcher executables and disables runfile symlink trees by default.
    return 0
  fi
  mkdir -p foo || fail "mkdir foo failed"
  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "shell_test", srcs = ["shell_test.sh"])
genrule(
    name = "gen",
    outs = ["shell_test.sh"],
    executable = 1,
    cmd = "echo #!/bin/bash\nexit 0 > $@"
)
EOF

  rm -f "$OUTPUT_PATH"
  bazel clean || fail "Cleaning failed"
  bazel test --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="$OUTPUT_PATH" \
      --experimental_execution_graph_log_dep_type=all //foo:shell_test \
      > "$TEST_log" || fail "Build failed"
  dump_execution_graph
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    # Includes RepoMappingManifest and its dependency edge on TestRunner.
    expect_log_n "metrics {" 9
    expect_log_n "dependent_index:" 9
  else
    expect_log_n "metrics {" 8
    expect_log_n "dependent_index:" 8
  fi
  expect_log_n "mnemonic: \".*Symlink.*\"" 3
  expect_log_n "mnemonic: \"Genrule\"" 1
  expect_log_n "mnemonic: \"TestRunner\"" 2
  expect_log_n "^index: 6$" 1
}

function test_flaky_sh_test() {
  if is_windows; then
    # Windows uses launcher executables and disables runfile symlink trees by default.
    return 0
  fi
  mkdir -p foo || fail "mkdir foo failed"
  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "shell_test", srcs = ["shell_test.sh"], flaky = 1)
EOF
  cat > foo/shell_test.sh <<'EOF'
#!/bin/bash
exit 1
EOF
  chmod a+x foo/shell_test.sh

  rm -f "$OUTPUT_PATH"
  bazel clean || fail "Cleaning failed"
  bazel test --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="$OUTPUT_PATH" \
      --experimental_execution_graph_log_dep_type=all //foo:shell_test \
      > "$TEST_log" || true
  dump_execution_graph
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    # RepoMappingManifest adds 1 node and shifts TestRunner indices by +1.
    expect_log_n "metrics {" 12
    expect_log_n "retry_of: 6" 1
    expect_log_n "retry_of: 7" 1
  else
    expect_log_n "metrics {" 11
    expect_log_n "retry_of: 5" 1
    expect_log_n "retry_of: 6" 1
  fi
  expect_log_n "mnemonic: \".*Symlink.*\"" 3
  expect_log_n "mnemonic: \"TestRunner\"" 6
  expect_log_n "mnemonic: \"RunfilesTree\"" 1
}

function test_shared_actions() {
  mkdir -p foo || fail "mkdir foo failed"
  cat > foo/defs.bzl <<'EOF'
def _shared_impl(ctx):
  f = ctx.actions.declare_file("shared")
  ctx.actions.run_shell(outputs = [f], command = "touch %s" % f.path)
  return DefaultInfo(files = depset([f]))

shared = rule(implementation = _shared_impl)
EOF
  cat > foo/BUILD <<'EOF'
load(":defs.bzl", "shared")
shared(name = "a")
shared(name = "b")
genrule(name = "gen_a", srcs = [":a"], outs = ["gen_a.out"], cmd = "cp $< $@")
genrule(name = "gen_b", srcs = [":b"], outs = ["gen_b.out"], cmd = "cp $< $@")
EOF

  rm -f "$OUTPUT_PATH"
  bazel clean || fail "Cleaning failed"
  bazel build //foo:all \
      --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="$OUTPUT_PATH" \
      --experimental_execution_graph_log_dep_type=all \
      >& "$TEST_log" || fail "Build failed"
  dump_execution_graph

  # 4 nodes: build-info, (a|b), gen_a, gen_b
  expect_log_n "metrics {" 4
  # 1 shared node for the shared action.
  expect_log_n "description: .*Action foo/shared.*" 1
  # Both gen_a and gen_b have edges to the shared node.
  expect_log_n "dependent_index:" 2
}

function test_relative_and_workspace_log_path() {
  mkdir -p foo || fail "mkdir foo failed"
  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "shell_test", srcs = ["shell_test.sh"])
EOF
  cat > foo/shell_test.sh <<'EOF'
#!/bin/bash
exit 0
EOF
  chmod a+x foo/shell_test.sh

  mkdir -p subdir && cd subdir || fail "cd subdir failed"
  rm -f relative_graph.log
  bazel test --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="relative_graph.log" //foo:shell_test \
      > "$TEST_log" || fail "Build failed"
  [[ -f relative_graph.log ]] || fail "relative_graph.log was not created in working directory"

  rm -f ../workspace_graph.log
  bazel test --experimental_enable_execution_graph_log \
      --experimental_execution_graph_log_path="%workspace%/workspace_graph.log" //foo:shell_test \
      > "$TEST_log" || fail "Build failed"
  [[ -f ../workspace_graph.log ]] || fail "workspace_graph.log was not created in workspace root"
  cd ..
}

run_suite "Integration tests for the execution graph."
