#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# Tests ninja_build build rule.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function setup_basic_ninja_build() {
  cat > WORKSPACE <<'EOF'
toplevel_output_directories(paths = ["out"])
EOF

mkdir -p test
  cat > test/BUILD <<'EOF'

ninja_graph(
    name = "ninjagraph",
    main = "build.ninja",
    output_root = "out",
)

ninja_build(
    name = "ninjabuild",
    ninja_graph = "ninjagraph",
    output_groups = {
        "out" : ["out/test/output.txt"],
    }
)
EOF

  cat > test/build.ninja <<'EOF'
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/output.txt: cattool test/cattool.sh
EOF

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/{one,two} > $OUTPUT

echo "$1 : test/one test/two" > $DEPFILE
EOF
  chmod +x test/cattool.sh
  printf "a" > test/one
  echo "b" > test/two

  bazel clean
}

# Test that the depfile specified in the ninja file is used to determine
# action inputs (and thus ninja actions are incrementally correct with
# respect to discovered dependencies).
function test_basic_depfile_processing() {
  setup_basic_ninja_build

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"

  printf "HELLO" > test/one

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "HELLOb"
}

function test_null_build() {
  setup_basic_ninja_build

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 1 process"

  # Verify null build with hot server.
  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 0 processes."

  bazel shutdown

  # Verify null build even after restart.
  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 0 processes."
}

# Tests that newly discovered dependencies cause a rebuild after restart.
function test_rebuild_discovered_deps_after_restart() {
  setup_basic_ninja_build

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"

  bazel shutdown
  printf "HELLO" > test/one

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "HELLOb"
}

# Tests that newly discovered dependencies cause a rebuild after restart.
function test_depfile_existing_dependencies() {
  setup_basic_ninja_build

  # Override ninja file to make test/one and test/two explicit dependencies.
  cat > test/build.ninja <<'EOF'
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/output.txt: cattool test/cattool.sh test/one test/two
EOF

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"

  printf "HELLO" > test/one

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "HELLOb"
}

# Tests Bazel behaves appropriately when depfile output is malformed.
function test_depfile_junk() {
  setup_basic_ninja_build
  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/{one,two} > $OUTPUT

echo "Haha this depfile is a bunch of garbage" > $DEPFILE
EOF

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"
}

# Tests Bazel behaves appropriately when depfiles contain invalid files.
function test_depfile_invalid_file() {
  setup_basic_ninja_build
  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/{one,two} > $OUTPUT

echo "$1 : test/one test/two test/doesntexist" > $DEPFILE
EOF

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"
}

# Tests build when depfiles contain generated (declared) inputs.
function test_depfile_generated_inputs() {
  setup_basic_ninja_build

  cat > test/build.ninja <<'EOF'
rule filecopy
  command = cat ${in} > ${out}
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/generated: filecopy test/two
build out/test/output.txt: cattool test/cattool.sh out/test/generated
EOF

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one out/test/generated  > $OUTPUT

echo "$1 : test/one out/test/generated" > $DEPFILE
EOF

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"

  printf "z" > test/two

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "az"
}

# Tests error of build when depfiles contain generated (undeclared) inputs.
function test_depfile_undeclared_generated_inputs() {
  setup_basic_ninja_build

  cat > test/build.ninja <<'EOF'
rule filecopy
  command = cat ${in} > ${out}
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/generated: filecopy test/two
build out/test/output.txt: cattool test/cattool.sh test/one
EOF

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one out/test/generated  > $OUTPUT

echo "$1 : test/one out/test/generated" > $DEPFILE
EOF

  ! bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "build should have failed"

  expect_log "depfile-declared dependency 'out/test/generated' is invalid"
}


# Tests Bazel behaves appropriately when depfiles aren't generated.
function test_depfile_not_generated() {
  setup_basic_ninja_build
  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}

cat test/{one,two} > $OUTPUT
EOF

  ! bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "build should have failed"

  expect_log "out/test/depfile.d (No such file or directory)"
}

function test_depfile_pruned_generated_input() {
  setup_basic_ninja_build

  cat > test/build.ninja <<'EOF'
rule filecopy
  command = cat ${in} > ${out}
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/generated: filecopy test/two
build out/test/output.txt: cattool test/cattool.sh test/one out/test/generated
EOF
  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one  > $OUTPUT

echo "$1 : test/cattool.sh test/one" > $DEPFILE
EOF

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 2 processes"

  # test/two is a dependency of out/test/generated, which was an originally
  # declared input, but not according to the depfile.

  echo "z" > test/two
  # Verify the root action is not run, as test/two is not an input to the action.
  # Note that ideally this build would be a null build, as the root action
  # should no longer depend on the filecopy action. However, Skyframe currently
  # does not support action dependency pruning. Thus, the only savings is
  # an action cache hit on the root action.
  bazel build -s //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 1 process"

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one out/test/generated > $OUTPUT

echo "$1 : test/cattool.sh test/one out/test/generated" > $DEPFILE
EOF

  # Build should re-execute, as cattool.sh has changed.
  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"
  expect_log "INFO: 1 process"

  # test/two should again be reflected as an input to the build via
  # the inclusion of out/test/generated in the depfile.
  echo "x" > test/two
  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "build should have failed"
  expect_log "INFO: 2 processes"
}

function test_external_source_dependency() {
  setup_basic_ninja_build

  cat > BUILD <<'EOF'
ninja_graph(
    name = "rootgraph",
    main = "build.ninja",
    output_root = "out",
)

ninja_build(
    name = "rootbuild",
    ninja_graph = "rootgraph",
    output_groups = {
        "out" : ["out/test/output.txt"],
    }
)
EOF

  cat > build.ninja <<'EOF'
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/output.txt: cattool test/cattool.sh test/one external/two
EOF

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one external/two > $OUTPUT

echo "$1 : test/one external/two" > $DEPFILE
EOF
  mkdir -p external
  echo "b" > external/two

  bazel build //:rootbuild --experimental_sibling_repository_layout --experimental_disable_external_package --experimental_ninja_actions &> $TEST_log \
      || fail "build should have succeeded"
  cat bazel-workspace/out/test/output.txt > $TEST_log
  expect_log "ab"

  echo "z" > external/two

  bazel build //:rootbuild --experimental_sibling_repository_layout --experimental_disable_external_package --experimental_ninja_actions &> $TEST_log \
      || fail "build should have succeeded"
  cat bazel-workspace/out/test/output.txt > $TEST_log
  expect_log "az"
}

# Tests a dependency on external package sources in the scenario where the
# ninja_build target has a dependency on //external:cc_toolchain. This is a
# regression test for a bug in which this dependency interfered with external
# source resolution.
function test_external_dependency_cctoolchain() {
  setup_basic_ninja_build

  cat > BUILD <<'EOF'
ninja_graph(
    name = "rootgraph",
    main = "build.ninja",
    output_root = "out",
)

genrule(
    name = "g",
    cmd = "echo 'c' > $@",
    outs = ["generated.txt"],
    # This ensures rootbuild depends on cc_toolchain transitively.
    srcs = ["//external:cc_toolchain"],
)

ninja_build(
    name = "rootbuild",
    ninja_graph = "rootgraph",
    output_groups = {
        "out" : ["out/test/output.txt"],
    },
    deps_mapping = {
        "out/dummy" : ":g",
    },
)
EOF

  cat > build.ninja <<'EOF'
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/output.txt: cattool test/cattool.sh test/one
EOF

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat test/one external/foo/two > $OUTPUT

echo "$1 : test/one external/foo/two" > $DEPFILE
EOF
  mkdir -p external/foo
  touch external/foo/BUILD
  echo "b" > external/foo/two

  bazel build //:rootbuild --experimental_sibling_repository_layout --experimental_disable_external_package --experimental_ninja_actions &> $TEST_log \
      || fail "build should have succeeded"
  cat bazel-workspace/out/test/output.txt > $TEST_log
  expect_log "ab"

  echo "z" > external/foo/two

  bazel build //:rootbuild --experimental_sibling_repository_layout --experimental_disable_external_package --experimental_ninja_actions &> $TEST_log \
      || fail "build should have succeeded"
  cat bazel-workspace/out/test/output.txt > $TEST_log
  expect_log "az"
}

function test_basic_depfile_processing() {
  setup_basic_ninja_build

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "ab"

  printf "HELLO" > test/one

  bazel build //test:ninjabuild --experimental_ninja_actions &> $TEST_log \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> $TEST_log
  expect_log "HELLOb"
}

function test_depfile_undeclared_generated_inputs() {
  setup_basic_ninja_build

  cat > test/build.ninja <<'EOF'
rule filecopy_with_side_effect
  command = test/filecopy_with_side_effect.sh ${in} ${out}
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/generated: filecopy_with_side_effect test/one
build out/test/output.txt: cattool test/cattool.sh out/test/generated
EOF

  cat > test/filecopy_with_side_effect.sh <<'EOF'
cat $1 > $2
cat $1 > out/side_effect
EOF
  chmod +x test/filecopy_with_side_effect.sh

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat out/test/generated out/side_effect > "$OUTPUT"

echo "$1 : out/test/generated out/side_effect" > "$DEPFILE"
EOF

  printf "HELLO" > test/one
  bazel build //test:ninjabuild --experimental_ninja_actions &> "$TEST_log" \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> "$TEST_log"
  expect_log "HELLOHELLO"

  printf "GOODBYE" > test/one

  bazel build //test:ninjabuild --experimental_ninja_actions &> "$TEST_log" \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> "$TEST_log"
  expect_log "GOODBYEGOODBYE"
}

function test_depfile_modify_side_effect_file() {
  setup_basic_ninja_build

  cat > test/build.ninja <<'EOF'
rule filecopy_with_side_effect
  command = test/filecopy_with_side_effect.sh ${in} ${out}
rule cattool
  depfile = out/test/depfile.d
  command = ${in} ${out}
build out/test/generated: filecopy_with_side_effect test/one
build out/test/output.txt: cattool test/cattool.sh out/test/generated
EOF

  cat > test/filecopy_with_side_effect.sh <<'EOF'
cat $1 > $2
cat $1 > out/side_effect
EOF
  chmod +x test/filecopy_with_side_effect.sh

  cat > test/cattool.sh <<'EOF'
OUTPUT=${!#}
DEPFILE="$(dirname $OUTPUT)/depfile.d"

cat out/test/generated out/side_effect > "$OUTPUT"

echo "$1 : out/test/generated out/side_effect" > "$DEPFILE"
EOF

  printf "HELLO" > test/one
  bazel build //test:ninjabuild --experimental_ninja_actions &> "$TEST_log" \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> "$TEST_log"
  expect_log "HELLOHELLO"

  printf "GOODBYE" > bazel-workspace/out/side_effect

  bazel build //test:ninjabuild --experimental_ninja_actions &> "$TEST_log" \
      || fail "should have generated output successfully"

  cat bazel-workspace/out/test/output.txt &> "$TEST_log"
  # This verifies that changing out/side_effect retriggers the action.
  # ("HELLO" is from out/test/generated, "GOODBYE" is from out/side_effect)
  expect_log "HELLOGOODBYE"
}

run_suite "ninja_build rule tests"
