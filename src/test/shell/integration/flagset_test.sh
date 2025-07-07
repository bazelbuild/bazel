#!/usr/bin/env bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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
# An end-to-end test for flagsets.

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

write_project_scl_definition

function set_up_project_file() {
  mkdir -p test
  cat > test/BUILD <<EOF
load("//test:test.bzl", "string_flag")

genrule(name='test', outs=['test.txt'], cmd='echo hi > \$@')

string_flag(
    name = "starlark_flags_always_affect_configuration",
    build_setting_default = "default",
)
EOF
  cat > test/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  enforcement_policy = "strict",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF

  touch test/test.bzl
  cat > test/test.bzl <<EOF
string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
EOF
}

function test_scl_config_plus_user_bazelrc_fails(){
  set_up_project_file
  add_to_bazelrc "build '--//test:starlark_flags_always_affect_configuration=yes'"
  add_to_bazelrc "build --define=bar=baz"
  cat .bazelrc >> test/test.bazelrc
  bazel --bazelrc=test/test.bazelrc build --nobuild //test:test --enforce_project_configs --scl_config=test_config --experimental_enable_scl_dialect &> "$TEST_log" && \
    fail "Scl enabled build expected to fail with starlark flag in user bazelrc"
  expect_log "does not allow output-affecting flags in the command line or user bazelrc"
  expect_log "--//test:starlark_flags_always_affect_configuration=yes"
  expect_log "--define=bar=baz"
}

function test_scl_config_plus_command_line_flag_fails(){
  set_up_project_file
  bazel build --nobuild //test:test --enforce_project_configs --scl_config=test_config --experimental_enable_scl_dialect --//test:starlark_flags_always_affect_configuration=yes --define=bar=baz &> "$TEST_log" && \
    fail "Scl enabled build expected to fail with command-line flags"
  expect_log "does not allow output-affecting flags in the command line or user bazelrc"
  expect_log "--//test:starlark_flags_always_affect_configuration=yes"
  expect_log "--define=bar=baz"
}

function test_scl_config_plus_expanded_command_line_flag_fails(){
  set_up_project_file
  bazel build --nobuild //test:test --enforce_project_configs --scl_config=test_config --experimental_enable_scl_dialect -c opt &> "$TEST_log" && \
    fail "Scl enabled build expected to fail with command line flag"
  expect_log "does not allow output-affecting flags in the command line or user bazelrc"
  expect_log "--compilation_mode=opt"
}


function test_scl_config_plus_test_suite_tests_outside_project_passes(){
  add_rules_shell "MODULE.bazel"
  mkdir -p test
  # Make the project file warn mode so we don't fail due to our fake global rc
  # file in tests
  cat > test/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  enforcement_policy = "warn",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "default",
          flags = [],
          is_default = True,
      )
  ],
)
EOF
  cat >> test/BUILD <<EOF
test_suite(name='test_suite', tests=['//other:other'])
EOF

    mkdir -p other
  cat > other/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(name='other', srcs=['other.sh'])
EOF

  touch other/other.sh
  cat > other/other.sh <<EOF
#!/usr/bin/env bash
echo hi
EOF

  bazel build --nobuild  //test:test_suite  &> "$TEST_log" || \
    fail "expected success"
}

function test_scl_config_plus_external_target_in_test_suite_fails() {
  add_rules_shell "MODULE.bazel"
  mkdir -p test
  # This failure kicks in as soon as there's a valid project file, even if it
  # doesn't contain any configs.
  cat > test/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat >> test/BUILD <<EOF
test_suite(name='test_suite', tests=['//other:other'])
EOF

    mkdir -p other
  cat > other/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name='other', srcs=['other.sh'])
EOF

  touch other/other.sh
  cat > other/other.sh <<EOF
#!/usr/bin/env bash
echo hi
EOF

  bazel build --nobuild //test:test_suite //other:other --scl_config=test_config \
    &> "$TEST_log" && fail "expected build to fail"

  expect_log "Can't set --scl_config for a build where only some targets have projects."
}

function test_multi_project_builds_fail_with_scl_config(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  enforcement_policy = "strict",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p test2
  cat > test2/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  enforcement_policy = "strict",
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test2/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //test2:h --scl_config=test_config \
    &> "$TEST_log" && fail "expected build to fail"

  expect_log "Can't set --scl_config for a multi-project build."
}

function test_multi_project_builds_succeed_with_consistent_default_config(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p test2
  cat > test2/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test2/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //test2:h  \
    &> "$TEST_log" || fail "expected success"
}

function test_multi_project_builds_succeed_with_no_defined_configs(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create()
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p test2
  cat > test2/PROJECT.scl <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create()
EOF
  cat > test2/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //test2:h  \
    &> "$TEST_log" || fail "expected success"
}

function test_multi_project_builds_fail_with_inconsistent_default_configs(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p test2
  cat > test2/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=baz"],
          is_default = True,
      )
  ],
)
EOF
  cat > test2/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //test2:h \
    &> "$TEST_log" && fail "expected build to fail"

  expect_log "Mismatching default configs for a multi-project build."
}

function test_partial_project_builds_fail_with_non_noop_default_config(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = ["--define=foo=bar"],
          is_default = True,
      )
  ],
)
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p noproject
  cat > noproject/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //noproject:h \
    &> "$TEST_log" && fail "expected build to fail"

  expect_log "Mismatching default configs for a build where only some targets have projects."
}


function test_partial_project_cquery_paths_succeed_with_non_noop_default_config(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
project = {
  "configs": {
    "test_config": ["--define=foo=bar"],
  },
  "default_config" : "test_config"
}
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p noproject
  cat > noproject/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel clean --expunge

  bazel cquery 'somepath(//test1:g, //noproject:h)' \
    &> "$TEST_log" || fail "expected build to succeed"

  # Confirm that we're applying the flags for the first target
  expect_log "--define=foo=bar"

  bazel cquery 'somepath(//noproject:h, //test1:g)' \
    &> "$TEST_log" || fail "expected build to succeed"

  bazel cquery 'allpaths(//test1:g, //noproject:h)' \
    &> "$TEST_log" || fail "expected build to succeed"

  bazel cquery 'allpaths(//noproject:h, //test1:g)' \
    &> "$TEST_log" || fail "expected build to succeed"

  expect_not_log "Mismatching default configs for a build where only some targets have projects."
}

function test_partial_project_builds_succeed_with_noop_default_config(){
  mkdir -p test1
  cat > test1/PROJECT.scl <<EOF
load(
  "//third_party/bazel/src/main/protobuf/project:project_proto.scl",
  "buildable_unit_pb2",
  "project_pb2",
)
project = project_pb2.Project.create(
  buildable_units = [
      buildable_unit_pb2.BuildableUnit.create(
          name = "test_config",
          flags = [],
          is_default = True,
      )
  ],
)
EOF
  cat > test1/BUILD <<EOF
genrule(name='g', outs=['g.txt'], cmd='echo hi > \$@')
EOF

  mkdir -p noproject
  cat > noproject/BUILD <<EOF
genrule(name='h', outs=['h.txt'], cmd='echo hi > \$@')
EOF

  bazel build --nobuild //test1:g //noproject:h \
    &> "$TEST_log" || fail "expected success"
}

run_suite "Integration tests for flagsets/scl_config"
