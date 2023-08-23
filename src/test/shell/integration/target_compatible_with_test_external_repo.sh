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
# target_compatible_with.sh variations for external repos.

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

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
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function set_up() {
  mkdir -p target_skipping || fail "couldn't create directory"
  cat > target_skipping/pass.sh <<EOF || fail "couldn't create pass.sh"
#!/bin/bash
exit 0
EOF
  chmod +x target_skipping/pass.sh

  cat > target_skipping/fail.sh <<EOF|| fail "couldn't create fail.sh"
#!/bin/bash
exit 1
EOF
  chmod +x target_skipping/fail.sh

  cat > target_skipping/BUILD <<EOF || fail "couldn't create BUILD file"
# We're not validating visibility here. Let everything access these targets.
package(default_visibility = ["//visibility:public"])

constraint_setting(name = "foo_version")

constraint_value(
    name = "foo1",
    constraint_setting = ":foo_version",
)

constraint_value(
    name = "foo2",
    constraint_setting = ":foo_version",
)

constraint_value(
    name = "foo3",
    constraint_setting = ":foo_version",
)

constraint_setting(name = "bar_version")

constraint_value(
    name = "bar1",
    constraint_setting = "bar_version",
)

constraint_value(
    name = "bar2",
    constraint_setting = "bar_version",
)

platform(
    name = "foo1_bar1_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo1",
        ":bar1",
    ],
)

platform(
    name = "foo2_bar1_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo2",
        ":bar1",
    ],
)

platform(
    name = "foo1_bar2_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo1",
        ":bar2",
    ],
)

platform(
    name = "foo3_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo3",
    ],
)

sh_test(
    name = "pass_on_foo1",
    srcs = ["pass.sh"],
    target_compatible_with = [":foo1"],
)

sh_test(
    name = "fail_on_foo2",
    srcs = ["fail.sh"],
    target_compatible_with = [":foo2"],
)

sh_test(
    name = "pass_on_foo1_bar2",
    srcs = ["pass.sh"],
    target_compatible_with = [
        ":foo1",
        ":bar2",
    ],
)

sh_binary(
    name = "some_foo3_target",
    srcs = ["pass.sh"],
    target_compatible_with = [
        ":foo3",
    ],
)
EOF

  cat > target_skipping/WORKSPACE <<EOF || fail "couldn't create WORKSPACE"
EOF
}

add_to_bazelrc "test --nocache_test_results"
add_to_bazelrc "build --incompatible_merge_genfiles_directory=true"

function tear_down() {
  bazel shutdown
}

function test_failure_on_incompatible_top_level_target_in_external_repo() {
  cat >> target_skipping/WORKSPACE <<EOF
local_repository(
    name = "test_repo",
    path = "third_party/test_repo",
)
EOF
  mkdir -p target_skipping/third_party/test_repo/
  touch target_skipping/third_party/test_repo/WORKSPACE
  cat > target_skipping/third_party/test_repo/BUILD <<EOF
cc_binary(
    name = "bin",
    srcs = ["bin.cc"],
    target_compatible_with = [
        "@//:foo1",
    ],
)
EOF
  cat > target_skipping/third_party/test_repo/bin.cc <<EOF
int main() {
    return 0;
}
EOF
  cd target_skipping || fail "couldn't cd into workspace"
  bazel test \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    --build_event_text_file="${TEST_log}".build.json \
    @test_repo//:bin &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR:.*Target @test_repo//:bin is incompatible and cannot be built'
  expect_log '^ERROR: Build did NOT complete successfully'
  # Now look at the build event log.
  mv "${TEST_log}".build.json "${TEST_log}"
  expect_log '^    name: "PARSING_FAILURE"$'
  expect_log 'Target @test_repo//:bin is incompatible and cannot be built.'
}

# Regression test for https://github.com/bazelbuild/bazel/issues/12374
function test_repository_defines_target_compatible_with() {
  cat > repo.bzl <<EOF
def _repo_rule_impl(repo_ctx):
    pass

repo_rule = repository_rule(
    implementation = _repo_rule_impl,
    attrs = {
        "target_compatible_with": attr.label_list(),
    },
)
EOF

  cat >> WORKSPACE <<EOF
load(':repo.bzl', 'repo_rule')
repo_rule(name = 'defines_tcw')
EOF

  cat > BUILD <<EOF
filegroup(name = "empty")
EOF

  bazel build //:empty &> "${TEST_log}" || fail "Bazel failed."
  expect_not_log "Error in repository_rule: There is already a built-in attribute 'target_compatible_with' which cannot be overridden"
}

run_suite "target_compatible_with_external_repo tests"
