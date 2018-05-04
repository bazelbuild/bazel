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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

test_result_recorded() {
  mkdir fetchrepo
  cd fetchrepo
  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.symlink(ctx.attr.build_file, "BUILD")
  return {"build_file": ctx.attr.build_file, "extra_arg": "foobar"}

trivial_rule = repository_rule(
  implementation = _rule_impl,
  attrs = { "build_file" : attr.label() },
)

EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.txt"],
  cmd = "echo bar > $@",
)
EOF
  touch BUILD
  cat  > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")
trivial_rule(
  name = "ext",
  build_file = "//:ext.BUILD",
)
EOF

  bazel clean --expunge
  bazel build --experimental_repository_resolved_file=../repo.bzl @ext//... \
      || fail "Expected success"

  # Verify that bazel can read the generated repo.bzl file and that it contains
  # the expected information
  cd ..
  mkdir analysisrepo
  mv repo.bzl analysisrepo
  cd analysisrepo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

[ genrule(
    name = "out",
    outs = ["out.txt"],
    cmd = "echo %s > $@" % entry["repositories"][0]["attributes"]["extra_arg"],
  ) for entry in resolved if entry["original_rule_class"] == "//:rule.bzl%trivial_rule"
]
EOF
  cat BUILD
  bazel build //:out || fail "Expected success"
  grep "foobar" `bazel info bazel-genfiles`/out.txt \
      || fail "Did not find the expected value"

}

run_suite "workspace_resolved_test tests"
