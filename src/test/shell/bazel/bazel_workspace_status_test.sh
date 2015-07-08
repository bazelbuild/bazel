#!/bin/bash
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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function test_workspace_status_invalidation() {
  create_new_workspace

  local ok=$TEST_TMPDIR/ok.sh
  local bad=$TEST_TMPDIR/bad.sh
  cat > $ok <<EOF
#!/bin/bash
exit 0
EOF
  cat >$bad <<EOF
#!/bin/bash
exit 1
EOF
  chmod +x $ok $bad

  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(name="a", srcs=[], outs=["a.out"], stamp=1, cmd="touch $@")
EOF

  bazel build --stamp //a --workspace_status_command=$bad \
    && fail "build succeeded"
  bazel build --stamp //a --workspace_status_command=$ok \
    || fail "build failed"
}

run_suite "workspace status tests"
