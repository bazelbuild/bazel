#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Test //external mechanisms
#

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function tear_down() {
  shutdown_server
}

function setup_zoo() {
  mkdir -p zoo
  cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["@endangered//jar"],
)
EOF

  cat > zoo/BallPit.java <<EOF
import carnivore.Mongoose;

public class BallPit {
    public static void main(String args[]) {
        Mongoose.frolic();
    }
}
EOF
}

function test_maven_jar() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$fileserver_port/',
    sha1 = '$sha1',
    sha1_src = '$sha1_src',
)
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function test_maven_jar_no_sha1_src() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$fileserver_port/',
    sha1 = '$sha1',
)
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$fileserver_port/',
)
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

# makes sure both jar and srcjar are downloaded
function test_maven_jar_downloads() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$fileserver_port/',
)
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  output_base="$(bazel info output_base)"
  test -e "${output_base}/external/endangered/jar/carnivore-1.23.jar" \
    || fail "jar not downloaded to expected place"
  test -e "${output_base}/external/endangered/jar/carnivore-1.23-sources.jar" \
    || fail "srcjar not downloaded to expected place"
}

function test_maven_jar_404() {
  setup_zoo
  serve_not_found

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$nc_port/',
)
EOF

  bazel clean --expunge
  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency: Could not find artifact"
}

function test_maven_jar_mismatched_sha1() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  wrong_sha1="0123456789012345678901234567890123456789"
  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$fileserver_port/',
    sha1 = '$wrong_sha1',
)
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  expect_log "has SHA-1 of $sha1, does not match expected SHA-1 ($wrong_sha1)"
}

function test_default_repository() {
  serve_artifact thing amabop 1.9
  cat > WORKSPACE <<EOF
maven_server(
    name = "default",
    url = "http://127.0.0.1:$fileserver_port/",
)

maven_jar(
    name = "thing_a_ma_bop",
    artifact = "thing:amabop:1.9",
)
EOF

  bazel build @thing_a_ma_bop//jar &> $TEST_log || fail "Building thing failed"
  expect_log "Target @thing_a_ma_bop//jar:jar up-to-date"
}

function test_settings() {
  serve_artifact thing amabop 1.9
  cat > WORKSPACE <<EOF
maven_server(
    name = "x",
    url = "http://127.0.0.1:$fileserver_port/",
    settings_file = "settings.xml",
)
maven_jar(
    name = "thing_a_ma_bop",
    artifact = "thing:amabop:1.9",
    server = "x",
)
EOF

  cat > settings.xml <<EOF
<settings>
  <servers>
    <server>
      <id>default</id>
    </server>
  </servers>
</settings>
EOF

  bazel build @thing_a_ma_bop//jar &> $TEST_log \
    || fail "Building thing failed"
  expect_log "Target @thing_a_ma_bop//jar:jar up-to-date"

  # Create an invalid settings.xml (by using a tag that isn't allowed in
  # settings).
  cat > settings.xml <<EOF
<settings>
  <repositories>
    <repository>
      <id>default</id>
    </repository>
  </repositories>
</settings>
EOF
  bazel clean --expunge
  bazel build @thing_a_ma_bop//jar &> $TEST_log \
    && fail "Building thing succeeded"
  expect_log "Unrecognised tag: 'repositories'"
}

function test_maven_server_dep() {
  cat > WORKSPACE <<EOF
maven_server(
    name = "x",
    url = "http://127.0.0.1:12345/",
)
EOF

  cat > BUILD <<EOF
sh_binary(
    name = "y",
    srcs = ["y.sh"],
    deps = ["@x//:bar"],
)
EOF

  touch y.sh
  chmod +x y.sh

  bazel build //:y &> $TEST_log && fail "Building thing failed"
  expect_log "does not represent an actual repository"
}

function test_auth() {
  startup_auth_server
  create_artifact thing amabop 1.9
  cat > WORKSPACE <<EOF
maven_server(
    name = "x",
    url = "http://127.0.0.1:$fileserver_port/",
    settings_file = "settings.xml",
)
maven_jar(
    name = "good_auth",
    artifact = "thing:amabop:1.9",
    server = "x",
)

maven_server(
    name = "y",
    url = "http://127.0.0.1:$fileserver_port/",
    settings_file = "settings.xml",
)
maven_jar(
    name = "bad_auth",
    artifact = "thing:amabop:1.9",
    server = "y",
)
EOF

  cat > settings.xml <<EOF
<settings>
  <servers>
    <server>
      <id>x</id>
      <username>foo</username>
      <password>bar</password>
    </server>
    <server>
      <id>y</id>
      <username>foo</username>
      <password>baz</password>
    </server>
  </servers>
</settings>
EOF

  bazel build @good_auth//jar &> $TEST_log \
    || fail "Expected correct password to work"
  expect_log "Target @good_auth//jar:jar up-to-date"

  bazel build @bad_auth//jar &> $TEST_log \
    && fail "Expected incorrect password to fail"
  expect_log "Unauthorized (401)"
}

run_suite "maven tests"
