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
# Test //external mechanisms
#

# Load test environment
src=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
source $src/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }
source $src/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function setup_zoo() {
  mkdir -p zoo
  cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["//external:mongoose"],
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

function tear_down() {
  shutdown_server
}

function test_maven_jar() {
  setup_zoo
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$fileserver_port/',
    sha1 = '$sha1',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1() {
  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$fileserver_port/',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function test_maven_jar_404() {
  setup_zoo
  http_response=$TEST_TMPDIR/http_response
  cat > $http_response <<EOF
HTTP/1.0 404 Not Found

EOF
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  nc_l $nc_port < $http_response &
  nc_pid=$!
  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel clean --expunge
  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency: Could not find artifact"
}

function test_maven_jar_mismatched_sha1() {
  setup_zoo
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
    sha1 = '$sha256',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "has SHA-1 of $sha1, does not match expected SHA-1 ($sha256)"
}

function test_default_repository() {
  serve_artifact thing amabop 1.9
  cat > WORKSPACE <<EOF
maven_server(
    name = "default",
    url = "http://localhost:$fileserver_port/",
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
    url = "http://localhost:$fileserver_port/",
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
    url = "http://localhost:12345/",
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
    url = "http://localhost:$fileserver_port/",
    settings_file = "settings.xml",
)
maven_jar(
    name = "good_auth",
    artifact = "thing:amabop:1.9",
    server = "x",
)

maven_server(
    name = "y",
    url = "http://localhost:$fileserver_port/",
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
