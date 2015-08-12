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
#
# Test //external mechanisms
#

# Load test environment
src=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $src/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }
source $src/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function set_up() {
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

function test_maven_jar() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
    sha1 = '$sha1',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit || fail "Fetch failed"
  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  kill_nc
  assert_contains "GET /com/example/carnivore/carnivore/1.23/carnivore-1.23.jar" $nc_log
  expect_log "Tra-la!"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit || fail "Fetch failed"
  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  kill_nc
  assert_contains "GET /com/example/carnivore/carnivore/1.23/carnivore-1.23.jar" $nc_log
  expect_log "Tra-la!"
}

function test_maven_jar_404() {
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

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency: Could not find artifact"
}

function test_maven_jar_mismatched_sha1() {
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
