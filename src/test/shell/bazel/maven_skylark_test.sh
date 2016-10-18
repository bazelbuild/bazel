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
# Test the Skylark implementation of the maven_jar() rule.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/remote_helpers.sh \
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

function test_maven_jar_skylark() {
  setup_zoo
  version="1.21"
  serve_artifact com.example.carnivore carnivore $version

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    repository = 'http://localhost:$fileserver_port/',
    sha1 = '$sha1',
    local_repository = "@m2//:BUILD",
)

# Make use of the pre-downloaded maven-dependency-plugin because there's no
# internet connection at this stage.
local_repository(
  name = "m2",
  path = "$TEST_SRCDIR/m2",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1_skylark() {
  setup_zoo
  version="1.22"
  serve_artifact com.example.carnivore carnivore $version

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    repository = 'http://localhost:$fileserver_port/',
    local_repository = "@m2//:BUILD",
)

local_repository(
  name = "m2",
  path = "$TEST_SRCDIR/m2",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  expect_log "Tra-la!"
}

function test_maven_jar_404_skylark() {
  setup_zoo
  version="1.23"
  serve_not_found

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    repository = 'http://localhost:$nc_port/',
    local_repository = "@m2//:BUILD",
)

local_repository(
  name = "m2",
  path = "$TEST_SRCDIR/m2",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel clean --expunge
  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency"
}

function test_maven_jar_mismatched_sha1_skylark() {
  setup_zoo
  version="1.24"
  serve_artifact com.example.carnivore carnivore 1.24

  wrong_sha1="0123456789012345678901234567890123456789"
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.24",
    repository = 'http://localhost:$fileserver_port/',
    sha1 = '$wrong_sha1',
    local_repository = "@m2//:BUILD",
)

local_repository(
  name = "m2",
  path = "$TEST_SRCDIR/m2",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  expect_log "has SHA-1 of $sha1, does not match expected SHA-1 ($wrong_sha1)"
}

function test_unimplemented_server_attr_skylark() {
  setup_zoo
  version="1.25"
  serve_jar

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:maven_rules.bzl", "maven_jar")

maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:$version",
    server = "attr_not_implemented",
    local_repository = "@m2//:BUILD"
)

local_repository(
  name = "m2",
  path = "$TEST_SRCDIR/m2",
)

bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel build //zoo:ball-pit >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "specifies a 'server' attribute which is currently not supported."
}

run_suite "maven skylark tests"
