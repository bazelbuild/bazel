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

set -u
DISTFILE=$(rlocation io_bazel/${1#./})
shift 1

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test that Bazel can be build itself with Docker sandboxing.
function test_build_bazel_using_docker()  {
    unzip -qo "${DISTFILE}" &> $TEST_log || fail "Could not unzip Bazel's distfile"

    bazel build \
      --host_javabase=@bazel_toolchains//configs/debian8_clang/0.3.0:jdk8 \
      --javabase=@bazel_toolchains//configs/debian8_clang/0.3.0:jdk8 \
      --crosstool_top=@bazel_toolchains//configs/debian8_clang/0.3.0/bazel_0.10.0:toolchain \
      --experimental_remote_platform_override='properties:{ name:"container-image" value:"docker://gcr.io/cloud-marketplace/google/rbe-debian8@sha256:1ede2a929b44d629ec5abe86eee6d7ffea1d5a4d247489a8867d46cfde3e38bd" }' \
      --spawn_strategy=docker --strategy=Javac=docker --genrule_strategy=sandboxed \
      --define=EXECUTOR=remote \
      //src:bazel \
      &> $TEST_log || fail "bazel build with Docker failed"
}

run_suite "bazel docker sandboxing test"
