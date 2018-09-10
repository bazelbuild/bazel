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
#
# This test has to be run with "bazel test --action_env=PATH --action_env=HOME", because the inner
# Bazel needs to be able to pass $HOME and $PATH to "docker" so that it can download the container
# from gcr.io.
function test_build_bazel_using_docker()  {
    unzip -qo "${DISTFILE}" &> $TEST_log || fail "Could not unzip Bazel's distfile"

    # The first set of flags comes from the instructions on the bazel-toolchains
    # website: https://releases.bazel.build/bazel-toolchains.html
    #
    # The second set of flags enables the Docker sandbox in Bazel.
    bazel build \
      --host_javabase=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:jdk8 \
      --javabase=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:jdk8 \
      --host_java_toolchain=@bazel_tools//tools/jdk:toolchain_hostjdk8 \
      --java_toolchain=@bazel_tools//tools/jdk:toolchain_hostjdk8 \
      --crosstool_top=@bazel_toolchains//configs/ubuntu16_04_clang/1.0/bazel_0.15.0/default:toolchain \
      --action_env=BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN=1 \
      --extra_toolchains=@bazel_toolchains//configs/ubuntu16_04_clang/1.0/bazel_0.15.0/cpp:cc-toolchain-clang-x86_64-default \
      --extra_execution_platforms=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:rbe_ubuntu1604 \
      --host_platform=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:rbe_ubuntu1604 \
      --platforms=@bazel_toolchains//configs/ubuntu16_04_clang/1.0:rbe_ubuntu1604 \
      \
      --experimental_enable_docker_sandbox --experimental_docker_verbose \
      --spawn_strategy=docker --strategy=Javac=docker --genrule_strategy=sandboxed \
      --define=EXECUTOR=remote \
      //src:bazel \
      &> $TEST_log || fail "bazel build with Docker failed"
}

run_suite "bazel docker sandboxing test"
