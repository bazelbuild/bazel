#!/bin/bash
#
# Copyright 2021 The Bazel Authors. All rights reserved.
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
# The script is used to generate rbe configurations. Run this at the root of
# the source tree:
#
#   configs/rbe/gen.sh
#

set -eu

BAZEL_VERSION=4.1.0

function generate() {
  TOOLCHAIN_NAME=$1
  TOOLCHAIN_CONTAINER=$2

  rm -rf configs/rbe/$TOOLCHAIN_NAME
  rbe_configs_gen \
    --bazel_version=$BAZEL_VERSION \
    --toolchain_container=$TOOLCHAIN_CONTAINER \
    --output_src_root=. \
    --output_config_path=configs/rbe/$TOOLCHAIN_NAME \
    --exec_os=linux \
    --target_os=linux

  BUILD_FILES=( "cc/BUILD" "config/BUILD" "java/BUILD" )
  # add filegroup srcs
  for BUILD_FILE in "${BUILD_FILES[@]}"
  do
    cat >> configs/rbe/$TOOLCHAIN_NAME/$BUILD_FILE <<EOF
filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//configs/rbe:__subpackages__"],
)
EOF
  done

  cat >> configs/rbe/$TOOLCHAIN_NAME/BUILD <<EOF
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//configs/rbe/$TOOLCHAIN_NAME/cc:srcs",
        "//configs/rbe/$TOOLCHAIN_NAME/java:srcs",
        "//configs/rbe/$TOOLCHAIN_NAME/config:srcs",
    ],
    visibility = ["//configs/rbe:__subpackages__"],
)
EOF
}

generate rbe_ubuntu1604_java8 gcr.io/bazel-public/ubuntu1604-bazel-java8:latest
generate rbe_ubuntu1804_java11 gcr.io/bazel-public/ubuntu1804-bazel-java11:latest
