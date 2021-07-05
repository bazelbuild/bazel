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
# The script is used to generate rbe configurations.
#

set -eu

cd $(dirname $BASH_SOURCE)/../..
BASEDIR=configs/rbe

BAZEL_VERSION=4.1.0

function generate() {
  TOOLCHAIN_NAME=$1
  TOOLCHAIN_CONTAINER=$2
  CPP_ENV_JSON=$3

  rm -rf $BASEDIR/$TOOLCHAIN_NAME
  rbe_configs_gen \
    --bazel_version=$BAZEL_VERSION \
    --toolchain_container=$TOOLCHAIN_CONTAINER \
    --cpp_env_json=$CPP_ENV_JSON \
    --output_src_root=. \
    --output_config_path=$BASEDIR/$TOOLCHAIN_NAME \
    --exec_os=linux \
    --target_os=linux

  # HACK: @bazel_tools/tools/cpp:clang is currently hardcoded in rbe_configs_gen
  # since clang is the default compiler for linux. As we use --cpp_env_json to
  # override the default compiler to gcc, we also need to replace clang with gcc
  # here.
  #
  # TODO: Add support for customizing execution_constrains to rbe_configs_gen.
  sed -i "s/@bazel_tools\/\/tools\/cpp:clang/@bazel_tools\/\/tools\/cpp:gcc/g" \
    $BASEDIR/$TOOLCHAIN_NAME/config/BUILD

  BUILD_FILES=( "cc/BUILD" "config/BUILD" "java/BUILD" )
  # add filegroup srcs
  for BUILD_FILE in "${BUILD_FILES[@]}"
  do
    cat >> $BASEDIR/$TOOLCHAIN_NAME/$BUILD_FILE <<EOF
filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//configs/rbe:__subpackages__"],
)
EOF
  done

  cat >> $BASEDIR/$TOOLCHAIN_NAME/BUILD <<EOF
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

generate rbe_ubuntu1604_java8 gcr.io/bazel-public/ubuntu1604-bazel-java8:latest $BASEDIR/cpp_env_ubuntu1604.json
generate rbe_ubuntu1804_java11 gcr.io/bazel-public/ubuntu1804-bazel-java11:latest $BASEDIR/cpp_env_ubuntu1804.json
