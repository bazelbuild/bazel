#! /usr/bin/env bash
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

# This script locates the android sdk (via ANDROID_HOME) and ndk
# (via ANDROID_NDK) on your system, and writes a WORKSPACE.user.bzl
# that points to them.

set -euo pipefail

while [ ! -f WORKSPACE ]; do
  cd ..
  if [ "/" = $(pwd) ]
  then
    echo "WORKSPACE not found." 1>&2
    exit 1
  fi
done

: ${ANDROID_BUILD_TOOLS_VERSION:="21.1.1"}
: ${ANDROID_API_LEVEL:="19"}

if [ -n "${ANDROID_HOME:-}" ]; then
  cat <<EOF >WORKSPACE.user.bzl
def android_sdk():
  native.android_sdk_repository(
    name = "androidsdk",
    path = "${ANDROID_HOME}",
    # Available versions are under /path/to/sdk/build-tools/.
    build_tools_version = "${ANDROID_BUILD_TOOLS_VERSION}",
    # Available versions are under /path/to/sdk/platforms/.
    api_level = ${ANDROID_API_LEVEL},
  )
  native.bind(name = "android_sdk_for_testing", actual = "@androidsdk//:files")
EOF
else
    cat <<EOF >WORKSPACE.user.bzl
def android_sdk():
    native.bind(name = "android_sdk_for_testing", actual = "//:dummy")
EOF
fi

if [ -n "${ANDROID_NDK:-}" ]; then
  cat <<EOF >>WORKSPACE.user.bzl
def android_ndk():
  native.android_ndk_repository(
    name = "androidndk",
    path = "${ANDROID_NDK}",
    api_level = ${ANDROID_API_LEVEL},
  )
  native.bind(name = "android_ndk_for_testing", actual = "@androidndk//:files")
EOF
else
  cat <<EOF >>WORKSPACE.user.bzl
def android_ndk():
    native.bind(name = "android_ndk_for_testing", actual = "//:dummy")
EOF
fi

cat <<EOF >>WORKSPACE.user.bzl
def android_repositories():
  android_sdk()
  android_ndk()
EOF
