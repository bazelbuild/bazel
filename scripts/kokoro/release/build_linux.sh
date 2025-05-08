#!/usr/bin/env bash

# Copyright 2020 The Bazel Authors. All rights reserved.
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

set -e
set -x

RELEASE_NAME=${RELEASE_NAME:-unknown}
ARCHITECTURE=`uname -m`

if [ $ARCHITECTURE = "aarch64" ]; then
    BAZELISK_EXT="arm64"
else
    BAZELISK_EXT="amd64"
fi

# Get Bazelisk
mkdir -p /tmp/tool
BAZELISK="/tmp/tool/bazelisk"
wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.7.2/bazelisk-linux-"${BAZELISK_EXT}" -O "${BAZELISK}"
chmod +x "${BAZELISK}"

"${BAZELISK}" build --sandbox_tmpfs_path=/tmp //src:bazel
mkdir output
cp bazel-bin/src/bazel output/bazel

output/bazel build \
    -c opt \
    --stamp \
    --sandbox_tmpfs_path=/tmp \
    --embed_label "${RELEASE_NAME}" \
    --workspace_status_command=scripts/ci/build_status_command.sh \
    src/bazel \
    scripts/packages/with-jdk/install.sh \
    scripts/packages/debian/bazel-debian.deb \
    scripts/packages/debian/bazel.dsc \
    scripts/packages/debian/bazel.tar.gz \
    bazel-distfile.zip

mkdir artifacts
cp "bazel-bin/src/bazel" "artifacts/bazel-${RELEASE_NAME}-linux-${ARCHITECTURE}"
cp "bazel-bin/scripts/packages/with-jdk/install.sh" "artifacts/bazel-${RELEASE_NAME}-installer-linux-${ARCHITECTURE}.sh"
cp "bazel-bin/scripts/packages/debian/bazel-debian.deb" "artifacts/bazel_${RELEASE_NAME}-linux-${ARCHITECTURE}.deb"

if [ $ARCHITECTURE = "x86_64" ]; then
    cp "bazel-bin/scripts/packages/debian/bazel.dsc" "artifacts/bazel_${RELEASE_NAME}.dsc"
    cp "bazel-bin/scripts/packages/debian/bazel.tar.gz" "artifacts/bazel_${RELEASE_NAME}.tar.gz"
    cp "bazel-bin/bazel-distfile.zip" "artifacts/bazel-${RELEASE_NAME}-dist.zip"
fi

