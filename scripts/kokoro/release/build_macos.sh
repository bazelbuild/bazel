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

# Get Bazelisk
mkdir -p /tmp/tool
BAZELISK="/tmp/tool/bazelisk"
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.2/bazelisk-darwin-amd64 -O "${BAZELISK}"
chmod +x "${BAZELISK}"

# Switch to Xcode 10.3
sudo xcode-select -s /Applications/Xcode_10.3.app/Contents/Developer

"${BAZELISK}" build //src:bazel
mkdir output
cp bazel-bin/src/bazel output/bazel

output/bazel build \
    --define IPHONE_SDK=1 \
    -c opt \
    --stamp \
    --embed_label "${RELEASE_NAME}" \
    --workspace_status_command=scripts/ci/build_status_command.sh \
    src/bazel \
    scripts/packages/with-jdk/install.sh \
    scripts/packages/dmg/bazel.dmg

mkdir artifacts
cp "bazel-bin/src/bazel"  "artifacts/bazel-${RELEASE_NAME}-darwin-x86_64"
cp "bazel-bin/scripts/packages/with-jdk/install.sh" "artifacts/bazel-${RELEASE_NAME}-installer-darwin-x86_64.sh"
cp "bazel-bin/scripts/packages/dmg/bazel.dmg" "artifacts/bazel-${RELEASE_NAME}-darwin-x86_64-unsigned.dmg"
