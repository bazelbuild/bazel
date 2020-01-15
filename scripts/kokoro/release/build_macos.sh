#!/bin/bash
set -e
set -x

RELEASE_NAME=${RELEASE_NAME:-unknown}

# Get Bazelisk
mkdir -p /tmp/tool
BAZELISK="/tmp/tool/bazelisk"
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-darwin-amd64 -O "${BAZELISK}"
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
    scripts/packages/with-jdk/install.sh

mkdir artifacts
cp "bazel-bin/src/bazel"  "artifacts/bazel-${RELEASE_NAME}-darwin-x86_64"
cp "bazel-bin/scripts/packages/with-jdk/install.sh" "artifacts/bazel-${RELEASE_NAME}-installer-darwin-x86_64.sh"
