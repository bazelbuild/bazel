#!/bin/bash
set -e
set -x

RELEASE_NAME=${RELEASE_NAME:-unknown}

# Get Bazelisk
mkdir -p /tmp/tool
BAZELISK="/tmp/tool/bazelisk"
wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-linux-amd64 -O "${BAZELISK}"
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
cp "bazel-bin/src/bazel" "artifacts/bazel-${RELEASE_NAME}-linux-x86_64"
cp "bazel-bin/scripts/packages/with-jdk/install.sh" "artifacts/bazel-${RELEASE_NAME}-installer-linux-x86_64.sh"
cp "bazel-bin/scripts/packages/debian/bazel-debian.deb" "artifacts/bazel_${RELEASE_NAME}-linux-x86_64.deb"
cp "bazel-bin/scripts/packages/debian/bazel.dsc" "artifacts/bazel_${RELEASE_NAME}.dsc"
cp "bazel-bin/scripts/packages/debian/bazel.tar.gz" "artifacts/bazel_${RELEASE_NAME}.tar.gz"
cp "bazel-bin/bazel-distfile.zip" "artifacts/bazel-${RELEASE_NAME}-dist.zip"

