#!/bin/bash

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
CURDIR="$(pwd)"
MACHINE_ARCH=`uname -m`

if [ ${MACHINE_ARCH} == 's390x' ]; then

# Install dependencies
sudo apt-get update
sudo apt-get install -y wget curl openjdk-11-jdk unzip patch build-essential zip python3 git libapr1

sudo ln -sf /usr/bin/python3 /usr/bin/python

# Bootstrap bazel
mkdir bazel-bootstrap && cd bazel-bootstrap
wget https://github.com/bazelbuild/bazel/releases/download/"${RELEASE_NAME}"/bazel-"${RELEASE_NAME}"-dist.zip
unzip bazel-"${RELEASE_NAME}"-dist.zip
chmod -R +w .
env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
export PATH=$PATH:$CURDIR/bazel-bootstrap/output/

# Compile imtermediate Binary
cd $CURDIR
bazel build --host_javabase=@local_jdk//:jdk --sandbox_tmpfs_path=/tmp //:bazel-distfile
cd $CURDIR
mkdir bazel-temp && cd bazel-temp
unzip $CURDIR/bazel-bin/bazel-distfile.zip
bash ./compile.sh
cd $CURDIR
mkdir output
cp "bazel-temp/output/bazel" output/bazel
else

# Get Bazelisk
mkdir -p /tmp/tool
BAZELISK="/tmp/tool/bazelisk"
wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-linux-amd64 -O "${BAZELISK}"
chmod +x "${BAZELISK}"

"${BAZELISK}" build --sandbox_tmpfs_path=/tmp //src:bazel
mkdir output
cp bazel-bin/src/bazel output/bazel
fi

if [ ${MACHINE_ARCH} == 's390x' ]; then

output/bazel build \
    -c opt \
    --stamp \
    --sandbox_tmpfs_path=/tmp \
    --embed_label "${RELEASE_NAME}" \
    --workspace_status_command=scripts/ci/build_status_command.sh \
      //:bazel-distfile
# Compile s390x Binary
cd $CURDIR
mkdir bazel-s390x && cd bazel-s390x
unzip $CURDIR/bazel-bin/bazel-distfile.zip
env EMBED_LABEL="${RELEASE_NAME}" bash ./compile.sh
cd $CURDIR

mkdir artifacts
cp bazel-s390x/output/bazel "artifacts/bazel-${RELEASE_NAME}-linux-s390x"
else
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

fi
