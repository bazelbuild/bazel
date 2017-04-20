#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
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

# Ideally we would call directly script/ci/build.sh just like we do
# for the linux script but we are not there yet.

# Ensure we are in the root directory
cd $(dirname $0)/../../..

# Even though there are no quotes around $* in the .bat file, arguments
# containing spaces seem to be passed properly.
source ./scripts/ci/build.sh

# Bazel still needs to know where bash is, take it from cygpath.
export BAZEL_SH="$(cygpath --windows /bin/bash)"
# Make sure JAVA_HOME is in Windows path style.
export JAVA_HOME="$(cygpath --windows "${JAVA_HOME}")"

# TODO(bazel-team): we should replace ./compile.sh by the same script we use
# for other platform
release_label="$(get_full_release_name)"

if [ -n "${release_label}" ]; then
  export EMBED_LABEL="${release_label}"
fi

# On windows-msvc-x86_64, we build a MSVC Bazel
MSVC_OPTS=""
MSVC_LABEL=""
if [[ $PLATFORM_NAME == windows-msvc-x86_64* ]]; then
  MSVC_OPTS="--cpu=x64_windows_msvc --copt=/w"
  MSVC_LABEL="-msvc"
fi

${BOOTSTRAP_BAZEL} --bazelrc=${BAZELRC:-/dev/null} --nomaster_bazelrc build \
    --embed_label=${release_label} --stamp \
    ${MSVC_OPTS} \
    //src:bazel //src:bazel_with_jdk

# Copy the resulting artifacts.
mkdir -p output/ci
cp bazel-bin/src/bazel output/ci/bazel${MSVC_LABEL}-$(get_full_release_name)-without-jdk.exe
cp bazel-bin/src/bazel_with_jdk output/ci/bazel${MSVC_LABEL}-$(get_full_release_name).exe
cp bazel-bin/src/bazel output/bazel.exe
zip -j output/ci/bazel${MSVC_LABEL}-$(get_full_release_name)-without-jdk.zip output/bazel.exe
cp -f bazel-bin/src/bazel_with_jdk output/bazel.exe
zip -j output/ci/bazel${MSVC_LABEL}-$(get_full_release_name).zip output/bazel.exe
