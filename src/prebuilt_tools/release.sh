#!/usr/bin/env bash

# Copyright 2026 The Bazel Authors. All rights reserved.
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

# Promotes the prebuilt tools archives that src/prebuilt_tools/upload.sh
# uploaded for the given commit to a release and prints their hashes for
# _PREBUILT_TOOLS_SHA256 in tools/test/extensions.bzl.
#
# Usage: src/prebuilt_tools/release.sh <commit hash> <version>

set -euo pipefail

commit_hash="$1"
version="$2"

platforms=(
  darwin_arm64
  darwin_x86_64
  linux_aarch64
  linux_x86_64
  windows_arm64
  windows_x86_64
)

tmp_dir="$(mktemp -d -t 'tmp_bazel_prebuilt_tools_XXXXXX')"
trap 'rm -fr "$tmp_dir"' EXIT

for platform in "${platforms[@]}"; do
  gcloud storage cp \
    "gs://bazel-mirror/bazel_prebuilt_tools/tmp/${commit_hash}/prebuilt_tools_${platform}.zip" \
    "${tmp_dir}/"
done

echo "_PREBUILT_TOOLS_VERSION = \"${version}\""
echo "_PREBUILT_TOOLS_SHA256 = {"
for platform in "${platforms[@]}"; do
  zip_file="${tmp_dir}/prebuilt_tools_${platform}.zip"
  gcloud storage cp --no-clobber "$zip_file" \
    "gs://bazel-mirror/bazel_prebuilt_tools/releases/v${version}/prebuilt_tools_${platform}-v${version}.zip"
  echo "    \"${platform}\": \"$(sha256sum "$zip_file" | cut -d" " -f1)\","
done
echo "}"
