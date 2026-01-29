#!/usr/bin/env bash
#
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
#
# Tests for the remote downlaader backed by the Remote Asset API.

set -euo pipefail

# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/bazel/remote_helpers.sh")" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

function set_up() {
  start_worker
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

function test_remote_downloader_http_archive() {
  local archive_dir="${TEST_TMPDIR}/archive"
  mkdir -p "${archive_dir}"
  cat > "${archive_dir}/BUILD.bazel" <<'EOF'
filegroup(
    name = "files",
    srcs = ["data.txt"],
    visibility = ["//visibility:public"],
)
EOF
  echo "Hello from remote archive" > "${archive_dir}/data.txt"
  touch "${archive_dir}/REPO.bazel"

  # Create the archive
  local archive_file="${TEST_TMPDIR}/mylib.tar.gz"
  tar -czf "${archive_file}" -C "${archive_dir}" .
  local sha256=$(sha256sum "${archive_file}" | cut -f 1 -d ' ')

  # Serve the archive
  serve_file "${archive_file}"

  # Set up the workspace
  mkdir -p main
  cd main
  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "mylib",
    url = "http://127.0.0.1:${nc_port}/served_file.$$",
    sha256 = "${sha256}",
    type = "tar.gz",
)
EOF

  cat > BUILD.bazel <<'EOF'
filegroup(
    name = "test",
    srcs = ["@mylib//:files"],
)
EOF

  # Build using the remote downloader
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:test >& $TEST_log \
      || fail "Failed to build with remote downloader"

  # Verify the content was downloaded correctly
  local output_base=$(bazel info output_base)
  local output_file="${output_base}/external/+http_archive+mylib/data.txt"
  assert_contains "Hello from remote archive" "${output_file}"
}

function test_remote_downloader_checksum_mismatch() {
  # Test that a checksum mismatch from the remote downloader is handled correctly
  local archive_dir="${TEST_TMPDIR}/archive2"
  mkdir -p "${archive_dir}"
  cat > "${archive_dir}/BUILD.bazel" <<'EOF'
filegroup(
    name = "data",
    srcs = ["content.txt"],
    visibility = ["//visibility:public"],
)
EOF
  echo "Checksum mismatch content" > "${archive_dir}/content.txt"
  touch "${archive_dir}/REPO.bazel"

  local archive_file="${TEST_TMPDIR}/pkg.zip"
  (cd "${archive_dir}" && zip -r "${archive_file}" .)
  # Use a wrong checksum intentionally
  local wrong_sha256="0000000000000000000000000000000000000000000000000000000000000000"

  serve_file "${archive_file}"

  mkdir -p main2
  cd main2
  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "pkg",
    url = "http://127.0.0.1:${nc_port}/served_file.$$",
    sha256 = "${wrong_sha256}",
    type = "zip",
)
EOF

  cat > BUILD.bazel <<'EOF'
filegroup(
    name = "check",
    srcs = ["@pkg//:data"],
)
EOF

  # The build should fail due to checksum mismatch
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:check >& $TEST_log \
      && fail "Expected build to fail due to checksum mismatch"

  expect_log "Checksum was"
  expect_log "${wrong_sha256}"
}

function test_remote_downloader_canonical_id() {
  # Test that the canonical_id qualifier is respected - same URL with different
  # canonical IDs should be treated as different entries

  local archive_dir="${TEST_TMPDIR}/archive4"
  mkdir -p "${archive_dir}"
  cat > "${archive_dir}/BUILD.bazel" <<'EOF'
filegroup(
    name = "files",
    srcs = ["version.txt"],
    visibility = ["//visibility:public"],
)
EOF
  echo "Version 1" > "${archive_dir}/version.txt"
  touch "${archive_dir}/REPO.bazel"

  local archive_file="${TEST_TMPDIR}/canonical.tar.gz"
  tar -czf "${archive_file}" -C "${archive_dir}" .
  local sha256=$(sha256sum "${archive_file}" | cut -f 1 -d ' ')

  serve_file "${archive_file}"

  mkdir -p main4
  cd main4

  # First, fetch with one canonical ID
  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "canonical_repo",
    url = "http://127.0.0.1:${nc_port}/served_file.$$",
    sha256 = "${sha256}",
    type = "tar.gz",
    canonical_id = "version-1",
)
EOF

  cat > BUILD.bazel <<'EOF'
filegroup(
    name = "test",
    srcs = ["@canonical_repo//:files"],
)
EOF

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:test >& $TEST_log \
      || fail "Failed first build with canonical_id"

  # Verify the first content
  local output_base=$(bazel info output_base)
  local output_file="${output_base}/external/+http_archive+canonical_repo/version.txt"
  assert_contains "Version 1" "${output_file}"

  # Now update the archive with different content
  echo "Version 2" > "${archive_dir}/version.txt"
  tar -czf "${archive_file}" -C "${archive_dir}" .
  local sha256_v2=$(sha256sum "${archive_file}" | cut -f 1 -d ' ')

  # Update the served file (serve_file copies to a different location)
  cat "${archive_file}" > "${TEST_TMPDIR}/served_file.$$"

  # Clean to force re-fetch with new repository definition
  bazel clean >& $TEST_log

  # Update MODULE.bazel with new canonical ID and checksum
  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "canonical_repo",
    url = "http://127.0.0.1:${nc_port}/served_file.$$",
    sha256 = "${sha256_v2}",
    type = "tar.gz",
    canonical_id = "version-2",
)
EOF

  # Build again with the new canonical ID - the remote downloader should
  # recognize this as a different request due to the different canonical_id
  # and re-download the file
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:test >& $TEST_log \
      || fail "Failed to build with new canonical_id"

  # Verify the updated content was fetched
  assert_contains "Version 2" "${output_file}"
}

function test_remote_downloader_caching() {
  # Test that the remote downloader caches downloaded files - when the same
  # URL with the same checksum is requested again, it should use the cached
  # version without re-downloading.

  local archive_dir="${TEST_TMPDIR}/archive5"
  mkdir -p "${archive_dir}"
  cat > "${archive_dir}/BUILD.bazel" <<'EOF'
filegroup(
    name = "files",
    srcs = ["cached.txt"],
    visibility = ["//visibility:public"],
)
EOF
  echo "Cached content" > "${archive_dir}/cached.txt"
  touch "${archive_dir}/REPO.bazel"

  local archive_file="${TEST_TMPDIR}/cached.tar.gz"
  tar -czf "${archive_file}" -C "${archive_dir}" .
  local sha256=$(sha256sum "${archive_file}" | cut -f 1 -d ' ')

  serve_file "${archive_file}"

  mkdir -p main5
  cd main5

  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "cached_repo",
    url = "http://127.0.0.1:${nc_port}/served_file.$$",
    sha256 = "${sha256}",
    type = "tar.gz",
)
EOF

  cat > BUILD.bazel <<'EOF'
filegroup(
    name = "test",
    srcs = ["@cached_repo//:files"],
)
EOF

  # First build - should download from the origin
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:test >& $TEST_log \
      || fail "Failed first build"

  # Shut down the file server to verify that the second fetch uses cache
  shutdown_server

  # Clean Bazel's local cache but keep the remote worker running
  bazel clean --expunge >& $TEST_log

  # Second build - should use the cached version from the remote downloader
  # (no HTTP server running, so it would fail if trying to re-download)
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --experimental_remote_downloader=grpc://localhost:${worker_port} \
      //:test >& $TEST_log \
      || fail "Failed second build - should have used cache"

  # Verify the content
  local output_base=$(bazel info output_base)
  local output_file="${output_base}/external/+http_archive+cached_repo/cached.txt"
  assert_contains "Cached content" "${output_file}"
}

run_suite "Remote downloader tests"
