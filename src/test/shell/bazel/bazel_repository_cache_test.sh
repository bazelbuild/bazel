#!/bin/bash
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
#
# Test repository cache mechanisms
# TODO(jingwen): Test Skylark's download and download_and_extract

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function set_up() {
  bazel clean --expunge >& $TEST_log
  mkdir -p zoo

  http_archive_helper
  repo_cache_dir=$TEST_TMPDIR/repository_cache

  # Delete the repository cache if it exists
  rm -rf "$repo_cache_dir"

  # Test with the extension
  serve_file $repo2_zip
  cat > WORKSPACE <<EOF
http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/bleh',
    sha256 = '$sha256',
    type = 'zip',
)
EOF
}

# Test downloading a file from a repository.
# This creates a simple repository containing:
#
# repo/
#   WORKSPACE
#   zoo/
#     BUILD
#     female.sh
#
# And a .zip file, which contains:
#
# WORKSPACE
# fox/
#   BUILD
#   male
#   male_relative -> male
#   male_absolute -> /fox/male
function http_archive_helper() {
  local write_workspace
  [[ $# -gt 1 ]] && [[ "$2" = "nowrite" ]] && write_workspace=1 || write_workspace=0

  if [[ $write_workspace -eq 0 ]]; then
    # Create a zipped-up repository HTTP response.
    repo2=$TEST_TMPDIR/repo2
    rm -rf $repo2
    mkdir -p $repo2/fox
    cd $repo2
    touch WORKSPACE
    cat > fox/BUILD <<EOF
filegroup(
    name = "fox",
    srcs = ["male"],
    visibility = ["//visibility:public"],
)
EOF
    what_does_the_fox_say="Fraka-kaka-kaka-kaka-kow"
    cat > fox/male <<EOF
#!/bin/bash
echo $what_does_the_fox_say
EOF
    chmod +x fox/male
    ln -s male fox/male_relative
    ln -s /fox/male fox/male_absolute
    # Add some padding to the .zip to test that Bazel's download logic can
    # handle breaking a response into chunks.
    dd if=/dev/zero of=fox/padding bs=1024 count=10240 >& $TEST_log
    repo2_zip=$TEST_TMPDIR/fox.zip
    zip -0 -ry $repo2_zip WORKSPACE fox >& $TEST_log
    repo2_name=$(basename $repo2_zip)
    sha256=$(sha256sum $repo2_zip | cut -f 1 -d ' ')
  fi
  serve_file $repo2_zip

  cd ${WORKSPACE_DIR}
  if [[ $write_workspace = 0 ]]; then
    cat > WORKSPACE <<EOF
http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/$repo2_name',
    sha256 = '$sha256'
)
EOF

    cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["@endangered//fox"],
)
EOF

    cat > zoo/female.sh <<EOF
#!/bin/bash
../endangered/fox/male
EOF
    chmod +x zoo/female.sh
fi

  bazel run //zoo:breeding-program >& $TEST_log --show_progress_rate_limit=0 \
    || echo "Expected build/run to succeed"
  kill_nc
  expect_log "$what_does_the_fox_say"

  base_external_path=bazel-out/../external/endangered/fox
  assert_files_same ${base_external_path}/male ${base_external_path}/male_relative
  assert_files_same ${base_external_path}/male ${base_external_path}/male_absolute
}

function assert_files_same() {
  assert_contains "$(cat $1)" $2 && return 0
  echo "Expected these to be the same:"
  echo "---------------------------"
  cat $1
  echo "==========================="
  cat $2
  echo "---------------------------"
  return 1
}

function test_build() {
  bazel run --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected build/run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

function test_fetch() {
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  kill_nc
  expect_log "All external dependencies fetched successfully"
}

function test_directory_structure() {
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  kill_nc
  if [ ! -d $repo_cache_dir/content_addressable/sha256/ ]; then
    fail "repository cache directories were not created"
  fi
}

function test_cache_entry_exists() {
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  kill_nc
  if [ ! -f $repo_cache_dir/content_addressable/sha256/$sha256/file ]; then
    fail "the file was not cached successfully"
  fi
}

function test_load_cached_value() {
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  kill_nc
  bazel clean --expunge

  # Fetch again
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}

function test_failed_fetch_without_cache() {
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server and reset state
  kill_nc
  bazel clean --expunge
  rm -rf "$repo_cache_dir"

  # Attempt to fetch again
  bazel fetch --experimental_repository_cache=$repo_cache_dir //zoo:breeding-program >& $TEST_log \
    && echo "Expected fetch to fail"

  expect_log "Error downloading"
}

run_suite "repository cache tests"
