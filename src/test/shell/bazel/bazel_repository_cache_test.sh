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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function set_up() {
  bazel clean --expunge >& $TEST_log
  repo_cache_dir=$TEST_TMPDIR/repository_cache
  # TODO(b/37617303): make test UI-independent
  add_to_bazelrc "fetch --noexperimental_ui"
  add_to_bazelrc "build --noexperimental_ui"
}

function tear_down() {
  shutdown_server
  rm -rf "$repo_cache_dir"
}

function setup_repository() {
  http_archive_helper

  # Test with the extension
  serve_file $repo2_zip
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/bleh',
    sha256 = '$sha256',
    type = 'zip',
)
EOF
}

function setup_skylark_repository() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"

  zip_file="${server_dir}/zip_file.zip"

  touch "${server_dir}"/WORKSPACE
  echo "some content" > "${server_dir}"/file
  zip -0 -ry $zip_file "${server_dir}"/WORKSPACE "${server_dir}"/file >& $TEST_log

  zip_sha256="$(sha256sum "${zip_file}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  cat > WORKSPACE <<EOF
load('//:test.bzl', 'repo')
repo(name = 'foo')
EOF
  touch BUILD
}

function setup_maven_repository() {
  mkdir -p zoo
  cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["//external:mongoose"],
)
EOF

  cat > zoo/BallPit.java <<EOF
import carnivore.Mongoose;

public class BallPit {
    public static void main(String args[]) {
        Mongoose.frolic();
    }
}
EOF

  serve_artifact com.example.carnivore carnivore 1.23

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$fileserver_port/',
    sha1 = '$sha1',
    sha1_src = '$sha1_src',
)
bind(name = 'mongoose', actual = '@endangered//jar')
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
    rm -rf "$repo2"
    mkdir -p "$repo2/fox"
    cd "$repo2"
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
#!/bin/sh
echo $what_does_the_fox_say
EOF
    chmod +x fox/male
    ln -s male fox/male_relative
    ln -s /fox/male fox/male_absolute
    # Add some padding to the .zip to test that Bazel's download logic can
    # handle breaking a response into chunks.
    dd if=/dev/zero of=fox/padding bs=1024 count=10240 >& $TEST_log
    repo2_zip="$TEST_TMPDIR/fox.zip"
    zip -0 -ry "$repo2_zip" WORKSPACE fox >& $TEST_log
    repo2_name=$(basename "$repo2_zip")
    sha256=$(sha256sum "$repo2_zip" | cut -f 1 -d ' ')
  fi
  serve_file "$repo2_zip"

  cd ${WORKSPACE_DIR}

  mkdir -p zoo

  if [[ $write_workspace = 0 ]]; then
    cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
#!/bin/sh
../endangered/fox/male
EOF
    chmod +x zoo/female.sh
fi

  bazel run //zoo:breeding-program >& $TEST_log --show_progress_rate_limit=0 \
    || echo "Expected build/run to succeed"
  shutdown_server
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
  setup_repository

  bazel run --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected build/run to succeed"
  expect_log $what_does_the_fox_say
}

function test_fetch() {
  setup_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  expect_log "All external dependencies fetched successfully"
}

function test_directory_structure() {
  setup_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  if [ ! -d $repo_cache_dir/content_addressable/sha256/ ]; then
    fail "repository cache directories were not created"
  fi
}

function test_cache_entry_exists() {
  setup_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"
  if [ ! -f $repo_cache_dir/content_addressable/sha256/$sha256/file ]; then
    fail "the file was not cached successfully"
  fi
}

function test_fetch_value_with_existing_cache_and_no_network() {
  setup_repository

  # Manual cache injection
  cache_entry="$repo_cache_dir/content_addressable/sha256/$sha256"
  mkdir -p "$cache_entry"
  cp "$repo2_zip" "$cache_entry/file" # Artifacts are named uniformly as "file" in the cache

  # Fetch without a server
  shutdown_server
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
      || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}


function test_load_cached_value() {
  setup_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}

function test_write_cache_without_hash() {
  setup_repository

  # Have a WORKSPACE file without the specified sha256
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/bleh',
    type = 'zip',
)
EOF

  # Fetch; as we did not specify a hash, we expect bazel to tell us the hash
  # in an info message.
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || fail "expected fetch to succeed"

  expect_log "${sha256}"

  # Shutdown the server; so fetching again won't work
  shutdown_server
  bazel clean --expunge

  # As we don't have a predicted cache, we expect fetching to fail now.
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    && fail "expected failure" || :

  # However, if we add the hash, the value is taken from cache
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/bleh',
    type = 'zip',
    sha256 = '${sha256}',
)
EOF
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || fail "expected fetch to succeed"
}

function test_failed_fetch_without_cache() {
  setup_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server and reset state
  shutdown_server
  bazel clean --expunge

  # Clean the repository cache
  rm -rf "$repo_cache_dir"

  # Attempt to fetch again
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:breeding-program >& $TEST_log \
    && echo "Expected fetch to fail"

  expect_log "Error downloading"
}

function test_skylark_download_file_exists_in_cache() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  if [ ! -f $repo_cache_dir/content_addressable/sha256/$zip_sha256/file ]; then
    fail "the file was not cached successfully"
  fi
}

function test_skylark_download_and_extract_file_exists_in_cache() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  if [ ! -f $repo_cache_dir/content_addressable/sha256/$zip_sha256/file ]; then
    fail "the file was not cached successfully"
  fi
}

function test_load_cached_value_skylark_download() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}

function test_skylark_download_fail_without_cache() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Clean the repository cache
  rm -rf "$repo_cache_dir"

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    && echo "Expected fetch to fail"

  expect_log "Error downloading"
}

function test_load_cached_value_skylark_download_and_extract() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}

function test_skylark_download_and_extract_fail_without_cache() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/zip_file.zip", "zip_file.zip", "${zip_sha256}")
  repository_ctx.file("BUILD")
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Clean the repository cache
  rm -rf "$repo_cache_dir"

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" @foo//:all >& $TEST_log \
    && echo "Expected fetch to fail"

  expect_log "Error downloading"
}

function test_maven_jar_exists_in_cache() {
  setup_maven_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:ball-pit >& $TEST_log \
    || echo "Expected fetch to succeed"

  if [ ! -f $repo_cache_dir/content_addressable/sha1/$sha1/file ]; then
    fail "the jar file was not cached successfully"
  fi

  if [ ! -f $repo_cache_dir/content_addressable/sha1/$sha1_src/file ]; then
    fail "the sources file was not cached successfully"
  fi
}

function test_load_cached_value_maven_jar() {
  setup_maven_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:ball-pit >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:ball-pit >& $TEST_log \
    || echo "Expected fetch to succeed"

  expect_log "All external dependencies fetched successfully"
}

function test_maven_jar_fail_without_cache() {
  setup_maven_repository

  bazel fetch --repository_cache="$repo_cache_dir" //zoo:ball-pit >& $TEST_log \
    || echo "Expected fetch to succeed"

  # Kill the server
  shutdown_server
  bazel clean --expunge

  # Clean the repository cache
  rm -rf "$repo_cache_dir"

  # Fetch again
  bazel fetch --repository_cache="$repo_cache_dir" //zoo:ball-pit >& $TEST_log \
    && echo "Expected fetch to fail"

  expect_log "Failed to fetch Maven dependency"
}

run_suite "repository cache tests"
