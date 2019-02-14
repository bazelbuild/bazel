#!/bin/bash
#
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
#
# Test //external mechanisms
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

set_up() {
  bazel clean --expunge >& $TEST_log
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
}

tear_down() {
  shutdown_server
}

function zip_up() {
  repo2_zip=$TEST_TMPDIR/fox.zip
  zip -0 -ry $repo2_zip WORKSPACE fox
}

function tar_gz_up() {
  repo2_zip=$TEST_TMPDIR/fox.tar.gz
  tar czf $repo2_zip WORKSPACE fox
}

function tar_xz_up() {
  repo2_zip=$TEST_TMPDIR/fox.tar.xz
  tar cJf $repo2_zip WORKSPACE fox
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
  zipper=$1
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
#!/bin/sh
echo $what_does_the_fox_say
EOF
    chmod +x fox/male
    touch -t 200403010021.42 fox/male
    ln -s male fox/male_relative
    ln -s /fox/male fox/male_absolute
    # Add some padding to the .zip to test that Bazel's download logic can
    # handle breaking a response into chunks.
    dd if=/dev/zero of=fox/padding bs=1024 count=10240 >& $TEST_log
    $zipper >& $TEST_log
    repo2_name=$(basename $repo2_zip)
    sha256=$(sha256sum $repo2_zip | cut -f 1 -d ' ')
  fi
  serve_file $repo2_zip

  cd ${WORKSPACE_DIR}
  if [[ $write_workspace = 0 ]]; then
    cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = 'endangered',
    url = 'http://127.0.0.1:$nc_port/$repo2_name',
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
  kill_nc
  expect_log $what_does_the_fox_say

  base_external_path=bazel-out/../external/endangered/fox
  assert_files_same ${base_external_path}/male ${base_external_path}/male_relative
  assert_files_same ${base_external_path}/male ${base_external_path}/male_absolute
  case "${PLATFORM}" in
    darwin)
      ts="$(stat -f %m ${base_external_path}/male)"
      ;;
    *)
      ts="$(stat -c %Y ${base_external_path}/male)"
      ;;
  esac
  assert_equals "1078100502" "$ts"
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

function test_http_archive_zip() {
  http_archive_helper zip_up

  # Test with the extension
  serve_file $repo2_zip
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = 'endangered',
    url = 'http://127.0.0.1:$nc_port/bleh',
    sha256 = '$sha256',
    type = 'zip',
)
EOF
  bazel run //zoo:breeding-program >& $TEST_log \
    || echo "Expected build/run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

function test_http_archive_tgz() {
  http_archive_helper tar_gz_up
  bazel shutdown
  http_archive_helper tar_gz_up
}

function test_http_archive_tar_xz() {
  http_archive_helper tar_xz_up
}

function test_http_archive_no_server() {
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(name = 'endangered', url = 'http://bad.example/repo.zip',
    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826')
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
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel fetch //zoo:breeding-program >& $TEST_log && fail "Expected fetch to fail"
  expect_log "Unknown host: bad.example"
}

function test_http_archive_mismatched_sha256() {
  # Create a zipped-up repository HTTP response.
  repo2=$TEST_TMPDIR/repo2
  rm -rf $repo2
  mkdir -p $repo2
  cd $repo2
  touch WORKSPACE
  repo2_zip=$TEST_TMPDIR/fox.zip
  zip -r $repo2_zip WORKSPACE
  serve_file $repo2_zip
  wrong_sha256=0000000000000000000000000000000000000000

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = 'endangered',
    url = 'http://127.0.0.1:$nc_port/repo.zip',
    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826',
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
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel fetch //zoo:breeding-program >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "Checksum"
}

# Bazel should not re-download the .zip unless the user requests it or the
# WORKSPACE file changes, so doing a BUILD where the "wrong" .zip is available
# on the server should work if the correct .zip is already available.
function test_sha256_caching() {
  # Download with correct sha256.
  http_archive_helper zip_up

  # Create another HTTP response.
  http_response=$TEST_TMPDIR/http_response
  cat > $http_response <<EOF
HTTP/1.0 200 OK



EOF

  nc_log=$TEST_TMPDIR/nc.log
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  # "Webserver" for http_archive rule.
  nc_l $nc_port < $http_response >& $nc_log &
  pid=$!

  bazel fetch //zoo:breeding-program || fail "Fetch failed"
  bazel run //zoo:breeding-program >& $TEST_log \
    || echo "Expected run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

function test_cached_across_server_restart() {
  http_archive_helper zip_up
  local marker_file=$(bazel info output_base)/external/\@endangered.marker
  echo "<MARKER>"
  cat "${marker_file}"
  echo "</MARKER>"
  bazel shutdown >& $TEST_log || fail "Couldn't shut down"
  bazel run //zoo:breeding-program >& $TEST_log --show_progress_rate_limit=0 \
    || echo "Expected build/run to succeed"
  echo "<MARKER>"
  cat "${marker_file}"
  echo "</MARKER>"
  expect_log $what_does_the_fox_say
  expect_not_log "Downloading from"
}

# Tests downloading a jar and using it as a Java dependency.
function test_jar_download() {
  serve_jar

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_jar")
http_jar(name = 'endangered', url = 'http://127.0.0.1:$nc_port/lib.jar')
EOF

  mkdir -p zoo
  cat > zoo/BUILD <<EOF
java_binary(
    name = "ball-pit",
    srcs = ["BallPit.java"],
    main_class = "BallPit",
    deps = ["@endangered//jar"],
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

  bazel run //zoo:ball-pit >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

function test_http_to_https_redirect() {
  http_response=$TEST_TMPDIR/http_response
  cat > $http_response <<EOF
HTTP/1.0 301 Moved Permantently
Location: https://127.0.0.1:123456789/bad-port-shouldnt-work
EOF
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  nc_l $nc_port < $http_response &
  nc_pid=$!

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(
    name = 'toto',
    urls = ['http://127.0.0.1:$nc_port/toto'],
    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826'
)
EOF
  bazel build @toto//file &> $TEST_log && fail "Expected run to fail"
  kill_nc
  # Observes that we tried to follow redirect, but failed due to ridiculous
  # port.
  expect_log "port out of range"
}

function test_http_404() {
  serve_not_found "Help, I'm lost!"

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(
    name = 'toto',
    urls = ['http://127.0.0.1:$nc_port/toto'],
    sha256 = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9826'
)
EOF
  bazel build @toto//file &> $TEST_log && fail "Expected run to fail"
  kill_nc
  expect_log "404 Not Found"
}

# Tests downloading a file and using it as a dependency.
function test_http_download() {
  local test_file=$TEST_TMPDIR/toto
  cat > $test_file <<EOF
#!/bin/sh
echo "Tra-la!"
EOF
  local sha256=$(sha256sum $test_file | cut -f 1 -d ' ')
  serve_file $test_file
  cd ${WORKSPACE_DIR}

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(name = 'toto', urls = ['http://127.0.0.1:$nc_port/toto'],
    sha256 = '$sha256', executable = True)
EOF

  mkdir -p test
  cat > test/BUILD <<'EOF'
sh_binary(
    name = "test",
    srcs = ["test.sh"],
    data = ["@toto//file"],
)

genrule(
  name = "test_sh",
  outs = ["test.sh"],
  srcs = ["@toto//file"],
  cmd = "echo '#!/bin/sh' > $@ && echo $(location @toto//file) >> $@",
)
EOF

  bazel run //test >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

# Tests downloading a file with a redirect.
function test_http_redirect() {
  local test_file=$TEST_TMPDIR/toto
  echo "Tra-la!" >$test_file
  local sha256=$(sha256sum $test_file | cut -f 1 -d ' ')
  serve_file $test_file
  cd ${WORKSPACE_DIR}
  serve_redirect "http://127.0.0.1:$nc_port/toto"

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(name = 'toto', urls = ['http://127.0.0.1:$redirect_port/toto'],
    sha256 = '$sha256')
EOF

  mkdir -p test
  cat > test/BUILD <<'EOF'
sh_binary(
    name = "test",
    srcs = ["test.sh"],
    data = ["@toto//file"],
)

genrule(
  name = "test_sh",
  outs = ["test.sh"],
  srcs = ["@toto//file"],
  cmd = "echo '#!/bin/sh' > $@ && echo cat $(location @toto//file) >> $@",
)
EOF

  bazel run //test >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

function test_empty_file() {
  rm -f empty
  touch empty
  tar czf x.tar.gz empty
  local sha256=$(sha256sum x.tar.gz | cut -f 1 -d ' ')
  serve_file x.tar.gz

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.tar.gz",
    sha256 = "$sha256",
    build_file = "@//:x.BUILD",
)
EOF
  cat > x.BUILD <<EOF
exports_files(["empty"])
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "rule",
  srcs = ["@x//:empty"],
  outs = ["timestamp"],
  cmd = "date > $@",
)
EOF

  bazel build //:rule || fail "Build failed"
  bazel shutdown || fail "Shutdown failed"
  cp bazel-genfiles/timestamp first_timestamp || fail "No output"
  sleep 1 # Make sure we're on a different second to avoid false passes.
  bazel build //:rule || fail "Build failed"
  diff bazel-genfiles/timestamp first_timestamp || fail "Output was built again"
}

function test_new_remote_repo_with_build_file() {
  do_new_remote_repo_test "build_file"
}

function test_new_remote_repo_with_build_file_content() {
  do_new_remote_repo_test "build_file_content"
}

function test_new_remote_repo_with_workspace_file() {
  do_new_remote_repo_test "workspace_file"
}

function test_new_remote_repo_with_workspace_file_content() {
  do_new_remote_repo_test "workspace_file_content"
}

function do_new_remote_repo_test() {
  # Create a zipped-up repository HTTP response.
  local repo2=$TEST_TMPDIR/repo2
  rm -rf $repo2
  mkdir -p $repo2/fox
  cd $repo2
  local what_does_the_fox_say="Fraka-kaka-kaka-kaka-kow"
  echo $what_does_the_fox_say > fox/male
  local repo2_zip=$TEST_TMPDIR/fox.zip
  rm -f $repo2_zip
  zip -r $repo2_zip fox
  local sha256=$(sha256sum $repo2_zip | cut -f 1 -d ' ')
  serve_file $repo2_zip

  cd ${WORKSPACE_DIR}

  # Create the build file for the http archive based on the requested attr style.
  local build_file_attr=""
  local workspace_file_attr=""

  local build_file_content="
filegroup(
    name = \"fox\",
    srcs = [\"fox/male\"],
    visibility = [\"//visibility:public\"],
)
  "

  if [ "$1" = "build_file" ] ; then
    touch BUILD
    echo ${build_file_content} > fox.BUILD
    build_file_attr="build_file = '@//:fox.BUILD'"
  else
    build_file_attr="build_file_content=\"\"\"${build_file_content}\"\"\""
  fi

  if [ "$1" = "workspace_file" ]; then
    touch BUILD
    cat > fox.WORKSPACE <<EOF
workspace(name="endangered-fox")
EOF
    workspace_file_attr="workspace_file = '@//:fox.WORKSPACE'"
  elif [ "$1" = "workspace_file_content" ]; then
    workspace_file_attr="workspace_file_content = 'workspace(name=\"endangered-fox\")'"
  fi

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = 'endangered',
    url = 'http://127.0.0.1:$nc_port/repo.zip',
    sha256 = '$sha256',
    ${build_file_attr},
    ${workspace_file_attr}
)
EOF

  mkdir -p zoo
  cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["@endangered//:fox"],
)
EOF

  cat > zoo/female.sh <<EOF
#!/bin/sh
cat ../endangered/fox/male
EOF
  chmod +x zoo/female.sh

  bazel run //zoo:breeding-program >& $TEST_log \
    || echo "Expected build/run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

function test_fetch() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://127.0.0.1:$nc_port/',
    sha1 = '$sha1',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  output_base=$(bazel info output_base)
  external_dir=$output_base/external
  needle=endangered
  [[ -d $external_dir/$needle ]] \
      && fail "$needle already exists in $external_dir" || true
  bazel fetch //zoo:ball-pit >& $TEST_log || fail "Fetch failed"
  [[ $(ls $external_dir | grep $needle) ]] || fail "$needle not added to $external_dir"

  # Rerun fetch while nc isn't serving anything to make sure the fetched result
  # is cached.
  bazel fetch //zoo:ball-pit >& $TEST_log || fail "Incremental fetch failed"

  # Make sure fetch isn't needed after a bazel restart.
  bazel shutdown
  bazel build //zoo:ball-pit >& $TEST_log || fail "Fetch shouldn't be required"

  # But it is required after a clean.
  bazel clean --expunge || fail "Clean failed"
  bazel build --fetch=false //zoo:ball-pit >& $TEST_log && fail "Expected build to fail"
  expect_log "bazel fetch //..."
}

function test_prefix_stripping_tar_gz() {
  mkdir -p x/y/z
  echo "abc" > x/y/z/w
  tar czf x.tar.gz x
  local sha256=$(sha256sum x.tar.gz | cut -f 1 -d ' ')
  serve_file x.tar.gz

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.tar.gz",
    sha256 = "$sha256",
    strip_prefix = "x/y/z",
    build_file = "@//:x.BUILD",
)
EOF
  cat > x.BUILD <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w"],
)
EOF
  touch BUILD

  bazel build @x//:catter &> $TEST_log || fail "Build failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
}

function test_prefix_stripping_zip() {
  mkdir -p x/y/z
  echo "abc" > x/y/z/w
  zip -r x x
  local sha256=$(sha256sum x.zip | cut -f 1 -d ' ')
  serve_file x.zip

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.zip",
    sha256 = "$sha256",
    strip_prefix = "x/y/z",
    build_file = "@//:x.BUILD",
)
EOF
  cat > x.BUILD <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w"],
)
EOF
  touch BUILD

  bazel build @x//:catter &> $TEST_log || fail "Build failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
}

function test_prefix_stripping_existing_repo() {
  mkdir -p x/y/z
  touch x/y/z/WORKSPACE
  echo "abc" > x/y/z/w
  cat > x/y/z/BUILD <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w"],
)
EOF
  zip -r x x
  local sha256=$(sha256sum x.zip | cut -f 1 -d ' ')
  serve_file x.zip

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.zip",
    sha256 = "$sha256",
    strip_prefix = "x/y/z",
)
EOF

  bazel build @x//:catter &> $TEST_log || fail "Build failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
}

function test_moving_build_file() {
  echo "abc" > w
  tar czf x.tar.gz w
  local sha256=$(sha256sum x.tar.gz | cut -f 1 -d ' ')
  serve_file x.tar.gz

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.tar.gz",
    sha256 = "$sha256",
    build_file = "@//:x.BUILD",
)
EOF
  touch BUILD
  cat > x.BUILD <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w"],
)
EOF

  bazel build @x//:catter &> $TEST_log || fail "Build 1 failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
  mv x.BUILD x.BUILD.new || fail "Moving x.BUILD failed"
  sed 's/x.BUILD/x.BUILD.new/g' WORKSPACE > WORKSPACE.tmp || \
    fail "Editing WORKSPACE failed"
  mv WORKSPACE.tmp WORKSPACE
  serve_file x.tar.gz
  bazel build @x//:catter &> $TEST_log || fail "Build 2 failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
}

function test_changing_build_file() {
  echo "abc" > w
  echo "def" > w.new
  echo "I'm a build file" > BUILD
  tar czf x.tar.gz w w.new BUILD
  local sha256=$(sha256sum x.tar.gz | cut -f 1 -d ' ')
  serve_file x.tar.gz

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "x",
    url = "http://127.0.0.1:$nc_port/x.tar.gz",
    sha256 = "$sha256",
    build_file = "@//:x.BUILD",
)
EOF
  cat > x.BUILD <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w"],
)
EOF

  cat > x.BUILD.new <<EOF
genrule(
    name = "catter",
    cmd = "cat \$< > \$@",
    outs = ["catter.out"],
    srcs = ["w.new"],
)
EOF

  bazel build @x//:catter || fail "Build 1 failed"
  assert_contains "abc" bazel-genfiles/external/x/catter.out
  sed 's/x.BUILD/x.BUILD.new/g' WORKSPACE > WORKSPACE.tmp || \
    fail "Editing WORKSPACE failed"
  mv WORKSPACE.tmp WORKSPACE
  serve_file x.tar.gz
  bazel build @x//:catter &> $TEST_log || fail "Build 2 failed"
  assert_contains "def" bazel-genfiles/external/x/catter.out
}

function test_android_sdk_basic_load() {
  cat >> WORKSPACE <<'EOF' || fail "Couldn't cat"
android_sdk_repository(
    name = "androidsdk",
    path = "/fake/path",
    api_level = 23,
    build_tools_version="23.0.0"
)
EOF

  bazel query "//external:androidsdk" 2> "$TEST_log" > "$TEST_TMPDIR/queryout" \
      || fail "Expected success"
  cat "$TEST_TMPDIR/queryout" > "$TEST_log"
  expect_log "//external:androidsdk"
}

function test_use_bind_as_repository() {
  cat > WORKSPACE <<'EOF'
local_repository(name = 'foobar', path = 'foo')
bind(name = 'foo', actual = '@foobar//:test')
EOF
  mkdir foo
  touch foo/WORKSPACE
  touch foo/test
  echo 'exports_files(["test"])' > foo/BUILD
  cat > BUILD <<'EOF'
genrule(
    name = "foo",
    srcs = ["@foo//:test"],
    cmd = "echo $< | tee $@",
    outs = ["foo.txt"],
)
EOF
  bazel build :foo &> "$TEST_log" && fail "Expected failure" || true
  expect_log "no such package '@foo//'"
}

function test_flip_flopping() {
  REPO_PATH=$TEST_TMPDIR/repo
  mkdir -p "$REPO_PATH"
  cd "$REPO_PATH"
  touch WORKSPACE BUILD foo
  zip -r repo.zip *
  startup_server $PWD
  # Make the remote repo and local repo slightly different.
  rm foo
  touch bar
  cd -

  cat > local_ws <<EOF
local_repository(
    name = "repo",
    path = "$REPO_PATH",
)
EOF
  cat > remote_ws <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "repo",
    url = "http://127.0.0.1:$fileserver_port/repo.zip",
)
EOF
  external_dir=$(bazel info output_base)/external
  for i in $(seq 1 3); do
    cp local_ws WORKSPACE
    bazel build @repo//:all &> $TEST_log || fail "Build failed"
    test -L "$external_dir/repo" || fail "creating local symlink failed"
    test -a "$external_dir/repo/bar" || fail "bar not found"
    cp remote_ws WORKSPACE
    bazel build @repo//:all &> $TEST_log || fail "Build failed"
    test -d "$external_dir//repo" || fail "creating remote repo failed"
    test -a "$external_dir/repo/foo" || fail "foo not found"
  done

  shutdown_server
}

function test_sha256_weird() {
  REPO_PATH=$TEST_TMPDIR/repo
  mkdir -p "$REPO_PATH"
  cd "$REPO_PATH"
  touch WORKSPACE BUILD foo
  zip -r repo.zip *
  startup_server $PWD
  cd -

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "repo",
    sha256 = "a random string",
    url = "http://127.0.0.1:$fileserver_port/repo.zip",
)
EOF
  bazel build @repo//... &> $TEST_log && fail "Expected to fail"
  expect_log "[Ii]nvalid SHA256 checksum"
  shutdown_server
}

function test_sha256_incorrect() {
  REPO_PATH=$TEST_TMPDIR/repo
  mkdir -p "$REPO_PATH"
  cd "$REPO_PATH"
  touch WORKSPACE BUILD foo
  zip -r repo.zip *
  startup_server $PWD
  cd -

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "repo",
    sha256 = "61a6f762aaf60652cbf332879b8dcc2cfd81be2129a061da957d039eae77f0b0",
    url = "http://127.0.0.1:$fileserver_port/repo.zip",
)
EOF
  bazel build @repo//... &> $TEST_log 2>&1 && fail "Expected to fail"
  expect_log "Error downloading \\[http://127.0.0.1:$fileserver_port/repo.zip\\] to"
  expect_log "but wanted 61a6f762aaf60652cbf332879b8dcc2cfd81be2129a061da957d039eae77f0b0"
  shutdown_server
}



function test_same_name() {
  mkdir ext
  echo foo> ext/foo
  EXTREPODIR=`pwd`
  zip ext.zip ext/*
  rm -rf ext

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  url="file://${EXTREPODIR}/ext.zip",
  build_file_content="exports_files([\"foo\"])",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "localfoo",
  srcs = ["@ext//:foo"],
  outs = ["foo"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:localfoo \
    || fail 'Expected @ext//:foo and //:foo not to conflict'
}

function test_missing_build() {
  mkdir ext
  echo foo> ext/foo
  EXTREPODIR=`pwd`
  rm -f ext.zip
  zip ext.zip ext/*
  rm -rf ext

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${EXTREPODIR}/ext.zip"],
  build_file="@//:ext.BUILD",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "localfoo",
  srcs = ["@ext//:foo"],
  outs = ["foo"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:localfoo && fail 'Expected failure' || :

  cat > ext.BUILD <<'EOF'
exports_files(["foo"])
EOF

  bazel build //:localfoo || fail 'Expected success'

  # Changes to the BUILD file in the external repository should be tracked
  rm -f ext.BUILD && touch ext.BUILD

  bazel build //:localfoo && fail 'Expected failure' || :

  # Verify that we don't call out unconditionally
  cat > ext.BUILD <<'EOF'
exports_files(["foo"])
EOF
  bazel build //:localfoo || fail 'Expected success'
  rm -f "${EXTREPODIR}/ext.zip"
  bazel build //:localfoo || fail 'Expected success'
}


function test_inherit_build() {
  # Verify that http_archive can use a BUILD file shipped with the
  # external archive.
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="hello",
  outs=["hello.txt"],
  cmd="echo Hello World > $@",
)
EOF
  EXTREPODIR=`pwd`
  rm -f ext.zip
  zip ext.zip ext/*
  rm -rf ext

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${EXTREPODIR}/ext.zip"],
)
EOF

  bazel build '@ext//:hello' || fail "expected success"
}

function test_failing_fetch_with_keep_going() {
  touch WORKSPACE
  cat > BUILD <<'EOF'
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
    deps = ["@fake//:fake"],
)
EOF
  touch hello-world.cc

  bazel fetch --keep_going //... >& $TEST_log && fail "Expected to fail" || true
}


function test_query_cached() {
  # Verify that external repositories are cached after being used once.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
)
genrule(
  name="bar",
  outs=["bar.txt"],
  srcs=[":foo"],
  cmd="cp $< $@",
)
EOF
  EXTREPODIR=`pwd`
  rm -f ext.zip
  zip ext.zip ext/*
  rm -rf ext

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${EXTREPODIR}/ext.zip"],
)
EOF
  bazel build '@ext//:bar' || fail "expected success"

  # Simulate going offline by removing the external archive
  rm -f "${EXTREPODIR}/ext.zip"
  bazel query 'deps("@ext//:bar")' > "${TEST_log}" 2>&1 \
    || fail "expected success"
  expect_log '@ext//:foo'
  bazel shutdown
  bazel query 'deps("@ext//:bar")' > "${TEST_log}" 2>&1 \
    || fail "expected success"
  expect_log '@ext//:foo'
}

function test_repository_cache_relative_path() {
  # Verify that --repository_cache works for query and caches soly
  # based on the predicted hash, for a repository-cache location given as path
  # relative to the WORKSPACE
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
)
genrule(
  name="bar",
  outs=["bar.txt"],
  srcs=[":foo"],
  cmd="cp $< $@",
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  rm -rf cache
  mkdir cache

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${WRKDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  # Use the external repository once to make sure it is cached.
  bazel build --repository_cache="../cache" '@ext//:bar' \
      || fail "expected success"

  # Now "go offline" and clean local resources.
  rm -f "${WRKDIR}/ext.zip"
  bazel clean --expunge
  bazel query 'deps("@ext//:bar")' && fail "Couldn't clean local cache" || :

  # The value should still be available from the repository cache
  bazel query 'deps("@ext//:bar")' \
        --repository_cache="../cache" > "${TEST_log}" \
      || fail "Expected success"
  expect_log '@ext//:foo'

  # Clean again.
  bazel clean --expunge
  # Even with a different source URL, the cache should be consulted.

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/invalidpath/othername.zip"],
  sha256="${sha256}",
)
EOF
  bazel query 'deps("@ext//:bar")' \
        --repository_cache="../cache" > "${TEST_log}" \
      || fail "Expected success"
  expect_log '@ext//:foo'
}

test_default_cache()
{
  # Verify that the default cache works for query and caches soly
  # based on the predicted hash, for a repository-cache location given as path
  # relative to the WORKSPACE
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
)
genrule(
  name="bar",
  outs=["bar.txt"],
  srcs=[":foo"],
  cmd="cp $< $@",
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${WRKDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  # Use the external repository once to make sure it is cached.
  bazel build '@ext//:bar' || fail "expected success"

  # Now "go offline" and clean local resources.
  rm -f "${WRKDIR}/ext.zip"
  bazel clean --expunge

  # The value should still be available from the repository cache
  bazel query 'deps("@ext//:bar")' > "${TEST_log}" || fail "Expected success"
  expect_log '@ext//:foo'

  # Clean again.
  bazel clean --expunge
  # Even with a different source URL, the cache should be consulted.

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/invalidpath/othername.zip"],
  sha256="${sha256}",
)
EOF
  bazel query 'deps("@ext//:bar")' > "${TEST_log}" || fail "Expected success"
  expect_log '@ext//:foo'
}

function test_repository_cache_default() {
  # Verify that the repository cache is enabled by default.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  TOPDIR=`pwd`
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${TOPDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  # Use the external repository once to make sure it is cached.
  bazel build '@ext//:foo' || fail "expected success"

  # Now "go offline" and clean local resources.
  rm -f "${TOPDIR}/ext.zip"
  bazel clean --expunge

  # Still, the file should be cached.
  bazel build '@ext//:foo' || fail "expected success"
}

function test_cache_disable {
  # Verify that the repository cache can be disabled.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  TOPDIR=`pwd`
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${TOPDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  # Use `--repository_cache` with no path to explicitly disable repository cache
  bazel build --repository_cache= '@ext//:foo' || fail "expected success"

  # make sure, the empty path is not interpreted relative to `pwd`; i.e., we do
  # not expect any new directories generated in the workspace, in particular
  # none named conent_addressable, which is the directory where the cache puts
  # its artifacts into.
  ls -al | grep content_addressable \
      && fail "Should not interpret empty path as cache directly in the work space" || :

  # Now "go offline" and clean local resources.
  rm -f "${TOPDIR}/ext.zip"
  bazel clean --expunge

  # The build should fail since we are not using the repository cache, but the
  # original file can no longer be "downloaded".
  bazel build --repository_cache= '@ext//:foo' \
      && fail "Should fail for lack of fetchable faile" || :
}

function test_repository_cache() {
  # Verify that --repository_cache works for query and caches soly
  # based on the predicted hash.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
)
genrule(
  name="bar",
  outs=["bar.txt"],
  srcs=[":foo"],
  cmd="cp $< $@",
)
EOF
  TOPDIR=`pwd`
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  rm -rf cache
  mkdir cache

  rm -rf main
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${TOPDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  # Use the external repository once to make sure it is cached.
  bazel build --repository_cache="${TOPDIR}/cache}" '@ext//:bar' \
      || fail "expected success"

  # Now "go offline" and clean local resources.
  rm -f "${TOPDIR}/ext.zip"
  bazel clean --expunge
  bazel query 'deps("@ext//:bar")' && fail "Couldn't clean local cache" || :

  # The value should still be available from the repository cache
  bazel query 'deps("@ext//:bar")' \
        --repository_cache="${TOPDIR}/cache}" > "${TEST_log}" \
      || fail "Expected success"
  expect_log '@ext//:foo'

  # Clean again.
  bazel clean --expunge
  # Even with a different source URL, the cache should be consulted.

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/invalidpath/othername.zip"],
  sha256="${sha256}",
)
EOF
  bazel query 'deps("@ext//:bar")' \
        --repository_cache="${TOPDIR}/cache}" > "${TEST_log}" \
      || fail "Expected success"
  expect_log '@ext//:foo'
}

function test_distdir() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  mkdir distfiles
  mv ext.zip distfiles

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/outdatedpath/ext.zip"],
  sha256="${sha256}",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:foo"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel clean --expunge
  bazel build --distdir="${WRKDIR}/distfiles" //:local \
    || fail "expected success"
}

function test_distdir_relative_path() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  mkdir distfiles
  mv ext.zip distfiles

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/outdatedpath/ext.zip"],
  sha256="${sha256}",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:foo"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel clean --expunge
  bazel build --distdir="../distfiles" //:local \
    || fail "expected success"
}

function test_distdir_non_existent() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  touch thisisafile
  mkdir thisisempty

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${WRKDIR}/ext.zip"],
  sha256="${sha256}",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:foo"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel clean --expunge
  # The local distdirs all do no provide the file; still, it should work by fetching
  # the file from upstream.
  bazel build --distdir=does/not/exist --distdir=/global/does/not/exist --distdir=../thisisafile --distdir=../thisisempty //:local \
    || fail "expected success"
}

function test_good_symlinks() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir -p ext/subdir
  echo foo > ext/file.txt
  ln -s ../file.txt ext/subdir/symlink.txt
  ls -alR ext
  zip -r --symlinks ext.zip ext

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${WRKDIR}/ext.zip"],
  build_file="@//:ext.BUILD"
)
EOF
  cat > ext.BUILD <<'EOF'
exports_files(["subdir/symlink.txt"])
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:subdir/symlink.txt"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:local || fail "Expected success"
}

function test_distdir_option_not_sticky() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  mkdir ext
  cat > ext/BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd="echo Hello World > $@",
  visibility = ["//visibility:public"],
)
EOF
  zip ext.zip ext/*
  rm -rf ext
  sha256=$(sha256sum ext.zip | head -c 64)

  mkdir distfiles
  mv ext.zip distfiles

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["http://doesnotexist.example.com/outdatedpath/ext.zip"],
  sha256="${sha256}",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:foo"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)

genrule(
  name = "unrelated",
  outs = ["unrelated.txt"],
  cmd = "echo Something > $@",
)
EOF

  bazel clean --expunge
  bazel build --distdir="../distfiles" //:unrelated \
    || fail "expected success"
  # As no --distdir option is given and upstream not available,
  # we expect the build to fail
  bazel build //:local && fail "Expected failure" || :
}

function test_bad_symlinks() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir -p ext/subdir
  echo foo > ext/file.txt
  ln -s ../file.txt ext/symlink.txt
  ls -alR ext
  zip -r --symlinks ext.zip ext

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${WRKDIR}/ext.zip"],
  build_file="@//:ext.BUILD"
)
EOF
  cat > ext.BUILD <<'EOF'
exports_files(["file.txt"])
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@ext//:file.txt"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:local \
    && fail "Expected failure due to unsupported symlink" || :
}

function test_progress_reporting() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.report_progress("First action")
  ctx.execute(["/bin/sh", "-c", "sleep 5"])
  ctx.report_progress("Second action")
  ctx.execute(["/bin/sh", "-c", "sleep 5"])
  ctx.report_progress("Actual files")
  ctx.file("data", "Hello world")
  ctx.file("BUILD", "exports_files(['data'])")

with_progress = repository_rule(
  implementation = _rule_impl,
  attrs = {},
)
EOF

  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "with_progress")
with_progress(name="foo")
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs = ["@foo//:data"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build --curses=yes //:local > "${TEST_log}" 2>&1 \
      || fail "exepected succes"
  expect_log "foo.*First action"
  expect_log "foo.*Second action"
}

function test_progress_reporting() {
  # Isse 7353 requested that even in the case of a syntactically invalid
  # checksum, the file still should be fetched and its checksum computed.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  touch ext.tar

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "ext",
  urls = ["file://${WRKDIR}/ext.tar"],
  sha256 = "badargument",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//:in.txt"],
  outs = ["out.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:it > "${TEST_log}" 2>&1 && fail "Expected failure" || :

  expect_log '@ext.*badargument'
  expect_log 'SHA256 (.*/ext.tar) = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
}

run_suite "external tests"
