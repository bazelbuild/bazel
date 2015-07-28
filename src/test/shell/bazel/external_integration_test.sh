#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function set_up() {
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

case "${PLATFORM}" in
  darwin)
    function nc_l() {
      nc -l $1
    }
    ;;
  *)
    function nc_l() {
      nc -l -p $1 -q 1
    }
    ;;
esac

# Serves $1 as a file on localhost:$nc_port.  Sets the following variables:
#   * nc_port - the port nc is listening on.
#   * nc_log - the path to nc's log.
#   * nc_pid - the PID of nc.
#   * http_response - the full response nc will provide to a request.
# This also creates the file $TEST_TMPDIR/http_response.
function serve_file() {
  http_response=$TEST_TMPDIR/http_response
  cat > $http_response <<EOF
HTTP/1.0 200 OK

EOF
  cat $1 >> $http_response
  # Assign random_port to nc_port if not already set.
  echo ${nc_port:=$(pick_random_unused_tcp_port)} > /dev/null
  nc_log=$TEST_TMPDIR/nc.log
  nc_l $nc_port < $http_response >& $nc_log &
  nc_pid=$!
}

# Creates a jar carnivore.Mongoose and serves it using serve_file.
function serve_jar() {
  pkg_dir=$TEST_TMPDIR/carnivore
  if [ -e "$pkg_dir" ]; then
    rm -fr $pkg_dir
  fi

  mkdir $pkg_dir
  cat > $pkg_dir/Mongoose.java <<EOF
package carnivore;
public class Mongoose {
    public static void frolic() {
        System.out.println("Tra-la!");
    }
}
EOF
  ${bazel_javabase}/bin/javac $pkg_dir/Mongoose.java
  test_jar=$TEST_TMPDIR/libcarnivore.jar
  cd ${TEST_TMPDIR}
  ${bazel_javabase}/bin/jar cf $test_jar carnivore/Mongoose.class

  sha256=$(sha256sum $test_jar | cut -f 1 -d ' ')
  # OS X doesn't have sha1sum, so use openssl.
  sha1=$(openssl sha1 $test_jar | cut -f 2 -d ' ')
  serve_file $test_jar
  cd ${WORKSPACE_DIR}
}

function kill_nc() {
  # Try to kill nc, otherwise the test will time out if Bazel has a bug and
  # didn't make a request to it.
  kill $nc_pid || true  # kill can fails if the process already finished
  [ -z "${nc_log:-}" ] || cat $nc_log
}

function zip_up() {
  repo2_zip=$TEST_TMPDIR/fox.zip
  zip -0 -r $repo2_zip WORKSPACE fox
}

function tar_gz_up() {
  repo2_zip=$TEST_TMPDIR/fox.tar.gz
  tar czf $repo2_zip WORKSPACE fox
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
function http_archive_helper() {
  zipper=$1
  local write_workspace
  [[ $# -gt 1 ]] && [[ "$2" = "nowrite" ]] && write_workspace=1 || write_workspace=0

  if [[ $write_workspace = 0 ]]; then
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
./external/endangered/fox/male
EOF
    chmod +x zoo/female.sh
fi

  bazel run //zoo:breeding-program >& $TEST_log --show_progress_rate_limit=0 \
    || echo "Expected build/run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

function test_http_archive_zip() {
  http_archive_helper zip_up

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

function test_http_archive_no_server() {
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  cat > WORKSPACE <<EOF
http_archive(name = 'endangered', url = 'http://localhost:$nc_port/repo.zip',
    sha256 = 'dummy')
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
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel fetch //zoo:breeding-program >& $TEST_log && fail "Expected fetch to fail"
  expect_log "Connection refused"
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
http_archive(name = 'endangered', url = 'http://localhost:$nc_port/repo.zip',
    sha256 = '$wrong_sha256')
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
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel fetch //zoo:breeding-program >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "does not match expected SHA-256"
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

function test_changed_zip() {
  nc_port=$(pick_random_unused_tcp_port) || fail "Couldn't get TCP port"
  http_archive_helper zip_up
  http_archive_helper zip_up "nowrite"
  expect_not_log "Downloading from"
  local readonly output_base=$(bazel info output_base)
  local readonly repo_zip=$output_base/external/endangered/fox.zip
  rm $repo_zip || fail "Couldn't delete $repo_zip"
  touch $repo_zip || fail "Couldn't touch $repo_zip"
  [[ -s $repo_zip ]] && fail "File size not 0"
  http_archive_helper zip_up "nowrite"
  expect_log "Downloading from"
  [[ -s $repo_zip ]] || fail "File size was 0"
}

# Tests downloading a jar and using it as a Java dependency.
function test_jar_download() {
  serve_jar

  cat > WORKSPACE <<EOF
http_jar(name = 'endangered', url = 'http://localhost:$nc_port/lib.jar',
    sha256 = '$sha256')
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

  bazel fetch //zoo:ball-pit || fail "Fetch failed"
  bazel run //zoo:ball-pit >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

# Tests downloading a file and using it as a dependency.
function test_http_download() {
  local test_file=$TEST_TMPDIR/toto
  echo "Tra-la!" >$test_file
  local sha256=$(sha256sum $test_file | cut -f 1 -d ' ')
  serve_file $test_file
  cd ${WORKSPACE_DIR}

  cat > WORKSPACE <<EOF
http_file(name = 'toto', url = 'http://localhost:$nc_port/toto',
    sha256 = '$sha256')
EOF

  mkdir -p test
  cat > test/BUILD <<EOF
sh_binary(
    name = "test",
    srcs = ["test.sh"],
    data = ["@toto//file"],
)
EOF

  cat > test/test.sh <<EOF
#!/bin/bash
cat external/toto/file/toto
EOF

  chmod +x test/test.sh
  bazel fetch //test || fail "Fetch failed"
  bazel run //test >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

function test_invalid_rule() {
  # http_jar with missing URL field.
  cat > WORKSPACE <<EOF
http_jar(name = 'endangered', sha256 = 'dummy')
EOF

  bazel fetch //external:endangered >& $TEST_log && fail "Expected fetch to fail"
  expect_log "missing value for mandatory attribute 'url' in 'http_jar' rule"
}

function test_maven_jar() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
    sha1 = '$sha1',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit || fail "Fetch failed"
  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  kill_nc
  assert_contains "GET /com/example/carnivore/carnivore/1.23/carnivore-1.23.jar" $nc_log
  expect_log "Tra-la!"
}

# Same as test_maven_jar, except omit sha1 implying "we don't care".
function test_maven_jar_no_sha1() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit || fail "Fetch failed"
  bazel run //zoo:ball-pit >& $TEST_log || fail "Expected run to succeed"
  kill_nc
  assert_contains "GET /com/example/carnivore/carnivore/1.23/carnivore-1.23.jar" $nc_log
  expect_log "Tra-la!"
}

function test_maven_jar_404() {
  http_response=$TEST_TMPDIR/http_response
  cat > $http_response <<EOF
HTTP/1.0 404 Not Found

EOF
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  nc_l $nc_port < $http_response &
  nc_pid=$!
  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency: Could not find artifact"
}

function test_maven_jar_mismatched_sha1() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    artifact = "com.example.carnivore:carnivore:1.23",
    repository = 'http://localhost:$nc_port/',
    sha1 = '$sha256',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel fetch //zoo:ball-pit >& $TEST_log && echo "Expected fetch to fail"
  kill_nc
  expect_log "has SHA-1 of $sha1, does not match expected SHA-1 ($sha256)"
}

function test_new_remote_repo() {
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
  cat > fox.BUILD <<EOF
filegroup(
    name = "fox",
    srcs = ["fox/male"],
    visibility = ["//visibility:public"],
)
EOF

  cat > WORKSPACE <<EOF
new_http_archive(
    name = 'endangered',
    url = 'http://localhost:$nc_port/repo.zip',
    sha256 = '$sha256',
    build_file = 'fox.BUILD'
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
#!/bin/bash
cat external/endangered/fox/male
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
    repository = 'http://localhost:$nc_port/',
    sha1 = '$sha1',
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  output_base=$(bazel info output_base)
  external_dir=$output_base/external
  needle=endangered
  [[ $(ls $external_dir | grep $needle) ]] && fail "$needle already in $external_dir"
  bazel fetch //zoo:ball-pit >& $TEST_log || fail "Fetch failed"
  [[ $(ls $external_dir | grep $needle) ]] || fail "$needle not added to $external_dir"

  # Rerun fetch while nc isn't serving anything to make sure the fetched result
  # is cached.
  bazel fetch //zoo:ball-pit >& $TEST_log || fail "Incremental fetch failed"

  # Make sure fetch isn't needed after a bazel restart.
  bazel shutdown
  bazel build //zoo:ball-pit >& $TEST_log || fail "Fetch shouldn't be required"

  # But it is required after a clean.
  bazel clean --expunge
  # TODO(kchodorow): remove the --fetch=false option once that's the default.
  bazel build --fetch=false //zoo:ball-pit >& $TEST_log && fail "Expected build to fail"
  expect_log "bazel fetch //..."
}

run_suite "external tests"
