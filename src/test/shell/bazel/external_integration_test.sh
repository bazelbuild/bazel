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
  bazel clean --expunge
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
  nc_port=$(pick_random_unused_tcp_port) || exit 1
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
  serve_file $test_jar
  cd ${WORKSPACE_DIR}
}

function kill_nc() {
  # Try to kill nc, otherwise the test will time out if Bazel has a bug and
  # didn't make a request to it.
  kill $nc_pid || true  # kill can fails if the process already finished
  [ -z "${nc_log:-}" ] || cat $nc_log
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
function test_http_archive() {
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
  echo $what_does_the_fox_say > fox/male
  # Add some padding to the .zip to test that Bazel's download logic can
  # handle breaking a response into chunks.
  dd if=/dev/zero of=fox/padding bs=1024 count=10240
  repo2_zip=$TEST_TMPDIR/fox.zip
  zip -0 -r $repo2_zip WORKSPACE fox
  sha256=$(sha256sum $repo2_zip | cut -f 1 -d ' ')
  serve_file $repo2_zip

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
http_archive(name = 'endangered', url = 'http://localhost:$nc_port/repo.zip',
    sha256 = '$sha256')
bind(name = 'stud', actual = '@endangered//fox')
EOF

  cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["//external:stud"],
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

function test_http_archive_no_server() {
  nc_port=$(pick_random_unused_tcp_port) || exit 1
  cat > WORKSPACE <<EOF
http_archive(name = 'endangered', url = 'http://localhost:$nc_port/repo.zip',
    sha256 = 'dummy')
bind(name = 'stud', actual = '@endangered//fox')
EOF

  cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["//external:stud"],
)
EOF

  cat > zoo/female.sh <<EOF
#!/bin/bash
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel run //zoo:breeding-program >& $TEST_log && echo "Expected build to fail"
  cat $TEST_log
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
bind(name = 'stud', actual = '@endangered//fox')
EOF

  cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["//external:stud"],
)
EOF

  cat > zoo/female.sh <<EOF
#!/bin/bash
cat fox/male
EOF
  chmod +x zoo/female.sh

  bazel run //zoo:breeding-program >& $TEST_log && echo "Expected build to fail"
  kill_nc
  expect_log "does not match expected SHA-256"
}

# Bazel should not re-download the .zip unless the user requests it or the
# WORKSPACE file changes, so doing a BUILD where the "wrong" .zip is available
# on the server should work if the correct .zip is already available.
function test_sha256_caching() {
  # Download with correct sha256.
  test_http_archive

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

  bazel run //zoo:breeding-program >& $TEST_log \
    || echo "Expected run to succeed"
  kill_nc
  expect_log $what_does_the_fox_say
}

# Tests downloading a jar and using it as a Java dependency.
function test_jar_download() {
  serve_jar

  cat > WORKSPACE <<EOF
http_jar(name = 'endangered', url = 'http://localhost:$nc_port/lib.jar',
    sha256 = '$sha256')
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

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

  bazel run //zoo:ball-pit >& $TEST_log || echo "Expected run to succeed"
  kill_nc
  expect_log "Tra-la!"
}

function test_invalid_rule() {
  # http_jar with missing URL field.
  cat > WORKSPACE <<EOF
http_jar(name = 'endangered', sha256 = 'dummy')
EOF

  bazel run //external:endangered >& $TEST_log && echo "Expected run to fail"
  expect_log "missing value for mandatory attribute 'url' in 'http_jar' rule"
}

function test_maven_jar() {
  serve_jar

  cat > WORKSPACE <<EOF
maven_jar(
    name = 'endangered',
    group_id = "com.example.carnivore",
    artifact_id = "carnivore",
    version = "1.23",
    repositories = ['http://localhost:$nc_port/']
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log || echo "Expected run to succeed"
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
    group_id = "carnivore",
    artifact_id = "carnivore",
    version = "1.23",
    repositories = ['http://localhost:$nc_port/']
)
bind(name = 'mongoose', actual = '@endangered//jar')
EOF

  bazel run //zoo:ball-pit >& $TEST_log && echo "Expected run to fail"
  kill_nc
  expect_log "Failed to fetch Maven dependency: Could not find artifact"
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
  rm $repo2_zip
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
bind(name = 'stud', actual = '@endangered//:fox')
EOF

  mkdir -p zoo
  cat > zoo/BUILD <<EOF
sh_binary(
    name = "breeding-program",
    srcs = ["female.sh"],
    data = ["//external:stud"],
)
EOF

  cat > zoo/female.sh <<EOF
#!/bin/bash
cat external/endangered/fox/male
EOF
  chmod +x zoo/female.sh

  bazel clean --expunge
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
    group_id = "com.example.carnivore",
    artifact_id = "carnivore",
    version = "1.23",
    repositories = ['http://localhost:$nc_port/']
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
}

run_suite "external tests"
