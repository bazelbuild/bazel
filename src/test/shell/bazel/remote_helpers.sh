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

set -eu

setup_localjdk_javabase

# Serves $1 as a file on localhost:$nc_port.  Sets the following variables:
#   * nc_port - the port nc is listening on.
#   * nc_log - the path to nc's log.
#   * nc_pid - the PID of nc.
function serve_file() {
  file_name=served_file.$$
  cat $1 > "${TEST_TMPDIR}/$file_name"
  nc_log="${TEST_TMPDIR}/nc.log"
  rm -f $nc_log
  touch $nc_log
  cd "${TEST_TMPDIR}"
  port_file=server-port.$$
  rm -f $port_file
  python3 $python_server always $file_name > $port_file &
  nc_pid=$!
  while ! grep started $port_file; do sleep 1; done
  nc_port=$(head -n 1 $port_file)
  fileserver_port=$nc_port
  wait_for_server_startup
  cd -
}

# Serves $1 as a file on localhost:$nc_port expecting authentication.
#   * nc_port - the port nc is listening on.
#   * nc_log - the path to nc's log.
#   * nc_pid - the PID of nc.
function serve_file_auth() {
  file_name=served_file.$$
  cat $1 > "${TEST_TMPDIR}/$file_name"
  nc_log="${TEST_TMPDIR}/nc.log"
  rm -f $nc_log
  touch $nc_log
  cd "${TEST_TMPDIR}"
  port_file=server-port.$$
  rm -f $port_file
  python3 $python_server auth $file_name > $port_file &
  nc_pid=$!
  while ! grep started $port_file; do sleep 1; done
  nc_port=$(head -n 1 $port_file)
  fileserver_port=$nc_port
  wait_for_server_startup
  cd -
}

# Creates a jar carnivore.Mongoose and serves it using serve_file.
function serve_jar() {
  make_test_jar
  serve_file $test_jar
  cd ${WORKSPACE_DIR}
}

function make_test_jar() {
  pkg_dir=$TEST_TMPDIR/carnivore
  rm -fr $pkg_dir
  mkdir -p $pkg_dir
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
  test_srcjar=$TEST_TMPDIR/libcarnivore-sources.jar
  cd ${TEST_TMPDIR}
  ${bazel_javabase}/bin/jar cf $test_jar carnivore/Mongoose.class
  ${bazel_javabase}/bin/jar cf $test_srcjar carnivore/Mongoose.java
  sha256=$(sha256sum $test_jar | cut -f 1 -d ' ')
  sha256_src=$(sha256sum $test_srcjar | cut -f 1 -d ' ')
  # OS X doesn't have sha1sum, so use openssl.
  sha1=$(openssl sha1 $test_jar | cut -f 2 -d ' ')
  sha1_src=$(openssl sha1 $test_srcjar | cut -f 2 -d ' ')
  cd -
}

function make_test_aar() {
  test_aar=${TEST_TMPDIR}/example.aar
  cd ${TEST_TMPDIR}
  cat > AndroidManifest.xml <<EOF
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
    <uses-sdk
        android:minSdkVersion="9"
        android:targetSdkVersion="20" />
    <application />
</manifest>
EOF
  mkdir -p com/herbivore
  cat > com/herbivore/Stegosaurus.java <<EOF
package com.herbivore;
class Stegosaurus {}
EOF
  ${bazel_javabase}/bin/javac -source 7 -target 7 com/herbivore/Stegosaurus.java
  ${bazel_javabase}/bin/jar cf0 classes.jar com/herbivore/Stegosaurus.class
  mkdir -p res/layout
  cat > res/layout/my_view.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout />
EOF
  zip -0 $test_aar AndroidManifest.xml classes.jar res/layout/my_view.xml
  sha256=$(sha256sum $test_aar | cut -f 1 -d ' ')
  # OS X doesn't have sha1sum, so use openssl.
  sha1=$(openssl sha1 $test_aar | cut -f 2 -d ' ')
  cd -
}

# Serves a redirection from localhost:$redirect_port to $1. Sets the following variables:
#   * redirect_port - the port nc is listening on.
#   * redirect_log - the path to nc's log.
#   * redirect_pid - the PID of nc.
function serve_redirect() {
  redirect_log="${TEST_TMPDIR}/redirect.log"
  rm -f $redirect_log
  touch $redirect_log
  cd "${TEST_TMPDIR}"
  port_file=server-port.$$
  # While we "own" the port_file for the life time of this process, there can
  # be a left-over file from a previous process that had the process id (there
  # are not that many possible process ids after all) or even the same process
  # having started and shut down a server for a different test case in the same
  # shard. So we have to remove any left-over file in order to not exit the
  # while loop below too early because of finding the string "started" in the
  # old file (and thus potentially even getting an outdated port information).
  rm -f $port_file
  python3 $python_server redirect $1 > $port_file &
  redirect_pid=$!
  while ! grep started $port_file; do sleep 1; done
  redirect_port=$(head -n 1 $port_file)
  fileserver_port=$redirect_port
  wait_for_server_startup
  cd -
}

# Serves a HTTP 404 Not Found response with an optional parameter for the
# response body.
function serve_not_found() {
  port_file=server-port.$$
  cd "${TEST_TMPDIR}"
  rm -f $port_file
  python3 $python_server 404 > $port_file &
  nc_pid=$!
  while ! grep started $port_file; do sleep 1; done
  nc_port=$(head -n 1 $port_file)
  fileserver_port=$nc_port
  wait_for_server_startup
  cd -
}

# Simulates a server timing out while trying to generate a response.
function serve_timeout() {
  port_file=server-port.$$
  cd "${TEST_TMPDIR}"
  rm -f $port_file
  python3 $python_server timeout  > $port_file &
  nc_pid=$!
  while ! grep started $port_file; do sleep 1; done
  nc_port=$(head -n 1 $port_file)
  fileserver_port=$nc_port
  cd -
}

# Serves a HTTP 200 Ok response with headers dumped into the file
# Args:
#  $1: required; path to the file
#  $2: optional; path to the file where headers will be written to.
function serve_file_header_dump() {
  file_name=served_file.$$
  cat $1 > "${TEST_TMPDIR}/$file_name"
  nc_log="${TEST_TMPDIR}/nc.log"
  rm -f $nc_log
  touch $nc_log
  cd "${TEST_TMPDIR}"
  port_file=server-port.$$
  rm -f $port_file
  python3 $python_server always $file_name --dump_headers ${2:-"headers.json"} > $port_file &
  nc_pid=$!
  while ! grep started $port_file; do sleep 1; done
  nc_port=$(head -n 1 $port_file)
  fileserver_port=$nc_port
  wait_for_server_startup
  cd -
}

# Waits for the SimpleHTTPServer to actually start up before the test is run.
# Otherwise the entire test can run before the server starts listening for
# connections, which causes flakes.
function wait_for_server_startup() {
  touch some-file
  while ! curl http://localhost:$fileserver_port/some-file > /dev/null; do
    echo "waiting for server, exit code: $?"
    sleep 1
  done
  echo "done waiting for server, exit code: $?"
  rm some-file
}


function create_artifact() {
  local group_id=$1
  local artifact_id=$2
  local version=$3
  local packaging=${4:-jar}
  # TODO(davido): This is unused for now.
  # Finalize the implementation once the underlying tests are fixed.
  local classifier=${5:-jar}
  if [ $packaging == "aar" ]; then
    make_test_aar
    local artifact=$test_aar
  else
    make_test_jar
    local artifact=$test_jar
    local srcjar_artifact=$test_srcjar
  fi
  maven_path=$PWD/$(echo $group_id | sed 's/\./\//g')/$artifact_id/$version
  mkdir -p $maven_path
  openssl sha1 $artifact > $maven_path/$artifact_id-$version.$packaging.sha1
  mv $artifact $maven_path/$artifact_id-$version.$packaging

  # srcjar_artifact is not created for AARs.
  if [ ! -z "${srcjar_artifact+x}" ]; then
    openssl sha1 $srcjar_artifact > $maven_path/$artifact_id-$version-sources.$packaging.sha1
    mv $srcjar_artifact $maven_path/$artifact_id-$version-sources.$packaging
  fi
}

function serve_artifact() {
  startup_server $PWD
  create_artifact $1 $2 $3 ${4:-jar}
}

function startup_server() {
  fileserver_root=$1
  cd $fileserver_root
  port_file=server-port.$$
  rm -f $port_file
  python3 $python_server > $port_file &
  fileserver_pid=$!
  while ! grep started $port_file; do sleep 1; done
  fileserver_port=$(head -n 1 $port_file)
  wait_for_server_startup
  cd -
}

function startup_auth_server() {
  port_file=server-port.$$
  rm -f $port_file
  python3 $python_server auth > $port_file &
  fileserver_pid=$!
  while ! grep started $port_file; do sleep 1; done
  fileserver_port=$(head -n 1 $port_file)
  wait_for_server_startup
}

function shutdown_server() {
  # Try to kill nc, otherwise the test will time out if Bazel has a bug and
  # didn't make a request to it.
  [ -z "${fileserver_pid:-}" ] || kill $fileserver_pid || true
  [ -z "${redirect_pid:-}" ] || kill $redirect_pid || true
  [ -z "${nc_pid:-}" ] || kill $nc_pid || true
  [ -z "${nc_log:-}" ] || cat $nc_log
  [ -z "${redirect_log:-}" ] || cat $redirect_log
}

function kill_nc() {
  shutdown_server
}

# Sets up a credential helper binary at ${TEST_TMPDIR}/credhelper and resets
# the call counter.
function setup_credential_helper() {
  # Each call atomically writes one byte to this file.
  # The file can be read later determine how many calls were made.
  cat > "${TEST_TMPDIR}/credhelper.callcount_${TEST_SHARD_INDEX}"

  cat > "${TEST_TMPDIR}/credhelper" <<'EOF'
#!/usr/bin/env python3
import json
import os
import sys

# Neither count nor add headers to requests to the BCR.
uri = json.load(sys.stdin)["uri"]
if uri.startswith("https://bcr.bazel.build/"):
  print("{}")
  sys.exit(0)

path = os.path.join(os.environ["TEST_TMPDIR"], "credhelper.callcount_" + os.environ["TEST_SHARD_INDEX"])
fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_APPEND)
os.write(fd, b"1")
os.close(fd)

# Must match //src/test/shell/bazel/testing_server.py.
print("""{"headers":{"Authorization":["Bearer TOKEN"]}}""")
EOF
  chmod +x "${TEST_TMPDIR}/credhelper"
}

# Asserts how many times the credential helper was called.
function expect_credential_helper_calls() {
  local -r expected=$1
  local -r actual=$(wc -c "${TEST_TMPDIR}/credhelper.callcount_${TEST_SHARD_INDEX}" | awk '{print $1}')
  if [[ "$expected" != "$actual" ]]; then
    fail "expected $expected instead of $actual credential helper calls"
  fi
}
