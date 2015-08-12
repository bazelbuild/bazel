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

# Serves a redirection from localhost:$redirect_port to $1. Sets the following variables:
#   * redirect_port - the port nc is listening on.
#   * redirect_log - the path to nc's log.
#   * redirect_pid - the PID of nc.
function serve_redirect() {
  # Assign random_port to nc_port if not already set.
  echo ${redirect_port:=$(pick_random_unused_tcp_port)} > /dev/null
  redirect_log=$TEST_TMPDIR/redirect.log
  local response=$(cat <<EOF
HTTP/1.0 301 Moved Permanently
Location: $1

EOF
)
  nc_l $redirect_port >& $redirect_log <<<"$response" &
  redirect_pid=$!
}

function kill_nc() {
  # Try to kill nc, otherwise the test will time out if Bazel has a bug and
  # didn't make a request to it.
  kill $nc_pid || true  # kill can fails if the process already finished
  [ -z "${redirect_pid:-}" ] || kill $redirect_pid || true
  [ -z "${nc_log:-}" ] || cat $nc_log
  [ -z "${redirect_log:-}" ] || cat $redirect_log
}
