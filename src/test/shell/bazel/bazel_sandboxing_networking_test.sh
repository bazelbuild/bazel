#!/usr/bin/env bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# Test sandboxing spawn strategy
#

# Set to a host:port address that is outside of the local machine to
# test remote network sandboxing features.
#
# Can be passed in via --test_env=REMOTE_NETWORK_ADDRESS=host:port.
: "${REMOTE_NETWORK_ADDRESS:=}"

# Load test environment
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/../sandboxing_test_utils.sh \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/remote_helpers.sh \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

function set_up() {
  add_to_bazelrc "build --spawn_strategy=sandboxed"
  add_to_bazelrc "build --genrule_strategy=sandboxed"
  # Allow the network socket to be seen in the sandbox.
  add_to_bazelrc "build --sandbox_add_mount_pair=/tmp"

  sed -i.bak '/sandbox_tmpfs_path/d' $TEST_TMPDIR/bazelrc
}

# Prepares common targets and services to be used by all network-related
# tests.  The tests for remote network access are only enabled if the
# user has requested them by setting REMOTE_NETWORK_ADDRESS in the
# environment.
function setup_network_tests() {
  local tags="${1}"; shift
  echo 'stuff to serve' > file_to_serve

  serve_file file_to_serve

  local socket_dir
  socket_dir="$(mktemp -d /tmp/test.XXXXXX)" || fail "mktemp failed"
  local socket="${socket_dir}/socket"
  python3 $python_server --unix_socket="${socket}" always file_to_serve &
  local pid="${!}"

  trap "kill_nc || true; kill '${pid}' || true; rm -f '${socket}'; rmdir '${socket_dir}'" EXIT

  mkdir pkg
  cat <<EOF >pkg/BUILD
genrule(
  name = "localhost",
  outs = [ "localhost.txt" ],
  cmd = "curl -fo \$@ localhost:${nc_port}",
  tags = [ ${tags} ],
)

genrule(
  name = "unix-socket",
  outs = [ "unix-socket.txt" ],
  cmd = "curl --unix-socket ${socket} -fo \$@ irrelevant-url",
  tags = [ ${tags} ],
)

genrule(
  name = "loopback",
  outs = [ "loopback.txt" ],
  cmd = "python3 $python_server always $(pwd)/file_to_serve >port.txt & "
      + "pid=\$\$!; "
      + "while ! grep started port.txt; do sleep 1; done; "
      + "port=\$\$(head -n 1 port.txt); "
      + "curl -fo \$@ localhost:\$\$port; "
      + "kill \$\$pid",
)
EOF

  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    local hostname="${REMOTE_NETWORK_ADDRESS%:*}"
    local remote_ip
    if which host 2>/dev/null; then
      remote_ip="$(host -t A "${hostname}" | head -n 1 | awk '{print $4}')"
    elif which dig 2>/dev/null; then
      remote_ip="$(dig -t A "${hostname}" | grep "^${hostname}" | awk '{print $5}')"
    else
      fail "Don't know how to query IP of remote host ${hostname}"
    fi
    if [[ -z "${remote_ip}" ]]; then
      fail "No IPv4 connectivity within unsandboxed test"
    fi

    cat <<EOF >>pkg/BUILD
genrule(
  name = "remote-ip",
  outs = [ "remote-ip.txt" ],
  cmd = "curl -fo \$@ ${remote_ip}:80",
  tags = [ ${tags} ],
)

genrule(
  name = "remote-name",
  outs = [ "remote-name.txt" ],
  cmd = "curl -fo \$@ '${REMOTE_NETWORK_ADDRESS}'",
  tags = [ ${tags} ],
)
EOF
  else
    echo "Not registering tests for remote network sandboxing;" \
      "REMOTE_NETWORK_ADDRESS has not been set"
  fi
}

# Checks that the given target name, which must have been created by
# a previous call to setup_network_tests, can access the network.
function check_network_ok() {
  local target="${1}"; shift

  (
    # macOS's /bin/bash is ancient and cannot reference $@ when -u is set.
    # https://unix.stackexchange.com/questions/16560/bash-su-unbound-variable-with-set-u
    set +u

    bazel build "${@}" "pkg:${target}" &>$TEST_log \
      || fail "'${target}' could not access the network"
  )
}

# Checks that the given target name, which must have been created by
# a previous call to setup_network_tests, cannot access the network.
function check_network_not_ok() {
  local target="${1}"; shift

  (
    # macOS's /bin/bash is ancient and cannot reference $@ when -u is set.
    # https://unix.stackexchange.com/questions/16560/bash-su-unbound-variable-with-set-u
    set +u

    bazel build "${@}" "pkg:${target}" &> $TEST_log \
      && fail "'${target}' trying to use network succeeded but should have failed" || true
  )
  [[ ! -f "bazel-genfiles/pkg/${target}.txt" ]] \
    || fail "'${target}' produced output but was expected to fail"
}

function test_sandbox_network_access() {
  setup_network_tests '"some-tag"'

  check_network_ok localhost
  check_network_ok unix-socket
  check_network_ok loopback
  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    check_network_ok remote-ip
    check_network_ok remote-name
  fi
}

function test_sandbox_block_network_access() {
  setup_network_tests '"some-tag"'

  case "$(uname -s)" in
    Linux)
      # TODO(jmmv): The linux-sandbox claims to allow localhost connectivity
      # within the network namespace... but that doesn't seem to be the case.
      check_network_not_ok localhost --experimental_sandbox_default_allow_network=false
      ;;

    *)
      check_network_ok localhost --experimental_sandbox_default_allow_network=false
      ;;
  esac
  check_network_ok unix-socket --experimental_sandbox_default_allow_network=false
  check_network_ok loopback --experimental_sandbox_default_allow_network=false
  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    check_network_not_ok remote-ip --experimental_sandbox_default_allow_network=false
    check_network_not_ok remote-name --experimental_sandbox_default_allow_network=false
  fi
}

function test_sandbox_network_access_with_local() {
  cat >>$TEST_TMPDIR/bazelrc <<'EOF'
# With `--incompatible_legacy_local_fallback` turned off, we need to explicitly
# include a non-sandboxed strategy.
build --spawn_strategy=sandboxed,standalone --genrule_strategy=sandboxed,standalone
EOF

  setup_network_tests '"local"'

  check_network_ok localhost
  check_network_ok unix-socket
  check_network_ok loopback
  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    check_network_ok remote-ip
    check_network_ok remote-name
  fi
}

function test_sandbox_network_access_with_requires_network() {
  setup_network_tests '"requires-network"'

  check_network_ok localhost --experimental_sandbox_default_allow_network=false
  check_network_ok unix-socket --experimental_sandbox_default_allow_network=false
  check_network_ok loopback --experimental_sandbox_default_allow_network=false
  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    check_network_ok remote-ip --experimental_sandbox_default_allow_network=false
    check_network_ok remote-name --experimental_sandbox_default_allow_network=false
  fi
}

function test_sandbox_network_access_with_block_network() {
  setup_network_tests '"block-network"'

  case "$(uname -s)" in
    Linux)
      # TODO(jmmv): The linux-sandbox claims to allow localhost connectivity
      # within the network namespace... but that doesn't seem to be the case.
      check_network_not_ok localhost --experimental_sandbox_default_allow_network=true
      ;;

    *)
      check_network_ok localhost --experimental_sandbox_default_allow_network=true
      ;;
  esac
  check_network_ok unix-socket --experimental_sandbox_default_allow_network=true
  check_network_ok loopback --experimental_sandbox_default_allow_network=true
  if [[ -n "${REMOTE_NETWORK_ADDRESS}" ]]; then
    check_network_not_ok remote-ip --experimental_sandbox_default_allow_network=true
    check_network_not_ok remote-name --experimental_sandbox_default_allow_network=true
  fi
}

function test_sandbox_can_resolve_own_hostname() {
  add_rules_java "MODULE.bazel"
  setup_javatest_support
  mkdir -p src/test/java/com/example
  cat > src/test/java/com/example/HostNameTest.java <<'EOF'
package com.example;

import static org.junit.Assert.*;

import org.junit.Test;
import java.net.*;
import java.io.*;

public class HostNameTest {
  @Test
  public void testGetHostName() throws Exception {
    // This will throw an exception, if the local hostname cannot be resolved via DNS.
    assertNotNull(InetAddress.getLocalHost().getHostName());
  }
}
EOF
  cat > src/test/java/com/example/BUILD <<'EOF'
load("@rules_java//java:java_test.bzl", "java_test")
java_test(
  name = "HostNameTest",
  srcs = ["HostNameTest.java"],
  deps = ['//third_party:junit4'],
)
EOF

  bazel test --test_output=streamed src/test/java/com/example:HostNameTest &> $TEST_log \
    || fail "test should have passed"
}

function test_hostname_inside_sandbox_is_localhost_when_using_sandbox_fake_hostname_flag() {
  if [[ "$(uname -s)" != Linux ]]; then
    echo "Skipping test: fake hostnames not supported in this system" 1>&2
    return 0
  fi

  add_rules_java "MODULE.bazel"
  setup_javatest_support
  mkdir -p src/test/java/com/example
  cat > src/test/java/com/example/HostNameIsLocalhostTest.java <<'EOF'
package com.example;

import static org.junit.Assert.*;

import org.junit.Test;
import java.net.*;
import java.io.*;

public class HostNameIsLocalhostTest {
  @Test
  public void testHostNameIsLocalhost() throws Exception {
    // This will throw an exception, if the local hostname cannot be resolved via DNS.
    assertEquals("localhost", InetAddress.getLocalHost().getHostName());
  }
}
EOF
  cat > src/test/java/com/example/BUILD <<'EOF'
load("@rules_java//java:java_test.bzl", "java_test")
java_test(
  name = "HostNameIsLocalhostTest",
  srcs = ["HostNameIsLocalhostTest.java"],
  deps = ['//third_party:junit4'],
)
EOF

  bazel test --sandbox_fake_hostname --test_output=streamed src/test/java/com/example:HostNameIsLocalhostTest &> $TEST_log \
    || fail "test should have passed"
}

# The test shouldn't fail if the environment doesn't support running it.
check_sandbox_allowed || exit 0

run_suite "sandbox"
