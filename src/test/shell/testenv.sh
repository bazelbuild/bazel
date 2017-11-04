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
# Testing environment for the Bazel integration tests
#
# TODO(bazel-team): This file is currently an append of the old testenv.sh and
# test-setup.sh files. This must be cleaned up eventually.

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function is_windows() {
  # On windows, the shell test is actually running on msys
  [[ "${PLATFORM}" =~ msys_nt* ]]
}

function is_darwin() {
  [[ "${PLATFORM}" =~ darwin ]]
}

function _log_base() {
  prefix=$1
  shift
  echo >&2 "${prefix}[$(basename "$0") $(date "+%H:%M:%S.%N (%z)")] $@"
}

function log_info() {
  _log_base "INFO" "$@"
}

function log_fatal() {
  _log_base "ERROR" "$@"
  exit 1
}


# Set some environment variables needed on Windows.
if is_windows; then
  # TODO(philwo) remove this once we have a Bazel release that includes the CL
  # moving the Windows-specific TEST_TMPDIR into TestStrategy.
  TEST_TMPDIR_BASENAME="$(basename "$TEST_TMPDIR")"
  export TEST_TMPDIR="c:/temp/${TEST_TMPDIR_BASENAME}"

  # Bazel (TMPDIR) and Windows (TEMP, TMP) have three envvars that specify the
  # location of the temp directory...
  export TMPDIR="$TEST_TMPDIR"
  export TEMP="$TEST_TMPDIR"
  export TMP="$TEST_TMPDIR"

  export JAVA_HOME="$(ls -d C:/Program\ Files/Java/jdk* | sort | tail -n 1)"
  export BAZEL_SH="c:/tools/msys64/usr/bin/bash.exe"
  export BAZEL_VC="c:/Program Files (x86)/Microsoft Visual Studio/2017/Professional/VC"
  if [ ! -d "$BAZEL_VC" ]; then
    # Maybe Visual C++ Build Tools 2017 then?
    export BAZEL_VC="c:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC"
  fi
  if [ ! -d "$BAZEL_VC" ]; then
    # OK, well, maybe Visual C++ 2015 then?
    export BAZEL_VC="c:/Program Files (x86)/Microsoft Visual Studio 14.0/VC"
  fi
  if [ -x /c/Python27/python.exe ]; then
    export BAZEL_PYTHON="C:/Python27/python.exe"
    export PATH="/c/Python27:$PATH"
  elif [ -x /c/python_27_amd64/files/python.exe ]; then
    export BAZEL_PYTHON="C:/python_27_amd64/files/python.exe"
    export PATH="/c/python_27_amd64/files:$PATH"
  fi
fi

# Make the command "bazel" available for tests.
PATH_TO_BAZEL_BIN=$(rlocation io_bazel/src/bazel)
PATH_TO_BAZEL_WRAPPER="$(dirname $(rlocation io_bazel/src/test/shell/bin/bazel))"
# Convert PATH_TO_BAZEL_WRAPPER to Unix path style on Windows, because it will be
# added into PATH. There's problem if PATH=C:/msys64/usr/bin:/usr/local,
# because ':' is used as both path seperator and in C:/msys64/...
if is_windows; then
  PATH_TO_BAZEL_WRAPPER="$(cygpath -u "$PATH_TO_BAZEL_WRAPPER")"
fi
[ ! -f "${PATH_TO_BAZEL_WRAPPER}/bazel" ] \
  && log_fatal "Unable to find the Bazel binary at $PATH_TO_BAZEL_WRAPPER/bazel"
export PATH="$PATH_TO_BAZEL_WRAPPER:$PATH"

################### shell/bazel/testenv ##################################
# Setting up the environment for Bazel integration tests.
#
[ -z "$TEST_SRCDIR" ] && log_fatal "TEST_SRCDIR not set!"
BAZEL_RUNFILES="$TEST_SRCDIR/io_bazel"

if ! type rlocation &> /dev/null; then
  function rlocation() {
    if [[ "$1" = /* ]]; then
      echo $1
    else
      echo "$TEST_SRCDIR/$1"
    fi
  }
  export -f rlocation
fi

# WORKSPACE file
workspace_file="${BAZEL_RUNFILES}/WORKSPACE"

# Bazel
bazel_tree="$(rlocation io_bazel/src/test/shell/bazel/doc-srcs.zip)"
bazel_data="${BAZEL_RUNFILES}"

# Java
if is_windows; then
  jdk_dir="$(cygpath -m $(cd $(rlocation local_jdk/bin/java.exe)/../..; pwd))"
else
  jdk_dir="${TEST_SRCDIR}/local_jdk"
fi
langtools="$(rlocation io_bazel/src/test/shell/bazel/langtools.jar)"

# Tools directory location
tools_dir="$(dirname $(rlocation io_bazel/tools/BUILD))"
langtools_dir="$(dirname $(rlocation io_bazel/third_party/java/jdk/langtools/BUILD))"

# Sandbox tools
process_wrapper="${BAZEL_RUNFILES}/src/main/tools/process-wrapper"
linux_sandbox="${BAZEL_RUNFILES}/src/main/tools/linux-sandbox"

# Test data
testdata_path=${BAZEL_RUNFILES}/src/test/shell/bazel/testdata
python_server="${BAZEL_RUNFILES}/src/test/shell/bazel/testing_server.py"

# Third-party
MACHINE_TYPE="$(uname -m)"
MACHINE_IS_64BIT='no'
if [ "${MACHINE_TYPE}" = 'amd64' ] || [ "${MACHINE_TYPE}" = 'x86_64' ] || [ "${MACHINE_TYPE}" = 's390x' ]; then
  MACHINE_IS_64BIT='yes'
fi

MACHINE_IS_Z='no'
if [ "${MACHINE_TYPE}" = 's390x' ]; then
  MACHINE_IS_Z='yes'
fi

case "${PLATFORM}" in
  darwin)
    if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
      protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-osx-x86_64.exe"
    else
      protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-osx-x86_32.exe"
    fi
    ;;
  *)
    if [ "${MACHINE_IS_64BIT}" = 'yes' ]; then
      if [ "${MACHINE_IS_Z}" = 'yes' ]; then
        protoc_compiler="${BAZEL_RUNFILES}//third_party/protobuf/protoc-linux-s390x_64.exe"
      else
        protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-linux-x86_64.exe"
      fi
    else
        protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-linux-x86_32.exe"
    fi
    ;;
esac

if [ -z ${RUNFILES_MANIFEST_ONLY+x} ]; then
  junit_jar="${BAZEL_RUNFILES}/third_party/junit/junit-*.jar"
  hamcrest_jar="${BAZEL_RUNFILES}/third_party/hamcrest/hamcrest-*.jar"
else
  junit_jar=$(rlocation io_bazel/third_party/junit/junit-.*.jar)
  hamcrest_jar=$(rlocation io_bazel/third_party/hamcrest/hamcrest-.*.jar)
fi


function use_bazel_workspace_file() {
  mkdir -p src/test/{shell/bazel,docker}
  cat >src/test/docker/docker_repository.bzl <<EOF
def docker_repository():
  pass
EOF
  cat >src/test/docker/flavours.bzl <<EOF
def pull_images_for_docker_tests():
  pass
EOF
  touch src/test/docker/BUILD
  cat >src/test/shell/bazel/list_source_repository.bzl <<EOF
def list_source_repository(name):
  pass
EOF
  touch src/test/shell/bazel/BUILD
  rm -f WORKSPACE
  ln -sf ${workspace_file} WORKSPACE
}

# This function copies the tools directory from Bazel.
function copy_tools_directory() {
  cp -RL ${tools_dir}/* tools
  # tools/jdk/BUILD file for JDK 7 is generated.
  # Only works if there's 0 or 1 matches.
  # If there are multiple, the test fails.
  if [ -f tools/jdk/BUILD.* ]; then
    cp tools/jdk/BUILD.* tools/jdk/BUILD
  fi
  if [ -f tools/jdk/BUILD ]; then
    chmod +w tools/jdk/BUILD
  fi
  # To support custom langtools
  cp ${langtools} tools/jdk/langtools.jar
  cat >>tools/jdk/BUILD <<'EOF'
filegroup(name = "test-langtools", srcs = ["langtools.jar"])
EOF

  mkdir -p third_party/java/jdk/langtools
  cp -R ${langtools_dir}/* third_party/java/jdk/langtools

  chmod -R +w .
  mkdir -p tools/defaults
  touch tools/defaults/BUILD

  mkdir -p third_party/py/gflags
  cat > third_party/py/gflags/BUILD <<EOF
licenses(["notice"])
package(default_visibility = ["//visibility:public"])

py_library(
    name = "gflags",
)
EOF
}

# Report whether a given directory name corresponds to a tools directory.
function is_tools_directory() {
  case "$1" in
    third_party|tools|src)
      true
      ;;
    *)
      false
      ;;
  esac
}

# Copy the examples of the base workspace
function copy_examples() {
  EXAMPLE="$(cd $(dirname $(rlocation io_bazel/examples/cpp/BUILD))/..; pwd)"
  cp -RL ${EXAMPLE} .
  chmod -R +w .
}

#
# Find a random unused TCP port
#
pick_random_unused_tcp_port () {
    perl -MSocket -e '
sub CheckPort {
  my ($port) = @_;
  socket(TCP_SOCK, PF_INET, SOCK_STREAM, getprotobyname("tcp"))
    || die "socket(TCP): $!";
  setsockopt(TCP_SOCK, SOL_SOCKET, SO_REUSEADDR, 1)
    || die "setsockopt(TCP): $!";
  return 0 unless bind(TCP_SOCK, sockaddr_in($port, INADDR_ANY));
  socket(UDP_SOCK, PF_INET, SOCK_DGRAM, getprotobyname("udp"))
    || die "socket(UDP): $!";
  return 0 unless bind(UDP_SOCK, sockaddr_in($port, INADDR_ANY));
  return 1;
}
for (1 .. 128) {
  my ($port) = int(rand() * 27000 + 32760);
  if (CheckPort($port)) {
    print "$port\n";
    exit 0;
  }
}
print "NO_FREE_PORT_FOUND\n";
exit 1;
'
}

#
# A uniform SHA-256 commands that works accross platform
#
case "${PLATFORM}" in
  darwin|freebsd)
    function sha256sum() {
      cat "$1" | shasum -a 256 | cut -f 1 -d " "
    }
    ;;
  *)
    # Under linux sha256sum should exists
    ;;
esac

################### shell/bazel/test-setup ###############################
# Setup bazel for integration tests
#

# OS X has a limit in the pipe length, so force the root to a shorter one
bazel_root="${TEST_TMPDIR}/root"

# Delete stale installation directory from previously failed tests. On Windows
# we regularly get the same TEST_TMPDIR but a failed test may only partially
# clean it up, and the next time the test runs, Bazel reports a corrupt
# installation error. See https://github.com/bazelbuild/bazel/issues/3618
rm -rf "${bazel_root}"
mkdir -p "${bazel_root}"

bazel_javabase="${jdk_dir}"

log_info "bazel binary is at $PATH_TO_BAZEL_WRAPPER"

# Here we unset variable that were set by the invoking Blaze instance
unset JAVA_RUNFILES

function setup_bazelrc() {
  cat >$TEST_TMPDIR/bazelrc <<EOF
# Set the user root properly for this test invocation.
startup --output_user_root=${bazel_root}
# Set the correct javabase from the outer bazel invocation.
startup --host_javabase=${bazel_javabase}

# Print all progress messages because we regularly grep the output in tests.
common --show_progress_rate_limit=-1

# Disable terminal-specific features.
common --color=no --curses=no

build -j 8
${EXTRA_BAZELRC:-}
EOF
}

function setup_android_sdk_support() {
  ANDROID_SDK=$PWD/android_sdk
  SDK_SRCDIR=$TEST_SRCDIR/androidsdk
  mkdir -p $ANDROID_SDK
  for i in $SDK_SRCDIR/*; do
    ln -s "$i" "$ANDROID_SDK/$(basename $i)"
  done
cat >> WORKSPACE <<EOF
android_sdk_repository(
    name = "androidsdk",
    path = "$ANDROID_SDK",
)
EOF
}

function setup_android_ndk_support() {
  ANDROID_NDK=$PWD/android_ndk
  NDK_SRCDIR=$TEST_SRCDIR/androidndk/ndk
  mkdir -p $ANDROID_NDK
  for i in $NDK_SRCDIR/*; do
    if [[ "$(basename $i)" != "BUILD" ]]; then
      ln -s "$i" "$ANDROID_NDK/$(basename $i)"
    fi
  done
  cat >> WORKSPACE <<EOF
android_ndk_repository(
    name = "androidndk",
    path = "$ANDROID_NDK",
)
EOF
}

function setup_javatest_common() {
  # TODO(bazel-team): we should use remote repositories.
  mkdir -p third_party
  if [ ! -f third_party/BUILD ]; then
    cat <<EOF >third_party/BUILD
package(default_visibility = ["//visibility:public"])
EOF
  fi

  [ -e third_party/junit.jar ] || ln -s ${junit_jar} third_party/junit.jar
  [ -e third_party/hamcrest.jar ] \
    || ln -s ${hamcrest_jar} third_party/hamcrest.jar
}

function setup_javatest_support() {
  setup_javatest_common
  grep -q 'name = "junit4"' third_party/BUILD \
    || cat <<EOF >>third_party/BUILD
java_import(
    name = "junit4",
    jars = [
        "junit.jar",
        "hamcrest.jar",
    ],
)
EOF
}

function setup_skylark_javatest_support() {
  setup_javatest_common
  grep -q "name = \"junit4-jars\"" third_party/BUILD \
    || cat <<EOF >>third_party/BUILD
filegroup(
    name = "junit4-jars",
    srcs = [
        "junit.jar",
        "hamcrest.jar",
    ],
)
EOF
}

# Sets up Objective-C tools. Mac only.
function setup_objc_test_support() {
  IOS_SDK_VERSION=$(xcrun --sdk iphoneos --show-sdk-version)
}

workspaces=()
# Set-up a new, clean workspace with only the tools installed.
function create_new_workspace() {
  new_workspace_dir=${1:-$(mktemp -d ${TEST_TMPDIR}/workspace.XXXXXXXX)}
  rm -fr ${new_workspace_dir}
  mkdir -p ${new_workspace_dir}
  workspaces+=(${new_workspace_dir})
  cd ${new_workspace_dir}
  mkdir tools
  mkdir -p third_party/java/jdk/langtools

  copy_tools_directory

  [ -e third_party/java/jdk/langtools/javac-9-dev-r4023-3.jar ] \
    || ln -s "${langtools_path}"  third_party/java/jdk/langtools/javac-9-dev-r4023-3.jar

  touch WORKSPACE
}

# Set-up a clean default workspace.
function setup_clean_workspace() {
  export WORKSPACE_DIR=${TEST_TMPDIR}/workspace
  log_info "setting up client in ${WORKSPACE_DIR}" >> $TEST_log
  rm -fr ${WORKSPACE_DIR}
  create_new_workspace ${WORKSPACE_DIR}
  [ "${new_workspace_dir}" = "${WORKSPACE_DIR}" ] \
    || log_fatal "Failed to create workspace"

  # On macOS, mktemp expects the template to have the Xs at the end.
  # On Linux, the Xs may be anywhere.
  local -r bazel_stdout="$(mktemp "${TEST_TMPDIR}/XXXXXXXX")"
  local -r bazel_stderr="${bazel_stdout}.err"
  # On Windows, we mustn't run Bazel in a subshell because of
  # https://github.com/bazelbuild/bazel/issues/3148.
  bazel info install_base >"$bazel_stdout" 2>"$bazel_stderr" \
    && export BAZEL_INSTALL_BASE=$(cat "$bazel_stdout") \
    || log_fatal "'bazel info install_base' failed, stderr: $(cat "$bazel_stderr")"
  bazel info bazel-genfiles >"$bazel_stdout" 2>"$bazel_stderr" \
    && export BAZEL_GENFILES_DIR=$(cat "$bazel_stdout") \
    || log_fatal "'bazel info bazel-genfiles' failed, stderr: $(cat "$bazel_stderr")"
  bazel info bazel-bin >"$bazel_stdout" 2>"$bazel_stderr" \
    && export BAZEL_BIN_DIR=$(cat "$bazel_stdout") \
    || log_fatal "'bazel info bazel-bin' failed, stderr: $(cat "$bazel_stderr")"
  rm -f "$bazel_stdout" "$bazel_stderr"

  if is_windows; then
    export BAZEL_SH="$(cygpath --windows /bin/bash)"
  fi
}

# Clean up all files that are not in tools directories, to restart
# from a clean workspace
function cleanup_workspace() {
  if [ -d "${WORKSPACE_DIR:-}" ]; then
    log_info "Cleaning up workspace" >> $TEST_log
    cd ${WORKSPACE_DIR}
    bazel clean >> $TEST_log 2>&1 # Clean up the output base

    for i in *; do
      if ! is_tools_directory "$i"; then
        rm -fr "$i"
      fi
    done
    touch WORKSPACE
  fi
  for i in "${workspaces[@]}"; do
    if [ "$i" != "${WORKSPACE_DIR:-}" ]; then
      rm -fr $i
    fi
  done
  workspaces=()
}

# Clean-up the bazel install base
function cleanup() {
  if [ -d "${BAZEL_INSTALL_BASE:-__does_not_exists__}" ]; then
    # Windows takes its time to shut down Bazel and we can't delete A-server.jar
    # until then, so just give it time and keep trying for 2 minutes.
    for i in {1..120}; do
      if rm -fr "${BAZEL_INSTALL_BASE}" ; then
        break
      fi
      if (( i == 10 )) || (( i == 30 )) || (( i == 60 )) ; then
        log_info "Test cleanup: couldn't delete ${BAZEL_INSTALL_BASE} after $i seconds" \
                 "(Timeout in $((120-i)) seconds.)"
      fi
      sleep 1
    done
  fi
}

function tear_down() {
  cleanup_workspace
}

#
# Simples assert to make the tests more readable
#
function assert_build() {
  bazel build -s --verbose_failures $* || fail "Failed to build $*"
}

function assert_build_output() {
  local OUTPUT=$1
  shift
  assert_build "$*"
  test -f "$OUTPUT" || fail "Output $OUTPUT not found for target $*"
}

function assert_build_fails() {
  bazel build -s $1 >> $TEST_log 2>&1 \
    && fail "Test $1 succeed while expecting failure" \
    || true
  if [ -n "${2:-}" ]; then
    expect_log "$2"
  fi
}

function assert_test_ok() {
  bazel test --test_output=errors $* >> $TEST_log 2>&1 \
    || fail "Test $1 failed while expecting success"
}

function assert_test_fails() {
  bazel test --test_output=errors $* >> $TEST_log 2>&1 \
    && fail "Test $* succeed while expecting failure" \
    || true
  expect_log "$1.*FAILED"
}

function assert_binary_run() {
  $1 >> $TEST_log 2>&1 || fail "Failed to run $1"
  [ -z "${2:-}" ] || expect_log "$2"
}

function assert_bazel_run() {
  bazel run $1 >> $TEST_log 2>&1 || fail "Failed to run $1"
    [ -z "${2:-}" ] || expect_log "$2"

  assert_binary_run "./bazel-bin/$(echo "$1" | sed 's|^//||' | sed 's|:|/|')" "${2:-}"
}

setup_bazelrc
setup_clean_workspace

################### shell/integration/testenv ############################
# Setting up the environment for our legacy integration tests.
#
PRODUCT_NAME=bazel
WORKSPACE_NAME=main
bazelrc=$TEST_TMPDIR/bazelrc

function put_bazel_on_path() {
  # do nothing as test-setup already does that
  true
}

function write_default_bazelrc() {
  setup_bazelrc
}

function add_to_bazelrc() {
  echo "$@" >> $bazelrc
}

function create_and_cd_client() {
  setup_clean_workspace
  echo "workspace(name = '$WORKSPACE_NAME')" >WORKSPACE
  touch .bazelrc
}

################### Extra ############################
# Functions that need to be called before each test.
create_and_cd_client
