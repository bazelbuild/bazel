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
# Setting up the environment for Bazel integration tests.
#
[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }
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

# Load the unit-testing framework
source "$(rlocation io_bazel/src/test/shell/unittest.bash)" || \
  { echo "Failed to source unittest.bash" >&2; exit 1; }

# WORKSPACE file
workspace_file="${BAZEL_RUNFILES}/WORKSPACE"

# Bazel
bazel_tree="$(rlocation io_bazel/src/test/shell/bazel/doc-srcs.zip)"
bazel="$(rlocation io_bazel/src/bazel)"
bazel_data="${BAZEL_RUNFILES}"

# Windows
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
function is_windows() {
  # On windows, the shell test actually running on msys
  if [[ "${PLATFORM}" =~ msys_nt* ]]; then
    true
  else
    false
  fi
}

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
EXTRA_BAZELRC="build --ios_sdk_version=8.4"

# Java tooling
javabuilder_path="$(find ${BAZEL_RUNFILES} -name JavaBuilder_*.jar)"
langtools_path="${BAZEL_RUNFILES}/third_party/java/jdk/langtools/javac.jar"
singlejar_path="${BAZEL_RUNFILES}/src/java_tools/singlejar/SingleJar_deploy.jar"
genclass_path="${BAZEL_RUNFILES}/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/genclass/GenClass_deploy.jar"
junitrunner_path="${BAZEL_RUNFILES}/src/java_tools/junitrunner/java/com/google/testing/junit/runner/Runner_deploy.jar"
ijar_path="${BAZEL_RUNFILES}/third_party/ijar/ijar"

# Sandbox tools
process_wrapper="${BAZEL_RUNFILES}/src/main/tools/process-wrapper"
linux_sandbox="${BAZEL_RUNFILES}/src/main/tools/linux-sandbox"

# iOS and Objective-C tooling
iossim_path="${BAZEL_RUNFILES}/third_party/iossim/iossim"
actoolwrapper_path="${BAZEL_RUNFILES}/src/tools/xcode/actoolwrapper/actoolwrapper.sh"
ibtoolwrapper_path="${BAZEL_RUNFILES}/src/tools/xcode/ibtoolwrapper/ibtoolwrapper.sh"
swiftstdlibtoolwrapper_path="${BAZEL_RUNFILES}/src/tools/xcode/swiftstdlibtoolwrapper/swiftstdlibtoolwrapper.sh"
momcwrapper_path="${BAZEL_RUNFILES}/src/tools/xcode/momcwrapper/momcwrapper.sh"
bundlemerge_path="${BAZEL_RUNFILES}/src/objc_tools/bundlemerge/bundlemerge_deploy.jar"
plmerge_path="${BAZEL_RUNFILES}/src/objc_tools/plmerge/plmerge_deploy.jar"
xcodegen_path="${BAZEL_RUNFILES}/src/objc_tools/xcodegen/xcodegen_deploy.jar"
stdredirect_path="${BAZEL_RUNFILES}/src/tools/xcode/stdredirect/StdRedirect.dylib"
realpath_path="${BAZEL_RUNFILES}/src/tools/xcode/realpath/realpath"
environment_plist_path="${BAZEL_RUNFILES}/src/tools/xcode/environment/environment_plist.sh"
xcrunwrapper_path="${BAZEL_RUNFILES}/src/tools/xcode/xcrunwrapper/xcrunwrapper.sh"

# Test data
testdata_path=${BAZEL_RUNFILES}/src/test/shell/bazel/testdata
python_server="${BAZEL_RUNFILES}/src/test/shell/bazel/testing_server.py"

# Third-party
MACHINE_TYPE="$(uname -m)"
MACHINE_IS_64BIT='no'
if [ "${MACHINE_TYPE}" = 'amd64' -o "${MACHINE_TYPE}" = 'x86_64' ]; then
  MACHINE_IS_64BIT='yes'
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
      protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-linux-x86_64.exe"
    else
      protoc_compiler="${BAZEL_RUNFILES}/third_party/protobuf/protoc-linux-x86_32.exe"
    fi
    ;;
esac

if [ -z ${RUNFILES_MANIFEST_ONLY+x} ]; then
  protoc_jar="${BAZEL_RUNFILES}/third_party/protobuf/protobuf-*.jar"
  junit_jar="${BAZEL_RUNFILES}/third_party/junit/junit-*.jar"
  hamcrest_jar="${BAZEL_RUNFILES}/third_party/hamcrest/hamcrest-*.jar"
else
  protoc_jar=$(rlocation io_bazel/third_party/protobuf/protobuf-.*.jar)
  junit_jar=$(rlocation io_bazel/third_party/junit/junit-.*.jar)
  hamcrest_jar=$(rlocation io_bazel/third_party/hamcrest/hamcrest-.*.jar)
fi

# This function copies the tools directory from Bazel.
function copy_tools_directory() {
  cp -RL ${tools_dir}/* tools
  # tools/jdk/BUILD file for JDK 7 is generated.
  if [ -f tools/jdk/BUILD.* ]; then
    cp tools/jdk/BUILD.* tools/jdk/BUILD
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
  darwin)
    function sha256sum() {
      cat "$1" | shasum -a 256 | cut -f 1 -d " "
    }
    ;;
  *)
    # Under linux sha256sum should exists
    ;;
esac
