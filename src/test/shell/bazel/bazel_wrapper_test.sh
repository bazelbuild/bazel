#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

# An end-to-end test for the wrapper used by our installer and distro packages.

# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

set -e

wrapper=$(rlocation io_bazel/scripts/packages/bazel.sh)

mock_bazel() {
  {
    cat <<'EOF'
#!/bin/bash
set -euo pipefail
if [[ ${1:-""} == "--version" ]]; then
  echo "bazel BAZEL_VERSION"
else
  echo "Hello from $(basename $0)!"
  echo "My args: $@"
fi
exit 0
EOF
  } | sed "s/BAZEL_VERSION/$2/" > "$1"
  chmod +x "$1"
}

setup_mock() {
  # Setup a mock workspace.
  mkdir ws ws/subdir
  touch ws/{BUILD,WORKSPACE}
  touch ws/subdir/BUILD

  # Setup a mock "/usr/bin" folder with the wrapper and some "Bazel" binaries.
  mkdir bin
  cp "$wrapper" "bin/bazel"
  chmod +x "bin/bazel"
  mock_bazel "bin/bazel-0.29.1" "0.29.1"
  mock_bazel "bin/bazel-1.0.1" "1.0.1"
  mock_bazel "bin/bazel-real" "1.1.0"

  # We don't want USE_BAZEL_VERSION passed by --test_env to affect this test.
  unset USE_BAZEL_VERSION
  cd ws
}

test_use_bazel_version_envvar() {
  setup_mock

  USE_BAZEL_VERSION="0.29.1" ../bin/bazel build &> "$TEST_log"
  expect_log "Hello from bazel-0.29.1"
  expect_log "My args: build"
}

test_bazelversion_file() {
  setup_mock

  echo "1.0.1" > .bazelversion

  ../bin/bazel info &> "$TEST_log"
  expect_log "Hello from bazel-1.0.1"
  expect_log "My args: info"

  cd subdir
  ../../bin/bazel build //src:bazel &> "$TEST_log"
  expect_log "Hello from bazel-1.0.1"
  expect_log "My args: build //src:bazel"
}

test_uses_bazelreal() {
  setup_mock

  ../bin/bazel &> "$TEST_log"
  expect_log "Hello from bazel-real"
  expect_log "My args:"
}

test_uses_latest_version() {
  setup_mock

  rm ../bin/bazel-real
  ../bin/bazel &> "$TEST_log"
  expect_log "Hello from bazel-1.0.1"

  rm ../bin/bazel-1.0.1
  ../bin/bazel &> "$TEST_log"
  expect_log "Hello from bazel-0.29.1"
}

test_error_message_for_no_available_bazel_version() {
  setup_mock

  rm ../bin/bazel-*
  if ../bin/bazel &> "$TEST_log"; then
    fail "Bazel wrapper should have failed"
  fi
  expect_log "No installed Bazel version found, cannot continue"
}

test_error_message_for_required_bazel_not_found() {
  setup_mock

  if USE_BAZEL_VERSION="foobar" ../bin/bazel &> "$TEST_log"; then
    fail "Bazel wrapper should have failed"
  fi
  expect_log "ERROR: The project you're trying to build requires Bazel foobar (specified in \$USE_BAZEL_VERSION)"
}

test_recommends_curl() {
  setup_mock

  mkdir mockpath
  for bin in uname readlink dirname tr; do
    ln -s "$(command -v $bin)" mockpath
  done
  touch mockpath/curl
  chmod +x mockpath/curl

  if PATH="$(pwd)/mockpath" USE_BAZEL_VERSION="foobar" ../bin/bazel &> "$TEST_log"; then
    fail "Bazel wrapper should have failed"
  fi
  expect_log "curl -LO"
  expect_not_log "wget"
}

test_recommends_wget() {
  setup_mock

  mkdir mockpath
  for bin in uname readlink dirname tr; do
    ln -s "$(command -v $bin)" mockpath
  done
  touch mockpath/wget
  chmod +x mockpath/wget

  if PATH="$(pwd)/mockpath" USE_BAZEL_VERSION="foobar" ../bin/bazel &> "$TEST_log"; then
    fail "Bazel wrapper should have failed"
  fi
  expect_log "wget"
  expect_not_log "curl"
}

test_recommends_manual_download() {
  setup_mock

  mkdir mockpath
  for bin in uname readlink dirname tr; do
    ln -s "$(command -v $bin)" mockpath
  done

  if PATH="$(pwd)/mockpath" USE_BAZEL_VERSION="foobar" ../bin/bazel &> "$TEST_log"; then
    fail "Bazel wrapper should have failed"
  fi
  expect_log "Please put the downloaded Bazel binary into this location"
  expect_not_log "curl"
  expect_not_log "wget"
}

test_delegates_to_wrapper_if_present() {
  setup_mock

  mkdir tools
  cat > tools/bazel <<'EOF'
#!/bin/bash
set -euo pipefail
echo "Hello from the wrapper tools/bazel!"
echo "BAZEL_REAL = ${BAZEL_REAL}"
echo "My args: $@"
exit 0
EOF
  chmod +x tools/bazel

  # Due to https://github.com/bazelbuild/bazel/issues/10356, we have to ignore
  # the .bazelversion file in case a tools/bazel executable is present.
  # Even if the requested Bazel version doesn't exist, the wrapper still has to
  # add the $BAZEL_REAL environment variable, so this will point to a
  # non-existing file in that case. It's up to the owner of the repo to decide
  # what to make of it - e.g. print an error message, fallback to something else
  # or completely ignore the $BAZEL_REAL variable.
  USE_BAZEL_VERSION="3.0.0" ../bin/bazel build //src:bazel &> "$TEST_log"
  expect_log "Hello from the wrapper tools/bazel!"
  expect_log "BAZEL_REAL = .*/bin/bazel-3.0.0"
  expect_log "My args: build //src:bazel"
}

test_gracefully_handles_bogus_bazelversion() {
  setup_mock

  mkdir tools
  cat > tools/bazel <<'EOF'
#!/bin/bash
set -euo pipefail
echo "Hello from the wrapper tools/bazel!"
echo "My args: $@"
exit 0
EOF
  chmod +x tools/bazel

  # Create a .bazelversion file that looks completely different than what we
  # actually support. The wrapper is supposed to not crash and still call the
  # tools/bazel executable. The content of the $BAZEL_REAL variable can be
  # completely bogus, of course.
  cat > .bazelversion <<'EOF'
mirrors: [http://mirror.example/bazel-5.0.0, http://github.com/example/]
# The above is our internal mirror. The syntax is only supported since
# Bazelisk 42.0.
EOF
  ../bin/bazel build //src:bazel &> "$TEST_log"
  expect_log "Hello from the wrapper tools/bazel!"
  expect_log "My args: build //src:bazel"
}

test_wrapper_detects_version_of_bazel_real() {
  setup_mock

  echo "1.1.0" > .bazelversion
  ../bin/bazel &> "$TEST_log"
  expect_log "Hello from bazel-real"
  expect_log "My args:"
}

run_suite "Integration tests for scripts/packages/bazel.sh."
