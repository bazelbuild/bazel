#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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

# End-to-end test for builtins injection and the various values of
# --experimental_builtins_bzl_path.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

# override rules_java in bazel otherwise no build succeeds without injection
[[ $(type -t mock_rules_java_to_avoid_downloading) == function ]] && mock_rules_java_to_avoid_downloading

function test_injection() {
  # //pkg prints _builtins_dummy when loaded.
  mkdir pkg
  cat > pkg/BUILD <<'EOF'
load(":foo.bzl", "foo")
foo()
EOF
  cat > pkg/foo.bzl <<'EOF'
def foo():
    print("dummy :: " + str(_builtins_dummy))
EOF
  # Provide a different value for the dummy at the path where builtins_bzl is
  # expected to be in a Bazel source tree.
  mkdir -p "$BUILTINS_PACKAGE_PATH_IN_SOURCE"
  cat > "$BUILTINS_PACKAGE_PATH_IN_SOURCE/exports.bzl" <<'EOF'
exported_toplevels = {"_builtins_dummy": "workspace value"}
exported_rules = {}
exported_to_java = {}
EOF
  # Provide yet another value at a different arbitrary path.
  mkdir alternate
  cat > alternate/exports.bzl <<'EOF'
exported_toplevels = {"_builtins_dummy": "alternate value"}
exported_rules = {}
exported_to_java = {}
EOF


  # With injection disabled.
  #
  # TODO(#11437): Remove this case once builtins injection can no longer be
  # disabled. Note that eventually, rule migration to Starlark will cause
  # implicit dependencies/tools to rely on injection, so even a trivial build
  # without injection may break. (That may also mean we have to update this test
  # at some point, so that the other builtins roots are based on the one in the
  # install base, instead of being virtually empty.)
  bazel build --nobuild //pkg:BUILD --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path= \
      &>"$TEST_log" || fail "bazel build failed"
  expect_log "dummy :: original value"

  # Using the builtins root that's bundled with bazel.
  bazel build --nobuild //pkg:BUILD --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path=%bundled% \
      &>"$TEST_log" || fail "bazel build failed"
  # "overridden value" comes from the exports.bzl in production Bazel.
  expect_log "dummy :: overridden value"

  # Using the builtins root located within the client workspace, as if we're
  # running Bazel in its own source tree.
  bazel build --nobuild //pkg:BUILD --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path=%workspace% \
      &>"$TEST_log" || fail "bazel build failed"
  expect_log "dummy :: workspace value"

  # Using the builtins root at the path given to the flag. (Need not be within
  # workspace, though this one is.)
  bazel build --nobuild //pkg:BUILD --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path=alternate \
      &>"$TEST_log" || fail "bazel build failed"
  expect_log "dummy :: alternate value"

  # Try an incremental updates to the builtins .bzl files.
  cat > alternate/exports.bzl <<'EOF'
exported_toplevels = {"_builtins_dummy": "second alternate value"}
exported_rules = {}
exported_to_java = {}
EOF
  bazel build --nobuild //pkg:BUILD --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path=alternate \
      &>"$TEST_log" || fail "bazel build failed"
  expect_log "dummy :: second alternate value"

  # Ensure builtins .bzl files aren't visible to bazel query the way normal .bzl
  # files are.
  bazel query 'buildfiles(//pkg:BUILD)' --experimental_builtins_dummy=true \
      --experimental_builtins_bzl_path=alternate \
      &>"$TEST_log" || fail "bazel query failed"
  expect_not_log "exports.bzl"
}

run_suite "builtins_injection_test"
