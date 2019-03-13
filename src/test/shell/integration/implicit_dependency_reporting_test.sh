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
#
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# --- begin runfiles.bash initialization ---
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### SETUP #############################################################

test_custom_message() {
  rm -rf custommessage
  mkdir custommessage
  cd custommessage

  touch WORKSPACE
  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  out = ctx.new_file(ctx.label.name + ".txt")
  ctx.action(
    inputs = ctx.files._data,
    outputs = [out],
    command = ["cp"] + [f.path for f in ctx.files._data] + [out.path],
    mnemonic = "copying",
    progress_message = "Copying implict data dependency for %s" % ctx.label
  )

implicit_rule = rule(
  implementation = _rule_impl,
  attrs = {
    "_data": attr.label(allow_files=True, default = "//magic/place:data",
                        doc="You must manually put the data to magic/place.")
  },
)
EOF
  cat > BUILD <<'EOF'
load("//:rule.bzl", "implicit_rule")

implicit_rule(name = "it")
EOF
  mkdir -p magic/place
  echo Hello World > magic/place/data
  echo 'exports_files(["data"])' > magic/place/BUILD

  bazel build //:it || fail "Rule should work"

  rm -rf magic

  bazel build //:it > "${TEST_log}" 2>&1 \
      && fail "Missing implict dependency should be detected" || :
  expect_log 'rule.*implicit_rule.*implicitly depends'
  expect_log 'attribute _data of.*implicit_rule'
  expect_log 'You must manually put the data to magic/place'
}

run_suite "Integration tests for reporing missing implicit dependencies"
