#!/bin/bash
#
# Copyright 2022 The Bazel Authors. All rights reserved.
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
# Tests remote build event uploader.

set -euo pipefail

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
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

function set_up() {
  start_worker \
        --incompatible_remote_symlinks
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

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
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

BEP_JSON=bep.json

function expect_bes_file_uploaded() {
  local file=$1
  if [[ $(cat $BEP_JSON) =~ ${file}\",\"uri\":\"bytestream://localhost:${worker_port}/blobs/([^/]*) ]]; then
    if ! remote_cas_file_exist ${BASH_REMATCH[1]}; then
      cat $BEP_JSON >> $TEST_log && append_remote_cas_files $TEST_log && fail "$file is not uploaded"
    fi
  else
    cat $BEP_JSON > $TEST_log
    fail "$file is not converted to bytestream://"
  fi
}

function expect_bes_file_not_uploaded() {
  local file=$1
  if [[ $(cat $BEP_JSON) =~ ${file}\",\"uri\":\"bytestream://localhost:${worker_port}/blobs/([^/]*) ]]; then
    if remote_cas_file_exist ${BASH_REMATCH[1]}; then
     cat $BEP_JSON >> $TEST_log && append_remote_cas_files $TEST_log && fail "$file is uploaded"
    fi
  else
    cat $BEP_JSON > $TEST_log
    fail "$file is not converted to bytestream://"
  fi
}

function test_upload_minimal_convert_paths_for_existed_blobs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_convert_paths_for_existed_blobs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_doesnt_upload_missing_blobs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
  tags = ["no-remote"],
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_upload_missing_blobs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
  tags = ["no-remote"],
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_respect_no_upload_results() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --remote_upload_local_results=false \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_ignore_no_upload_results() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --remote_upload_local_results=false \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_respect_no_upload_results_combined_cache() {
  local cache_dir="${TEST_TMPDIR}/disk_cache"
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  rm -rf $cache_dir
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --disk_cache=$cache_dir \
      --remote_upload_local_results=false \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
  remote_cas_files="$(count_remote_cas_files)"
  [[ "$remote_cas_files" == 1 ]] || fail "Expected 1 remote cas entries, not $remote_cas_files"
  disk_cas_files="$(count_disk_cas_files $cache_dir)"
  # foo.txt, stdout and stderr for action 'foo'
  [[ "$disk_cas_files" == 3 ]] || fail "Expected 3 disk cas entries, not $disk_cas_files"
}

function test_upload_all_combined_cache() {
  local cache_dir="${TEST_TMPDIR}/disk_cache"
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  rm -rf $cache_dir
  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --disk_cache=$cache_dir \
      --remote_upload_local_results=false \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
  remote_cas_files="$(count_remote_cas_files)"
  [[ "$remote_cas_files" == 1 ]] || fail "Expected 1 remote cas entries, not $remote_cas_files"
  disk_cas_files="$(count_disk_cas_files $cache_dir)"
  # foo.txt, stdout and stderr for action 'foo'
  [[ "$disk_cas_files" == 3 ]] || fail "Expected 3 disk cas entries, not $disk_cas_files"
}

function test_upload_minimal_alias_action_doesnt_upload_missing_blobs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
  tags = ["no-remote"],
)

alias(
  name = 'foo-alias',
  actual = '//a:foo',
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo-alias >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_alias_action() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
  tags = ["no-remote"],
)

alias(
  name = 'foo-alias',
  actual = '//a:foo',
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo-alias >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_trees_doesnt_upload_missing_blobs() {
  mkdir -p a
  cat > a/output_dir.bzl <<'EOF'
def _gen_output_dir_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.attr.outdir)
    ctx.actions.run_shell(
        outputs = [output_dir],
        inputs = [],
        command = """
          echo 0 > $1/0.txt
          echo 1 > $1/1.txt
          mkdir -p $1/sub
          echo "Shuffle, duffle, muzzle, muff" > $1/sub/bar
        """,
        arguments = [output_dir.path],
        execution_requirements = {"no-remote": ""},
    )
    return [
        DefaultInfo(files = depset(direct = [output_dir])),
    ]
gen_output_dir = rule(
    implementation = _gen_output_dir_impl,
    attrs = {
        "outdir": attr.string(mandatory = True),
    },
)
EOF

  cat > a/BUILD <<EOF
load(":output_dir.bzl", "gen_output_dir")
gen_output_dir(
    name = "foo",
    outdir = "dir",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded dir/0.txt
  expect_bes_file_not_uploaded dir/1.txt
  expect_bes_file_not_uploaded dir/sub/bar
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_trees() {
  mkdir -p a
  cat > a/output_dir.bzl <<'EOF'
def _gen_output_dir_impl(ctx):
    output_dir = ctx.actions.declare_directory(ctx.attr.outdir)
    ctx.actions.run_shell(
        outputs = [output_dir],
        inputs = [],
        command = """
          echo 0 > $1/0.txt
          echo 1 > $1/1.txt
          mkdir -p $1/sub
          echo "Shuffle, duffle, muzzle, muff" > $1/sub/bar
        """,
        arguments = [output_dir.path],
        execution_requirements = {"no-remote": ""},
    )
    return [
        DefaultInfo(files = depset(direct = [output_dir])),
    ]
gen_output_dir = rule(
    implementation = _gen_output_dir_impl,
    attrs = {
        "outdir": attr.string(mandatory = True),
    },
)
EOF

  cat > a/BUILD <<EOF
load(":output_dir.bzl", "gen_output_dir")
gen_output_dir(
    name = "foo",
    outdir = "dir",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded dir/0.txt
  expect_bes_file_uploaded dir/1.txt
  expect_bes_file_uploaded dir/sub/bar
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_upload_testlogs() {
  mkdir -p a
  cat > a/BUILD <<EOF
sh_test(
  name = 'test',
  srcs = ['test.sh'],
  tags = ['no-remote'],
)
EOF
  cat > a/test.sh <<EOF
echo 'it works!'
EOF
  chmod +x a/test.sh

  bazel test \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:test >& $TEST_log || fail "Failed to build"

  expect_bes_file_not_uploaded test.sh
  expect_bes_file_uploaded test.log
  expect_bes_file_uploaded test.xml
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_upload_testlogs() {
  mkdir -p a
  cat > a/BUILD <<EOF
sh_test(
  name = 'test',
  srcs = ['test.sh'],
  tags = ['no-remote'],
)
EOF
  cat > a/test.sh <<EOF
echo 'it works!'
EOF
  chmod +x a/test.sh

  bazel test \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:test >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded test.sh
  expect_bes_file_uploaded test.log
  expect_bes_file_uploaded test.xml
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_upload_buildlogs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ['foo.txt'],
  cmd  = 'echo "stdout" && echo "stderr" >&2 && exit 1',
  tags = ['no-remote'],
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || true

  expect_bes_file_uploaded stdout
  expect_bes_file_uploaded stderr
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_all_upload_buildlogs() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ['foo.txt'],
  cmd  = 'echo "stdout" && echo "stderr" >&2 && exit 1',
  tags = ['no-remote'],
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || true

  expect_bes_file_uploaded stdout
  expect_bes_file_uploaded stderr
  expect_bes_file_uploaded command.profile.gz
}

function test_upload_minimal_upload_profile() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --profile=mycommand.profile.gz \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded "mycommand.profile.gz"
}

function test_upload_all_upload_profile() {
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --profile=mycommand.profile.gz \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded "mycommand.profile.gz"
}

run_suite "Remote build event uploader tests"
