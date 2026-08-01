#!/usr/bin/env bash
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
  start_worker
}

function tear_down() {
  bazel clean >& $TEST_log
  stop_worker
}

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

# Asserts that the ActionExecuted event for a remotely executed action references
# the given stream ("stdout" or "stderr") by a bytestream:// URI whose digest
# matches the expected contents and whose blob is present in the CAS.
function expect_bes_action_stdouterr_in_cas() {
  local stream=$1
  local expected_contents=$2
  local expected_hash
  expected_hash="$(printf '%s\n' "$expected_contents" | sha256sum | cut -d ' ' -f 1)"

  if [[ ! $(cat $BEP_JSON) =~ \"name\":\"${stream}\",\"uri\":\"bytestream://localhost:${worker_port}/blobs/([0-9a-f]+)/[0-9]+\" ]]; then
    cat $BEP_JSON > $TEST_log
    fail "BEP has no bytestream:// reference for action ${stream}"
  fi
  local actual_hash=${BASH_REMATCH[1]}
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    cat $BEP_JSON > $TEST_log
    fail "BEP ${stream} digest ${actual_hash} does not match expected ${expected_hash}"
  fi
  if ! remote_cas_file_exist "$actual_hash"; then
    cat $BEP_JSON >> $TEST_log && append_remote_cas_files $TEST_log
    fail "BEP ${stream} blob ${actual_hash} is not in the CAS"
  fi
}

function write_stdouterr_genrule() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.txt"],
  cmd = "echo some_stdout; echo some_stderr 1>&2; touch $@",
)
EOF
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

  expect_bes_file_uploaded foo.txt
  expect_bes_file_uploaded command.profile.gz
  remote_cas_files="$(count_remote_cas_files)"
  [[ "$remote_cas_files" == 2 ]] || fail "Expected 2 remote cas entries, not $remote_cas_files"
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
  add_rules_shell "MODULE.bazel"
  mkdir -p a
  cat > a/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
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
  add_rules_shell "MODULE.bazel"
  mkdir -p a
  cat > a/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
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

  expect_bes_file_uploaded "command.profile.gz"
}

function test_upload_minimal_upload_compact_exec_log() {
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
      --experimental_execution_log_compact_file=myexeclog \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded "execution_log.binpb.zst"
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

  expect_bes_file_uploaded "command.profile.gz"
}

function test_upload_upload_uncompressed_profile() {
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
      --profile=mycommand.profile \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_bes_file_uploaded "command.profile.json"
}

function test_publish_all_actions_stdouterr_download_all() {
  # Baseline: with the default --remote_download_stdouterr=all the action
  # stdout/stderr are downloaded and still referenced by their CAS entries.
  write_stdouterr_genrule

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --remote_download_stdouterr=all \
      --build_event_publish_all_actions \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_log "some_stdout"
  expect_log "some_stderr"
  expect_bes_action_stdouterr_in_cas "stdout" "some_stdout"
  expect_bes_action_stdouterr_in_cas "stderr" "some_stderr"
}

function test_publish_all_actions_stdouterr_download_failed() {
  # --remote_download_stdouterr=failed skips the download for successful actions,
  # but BEP must still reference the CAS entries.
  write_stdouterr_genrule

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --remote_download_stdouterr=failed \
      --build_event_publish_all_actions \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_not_log "some_stdout"
  expect_not_log "some_stderr"
  expect_bes_action_stdouterr_in_cas "stdout" "some_stdout"
  expect_bes_action_stdouterr_in_cas "stderr" "some_stderr"
}

function test_publish_all_actions_stdouterr_download_uncached() {
  # Same as above, but for a cache hit under --remote_download_stdouterr=uncached.
  write_stdouterr_genrule

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --remote_download_stdouterr=uncached \
      --build_event_publish_all_actions \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_log "some_stdout"
  expect_log "some_stderr"

  bazel clean >& $TEST_log || fail "Failed to clean"

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=minimal \
      --remote_download_stdouterr=uncached \
      --build_event_publish_all_actions \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_not_log "some_stdout"
  expect_not_log "some_stderr"
  expect_bes_action_stdouterr_in_cas "stdout" "some_stdout"
  expect_bes_action_stdouterr_in_cas "stderr" "some_stderr"
}

function test_publish_all_actions_stdouterr_download_failed_upload_all() {
  # --remote_build_event_upload=all must not try to upload the stdout/stderr that
  # were never downloaded; they are already in the CAS.
  write_stdouterr_genrule

  bazel build \
      --remote_executor=grpc://localhost:${worker_port} \
      --remote_build_event_upload=all \
      --remote_download_stdouterr=failed \
      --build_event_publish_all_actions \
      --build_event_json_file=$BEP_JSON \
      //a:foo >& $TEST_log || fail "Failed to build"

  expect_not_log "Uploading BEP referenced local file"
  expect_bes_action_stdouterr_in_cas "stdout" "some_stdout"
  expect_bes_action_stdouterr_in_cas "stderr" "some_stderr"
}

run_suite "Remote build event uploader tests"
