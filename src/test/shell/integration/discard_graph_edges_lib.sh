#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# discard_graph_edges_lib.sh: functions needed by discard_graph_edges_test.sh

STARTUP_FLAGS="--batch"
BUILD_FLAGS="--discard_analysis_cache --notrack_incremental_state"

function extract_histogram_count() {
  local histofile="$1"
  local item="$2"
  # We can't use + here because Macs don't recognize it as a special character
  # by default.
  grep "$item" "$histofile" \
      | sed -e 's/^ *[0-9][0-9]*: *\([0-9][0-9]*\) .*$/\1/' \
      || fail "Couldn't get item from $histofile"
}

function run_test_actions_deleted_after_execution() {
  readonly local product="$1"
  readonly local javabase="$2"
  readonly local get_pid_expression="$3"
  readonly local extra_build_arg="$4"
  rm -rf histodump
  mkdir -p histodump || fail "Couldn't create directory"
  readonly local wait_fifo="$TEST_TMPDIR/wait_fifo"
  readonly local exec_fifo="$TEST_TMPDIR/exec_fifo"
  readonly local server_pid_file="$TEST_TMPDIR/server_pid.txt"
  cat > histodump/BUILD <<EOF || fail "Couldn't create BUILD file"
genrule(name = 'action0',
        outs = ['wait.out'],
        local = 1,
        cmd = 'echo "" > $exec_fifo; cat $wait_fifo > /dev/null; touch \$@'
        )
EOF
  for i in $(seq 1 3); do
    iminus=$((i-1))
    cat >> histodump/BUILD <<EOF || fail "Couldn't append"
genrule(name = 'action${i}',
        srcs = [':action${iminus}'],
        outs = ['histo.${i}'],
        local = 1,
        cmd = 'server_pid=\$\$(cat $server_pid_file) ; ' +
              '$javabase/bin/jmap -histo:live \$\$server_pid > ' +
              '\$(location histo.${i}) ' +
              '|| echo "server_pid in genrule: \$\$server_pid"'
       )
EOF
  done
  mkfifo "$wait_fifo" "$exec_fifo"
  local readonly histo_root="$("$product" info \
      "${PRODUCT_NAME:-$product}-genfiles" 2> /dev/null)/histodump/histo."
  "$product" clean >& "$TEST_log" || fail "Couldn't clean"
  readonly local explicit_server_pid="$("$product" $STARTUP_FLAGS info \
      server_pid)"
  "$product" $STARTUP_FLAGS build --show_timestamps $BUILD_FLAGS \
      $extra_build_arg //histodump:action3 >> "$TEST_log" 2>&1 &
  subshell_pid="$!"
  cat "$exec_fifo" > /dev/null
  # We plan to remove batch mode from the relevant flags for discarding
  # incrementality state. In the interim, tests that are not in batch mode
  # explicitly pass --nobatch, so we can use it as a signal.
  if [[ "$STARTUP_FLAGS" =~ "--nobatch" ]]; then
    server_pid="$explicit_server_pid"
  else
    if [[ -z "$get_pid_expression" ]]; then
      server_pid="$subshell_pid"
    else
      server_pid="$($get_pid_expression)"
    fi
  fi
  echo "server_pid in main thread is ${server_pid}" # >> "$TEST_log"
  echo "$server_pid" > "$server_pid_file"
  echo "Finished writing pid to fifo at " >> "$TEST_log"
  date >> "$TEST_log"
  echo "" > "$wait_fifo"
  # Wait for previous command to finish.
  wait "$subshell_pid" || fail "Bazel command failed"
  local genrule_action_count=100
  for i in $(seq 1 3); do
    local histo_file="$histo_root$i"
    local new_genrule_action_count="$(extract_histogram_count "$histo_file" \
        "GenRuleAction$")"
    if [[ "$new_genrule_action_count" -ge "$genrule_action_count" ]]; then
      cat "$histo_file" >> "$TEST_log"
      fail "Number of genrule actions did not decrease: $new_genrule_action_count vs. $genrule_action_count"
    fi
    if [[ -z "$new_genrule_action_count" ]]; then
      cat "$histo_file" >> "$TEST_log"
      fail "No genrule actions? Class may have been renamed"
    fi
    genrule_action_count="$new_genrule_action_count"
  done
}
