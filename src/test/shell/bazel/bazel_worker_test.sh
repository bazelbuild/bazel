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
# Test rules provided in Bazel not tested by examples
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# TODO(philwo): Change this so the path to the custom worker gets passed in as an argument to the
# test, once the bug that makes using the "args" attribute with sh_tests in Bazel impossible is
# fixed.
example_worker=$(find $TEST_SRCDIR -name ExampleWorker_deploy.jar)

function set_up() {
  workers=$(print_workers)
  if [[ ! -z "${workers}" ]]; then
    kill $workers

    # Wait at most 10 seconds for all workers to shut down.
    for i in 0 1 2 3 4 5 6 7 8 9; do
      still_running_workers=$(for pid in $workers; do kill -0 $pid &>/dev/null && echo $pid || true; done)

      if [[ ! -z "${still_running_workers}" ]]; then
        if [[ $i -eq 3 ]]; then
          kill -TERM $still_running_workers
        fi

        sleep 1
      fi
    done
  fi

  assert_workers_not_running
}

function write_hello_library_files() {
  mkdir -p java/main
  cat >java/main/BUILD <<EOF
java_binary(name = 'main',
    deps = ['//java/hello_library'],
    srcs = ['Main.java'],
    main_class = 'main.Main')
EOF

  cat >java/main/Main.java <<EOF
package main;
import hello_library.HelloLibrary;
public class Main {
  public static void main(String[] args) {
    HelloLibrary.funcHelloLibrary();
    System.out.println("Hello, World!");
  }
}
EOF

  mkdir -p java/hello_library
  cat >java/hello_library/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'hello_library',
             srcs = ['HelloLibrary.java']);
EOF

  cat >java/hello_library/HelloLibrary.java <<EOF
package hello_library;
public class HelloLibrary {
  public static void funcHelloLibrary() {
    System.out.print("Hello, Library!;");
  }
}
EOF
}

function print_workers() {
  pid=$(bazel info | fgrep server_pid | cut -d' ' -f2)
  pgrep -P $pid || true
}

function shutdown_and_print_unkilled_workers() {
  workers=$(print_workers)
  bazel shutdown || fail "shutdown failed"

  # Wait at most 10 seconds for all workers to shut down, then print the remaining (if any).
  for i in 0 1 2 3 4 5 6 7 8 9; do
    still_running_workers=$(for pid in $workers; do kill -0 $pid &>/dev/null && echo $pid || true; done)
    if [[ ! -z "${still_running_workers}" ]]; then
      sleep 1
    fi
  done

  if [ ! -z "$still_running_workers" ]; then
    fail "Worker processes were still running after shutdown: ${unkilled_workers}"
  fi
}

function assert_workers_running() {
  workers=$(print_workers)
  if [[ -z "${workers}" ]]; then
    fail "Expected workers to be running, but found none"
  fi
}

function assert_workers_not_running() {
  workers=$(print_workers)
  if [[ ! -z "${workers}" ]]; then
    fail "Expected no workers, but found some running: ${workers}"
  fi
}

function test_compiles_hello_library_using_persistent_javac() {
  write_hello_library_files

  bazel build --strategy=Javac=worker //java/main:main || fail "build failed"
  bazel-bin/java/main/main | grep -q "Hello, Library!;Hello, World!" \
    || fail "comparison failed"
  assert_workers_running
  shutdown_and_print_unkilled_workers
}

function test_incremental_heuristic() {
  write_hello_library_files

  # Default strategy is assumed to not use workers.
  bazel build //java/main:main || fail "build failed"
  assert_workers_not_running

  # No workers used, because too many files changed.
  echo '// hello '>> java/hello_library/HelloLibrary.java
  echo '// hello' >> java/main/Main.java
  bazel build --worker_max_changed_files=1 --strategy=Javac=worker //java/main:main \
    || fail "build failed"
  assert_workers_not_running

  # Workers used, because changed number of files is less-or-equal to --worker_max_changed_files=2.
  echo '// again '>> java/hello_library/HelloLibrary.java
  echo '// again' >> java/main/Main.java
  bazel build --worker_max_changed_files=2 --strategy=Javac=worker //java/main:main \
    || fail "build failed"
  assert_workers_running
}

function test_workers_quit_after_build() {
  write_hello_library_files

  bazel build --worker_quit_after_build --strategy=Javac=worker //java/main:main \
    || fail "build failed"
  assert_workers_not_running
}

function prepare_example_worker() {
  cp ${example_worker} worker_lib.jar
  chmod +w worker_lib.jar
  echo "exampledata" > worker_data.txt

  cat >work.bzl <<'EOF'
def _impl(ctx):
  worker = ctx.executable.worker
  output = ctx.outputs.out

  # Generate the "@"-file containing the command-line args for the unit of work.
  argfile = ctx.new_file(ctx.configuration.bin_dir, "%s_worker_input" % ctx.label.name)
  argfile_contents = "\n".join(["--output_file=" + output.path] + ctx.attr.args)
  ctx.file_action(output=argfile, content=argfile_contents)

  ctx.action(
      inputs=[argfile] + ctx.files.srcs,
      outputs=[output],
      executable=worker,
      progress_message="Working on %s" % ctx.label.name,
      mnemonic="Work",
      arguments=ctx.attr.worker_args + ["@" + argfile.path],
  )

work = rule(
    implementation=_impl,
    attrs={
        "worker": attr.label(cfg=HOST_CFG, mandatory=True, allow_files=True, executable=True),
        "worker_args": attr.string_list(),
        "args": attr.string_list(),
        "srcs": attr.label_list(allow_files=True),
    },
    outputs = {"out": "%{name}.out"},
)
EOF
  cat >BUILD <<EOF
load("work", "work")

java_import(
  name = "worker_lib",
  jars = ["worker_lib.jar"],
)

java_binary(
  name = "worker",
  main_class = "com.google.devtools.build.lib.worker.ExampleWorker",
  runtime_deps = [
    ":worker_lib",
  ],
  data = [
    ":worker_data.txt"
  ]
)
EOF
}

function test_example_worker() {
  prepare_example_worker
  cat >>BUILD <<EOF
work(
  name = "hello_world",
  worker = ":worker",
  args = ["hello world"],
)

work(
  name = "hello_world_uppercase",
  worker = ":worker",
  args = ["--uppercase", "hello world"],
)
EOF

  bazel build --strategy=Work=worker :hello_world \
    || fail "build failed"
  assert_equals "hello world" "$(cat bazel-bin/hello_world.out)"
  assert_workers_running

  bazel build --worker_quit_after_build --strategy=Work=worker :hello_world_uppercase \
    || fail "build failed"
  assert_equals "HELLO WORLD" "$(cat bazel-bin/hello_world_uppercase.out)"
  assert_workers_not_running
}

function test_worker_restarts_after_exit() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  worker_args = ["--exit_after=2"],
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_1 \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_2 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count
  assert_workers_not_running

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_3 \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  # Check that we used a new worker.
  assert_not_equals "$worker_uuid_2" "$worker_uuid_3"
}

function test_worker_restarts_when_worker_binary_changes() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_1 \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_2 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count
  assert_workers_running

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  # Modify the example worker jar to trigger a rebuild of the worker.
  tr -cd '[:alnum:]' < /dev/urandom | head -c32 > dummy_file
  zip worker_lib.jar dummy_file
  rm dummy_file

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_3 \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  # Check that we used a new worker.
  assert_not_equals "$worker_uuid_2" "$worker_uuid_3"
}

function test_worker_restarts_when_worker_runfiles_change() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_1 \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_2 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count
  assert_workers_running

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  echo "changeddata" > worker_data.txt

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_3 \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  assert_workers_running

  # Check that we used a new worker.
  assert_not_equals "$worker_uuid_2" "$worker_uuid_3"
}

# When a worker does not conform to the protocol and returns a response that is not a parseable
# protobuf, it must be killed, the output thrown away, a new worker restarted and Bazel has to retry
# the action without struggling.
function test_bazel_recovers_from_worker_returning_junk() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  worker_args = ["--poison_after=1"],
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_1 \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  assert_workers_running

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_2 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  assert_workers_running

  # Check that the worker failed & was restarted.
  assert_not_equals "$worker_uuid_1" "$worker_uuid_2"
}

function test_input_digests() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_uuid", "--print_inputs"],
  srcs = [":input.txt"],
) for idx in range(10)]
EOF

  echo "hello world" > input.txt
  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_1 \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  hash1=$(fgrep "INPUT input.txt " bazel-bin/hello_world_1.out | cut -d' ' -f3)
  assert_workers_running

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_2 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  hash2=$(fgrep "INPUT input.txt " bazel-bin/hello_world_2.out | cut -d' ' -f3)
  assert_workers_running

  assert_equals "$worker_uuid_1" "$worker_uuid_2"
  assert_equals "$hash1" "$hash2"

  echo "changeddata" > input.txt

  bazel build --strategy=Work=worker --worker_max_instances=1 :hello_world_3 \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  hash3=$(fgrep "INPUT input.txt " bazel-bin/hello_world_3.out | cut -d' ' -f3)
  assert_workers_running

  assert_equals "$worker_uuid_2" "$worker_uuid_3"
  assert_not_equals "$hash2" "$hash3"
}

run_suite "Worker integration tests"
