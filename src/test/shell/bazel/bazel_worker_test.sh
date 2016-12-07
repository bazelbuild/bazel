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

set -u
ADDITIONAL_BUILD_FLAGS=$1
WORKER_TYPE_LOG_STRING=$2
shift 2

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# TODO(philwo): Change this so the path to the custom worker gets passed in as an argument to the
# test, once the bug that makes using the "args" attribute with sh_tests in Bazel impossible is
# fixed.
example_worker=$(find $BAZEL_RUNFILES -name ExampleWorker_deploy.jar)

function set_up() {
  # This causes Bazel to shut down all running workers.
  bazel build ${ADDITIONAL_BUILD_FLAGS} --worker_quit_after_build &> $TEST_log
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

function test_compiles_hello_library_using_persistent_javac() {
  write_hello_library_files

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Javac=worker //java/main:main &> $TEST_log \
    || fail "build failed"
  expect_log "Created new ${WORKER_TYPE_LOG_STRING} Javac worker (id [0-9]\+)"
  bazel-bin/java/main/main | grep -q "Hello, Library!;Hello, World!" \
    || fail "comparison failed"
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
  argfile = ctx.new_file(ctx.bin_dir, "%s_worker_input" % ctx.label.name)
  argfile_contents = "\n".join(["--output_file=" + output.path] + ctx.attr.args)
  ctx.file_action(output=argfile, content=argfile_contents)

  ctx.action(
      inputs=[argfile] + ctx.files.srcs,
      outputs=[output],
      executable=worker,
      progress_message="Working on %s" % ctx.label.name,
      mnemonic="Work",
      execution_requirements={"supports-workers": "1"},
      arguments=ctx.attr.worker_args + ["@" + argfile.path],
  )

work = rule(
    implementation=_impl,
    attrs={
        "worker": attr.label(cfg="host", mandatory=True, allow_files=True, executable=True),
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

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker :hello_world &> $TEST_log \
    || fail "build failed"
  assert_equals "hello world" "$(cat bazel-bin/hello_world.out)"

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker :hello_world_uppercase &> $TEST_log \
    || fail "build failed"
  assert_equals "HELLO WORLD" "$(cat bazel-bin/hello_world_uppercase.out)"
}

function test_workers_quit_after_build() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_counter"],
) for idx in range(10)]
EOF

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world_1 &> $TEST_log \
    || fail "build failed"
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world_2 &> $TEST_log \
    || fail "build failed"
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  # If the worker hadn't quit as we told it, it would have been reused, causing this to be a "2".
  assert_equals "1" $work_count
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

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_1 &> $TEST_log \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_2 &> $TEST_log \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_3 &> $TEST_log \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count
  expect_log "worker .* can no longer be used, because its process terminated itself or got killed"

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

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_1 &> $TEST_log \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_2 &> $TEST_log \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  # Modify the example worker jar to trigger a rebuild of the worker.
  tr -cd '[:alnum:]' < /dev/urandom | head -c32 > dummy_file
  zip worker_lib.jar dummy_file
  rm dummy_file

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_3 &> $TEST_log \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  expect_log "worker .* can no longer be used, because its files have changed on disk"

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

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_1 &> $TEST_log \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_1.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_2 &> $TEST_log \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_2.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "2" $work_count

  # Check that the same worker was used twice.
  assert_equals "$worker_uuid_1" "$worker_uuid_2"

  # "worker_data.txt" is included in the "data" attribute of the example worker.
  echo "changeddata" > worker_data.txt

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_3 &> $TEST_log \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  work_count=$(cat bazel-bin/hello_world_3.out | grep COUNTER | cut -d' ' -f2)
  assert_equals "1" $work_count

  expect_log "worker .* can no longer be used, because its files have changed on disk"

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

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_1 &> $TEST_log \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_2 &> $TEST_log \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)

  # Check that the worker failed & was restarted.
  expect_log "invalidating and retrying with new worker"
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
  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_1 &> $TEST_log \
    || fail "build failed"
  worker_uuid_1=$(cat bazel-bin/hello_world_1.out | grep UUID | cut -d' ' -f2)
  hash1=$(fgrep "INPUT input.txt " bazel-bin/hello_world_1.out | cut -d' ' -f3)

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_2 >> $TEST_log 2>&1 \
    || fail "build failed"
  worker_uuid_2=$(cat bazel-bin/hello_world_2.out | grep UUID | cut -d' ' -f2)
  hash2=$(fgrep "INPUT input.txt " bazel-bin/hello_world_2.out | cut -d' ' -f3)

  assert_equals "$worker_uuid_1" "$worker_uuid_2"
  assert_equals "$hash1" "$hash2"

  echo "changeddata" > input.txt

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 :hello_world_3 >> $TEST_log 2>&1 \
    || fail "build failed"
  worker_uuid_3=$(cat bazel-bin/hello_world_3.out | grep UUID | cut -d' ' -f2)
  hash3=$(fgrep "INPUT input.txt " bazel-bin/hello_world_3.out | cut -d' ' -f3)

  assert_equals "$worker_uuid_2" "$worker_uuid_3"
  assert_not_equals "$hash2" "$hash3"
}

function test_worker_verbose() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world_1 &> $TEST_log \
    || fail "build failed"
  expect_log "Created new ${WORKER_TYPE_LOG_STRING} Work worker (id [0-9]\+)"
  expect_log "Destroying Work worker (id [0-9]\+)"
  expect_log "Build completed, shutting down worker pool..."
}

function test_logs_are_deleted_on_server_restart() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
[work(
  name = "hello_world_%s" % idx,
  worker = ":worker",
  args = ["--write_uuid", "--write_counter"],
) for idx in range(10)]
EOF

  bazel build ${ADDITIONAL_BUILD_FLAGS} -s --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world_1 &> $TEST_log \
    || fail "build failed"

  expect_log "Created new ${WORKER_TYPE_LOG_STRING} Work worker (id [0-9]\+)"

  worker_log=$(egrep -o -- 'logging to .*/bazel-workers/worker-[0-9]-Work.log' "$TEST_log" | sed 's/^logging to //')

  [ -e "$worker_log" ] \
    || fail "Worker log was not found"

  # Running a build after a server shutdown should trigger the removal of old worker log files.
  bazel shutdown &> $TEST_log
  bazel build &> $TEST_log

  [ ! -e "$worker_log" ] \
    || fail "Worker log was not deleted"
}

function test_missing_execution_requirements_gives_warning() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
work(
  name = "hello_world",
  worker = ":worker",
  args = ["--write_uuid", "--write_counter"],
)
EOF

  sed -i.bak '/execution_requirements/d' work.bzl
  rm -f work.bzl.bak

  bazel build ${ADDITIONAL_BUILD_FLAGS} --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world &> $TEST_log \
    || fail "build failed"

  expect_log "Worker strategy cannot execute this Work action, because the action's execution info does not contain 'supports-workers=1'"
  expect_not_log "Created new ${WORKER_TYPE_LOG_STRING} Work worker (id [0-9]\+)"
  expect_not_log "Destroying Work worker (id [0-9]\+)"

  # WorkerSpawnStrategy falls back to standalone strategy, so we still expect the output to be generated.
  [ -e "bazel-bin/hello_world.out" ] \
    || fail "Worker did not produce output"
}

function test_environment_is_clean() {
  prepare_example_worker
  cat >>BUILD <<'EOF'
work(
  name = "hello_world",
  worker = ":worker",
  args = ["--print_env"],
)
EOF

  bazel shutdown &> $TEST_log \
    || fail "shutdown failed"
  CAKE=LIE bazel build ${ADDITIONAL_BUILD_FLAGS} --worker_verbose --strategy=Work=worker --worker_max_instances=1 --worker_quit_after_build :hello_world &> $TEST_log \
    || fail "build failed"

  fgrep CAKE=LIE bazel-bin/hello_world.out \
    && fail "environment variable leaked into worker env" || true
}

run_suite "Worker integration tests"
