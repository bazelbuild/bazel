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
example_worker=$(find $BAZEL_RUNFILES -name ExampleWorkerMultiplexer_deploy.jar)

add_to_bazelrc "build -s"
add_to_bazelrc "build --spawn_strategy=worker,standalone"
add_to_bazelrc "build --worker_verbose --worker_max_instances=1"
add_to_bazelrc "build --debug_print_action_contexts"
add_to_bazelrc "build ${ADDITIONAL_BUILD_FLAGS}"

function set_up() {
  # Run each test in a separate folder so that their output files don't get cached.
  WORKSPACE_SUBDIR=$(basename $(mktemp -d ${WORKSPACE_DIR}/testXXXXXX))
  cd ${WORKSPACE_SUBDIR}
  BINS=$(bazel info $PRODUCT_NAME-bin)/${WORKSPACE_SUBDIR}

  # This causes Bazel to shut down all running workers.
  bazel build --worker_quit_after_build &> $TEST_log \
    || fail "'bazel build --worker_quit_after_build' during test set_up failed"
}

function prepare_example_worker() {
  cp ${example_worker} worker_lib.jar
  chmod +w worker_lib.jar
  echo "exampledata" > worker_data.txt

  mkdir worker_data_dir
  echo "veryexample" > worker_data_dir/more_data.txt

  cat >work.bzl <<'EOF'
def _impl(ctx):
  worker = ctx.executable.worker
  output = ctx.outputs.out

  argfile_inputs = []
  argfile_arguments = []
  if ctx.attr.multiflagfiles:
    # Generate one flagfile per command-line arg, alternate between @ and --flagfile= style.
    # This is used to test the code that handles multiple flagfiles and the --flagfile= style.
    idx = 1
    for arg in ["--output_file=" + output.path] + ctx.attr.args:
      argfile = ctx.actions.declare_file("%s_worker_input_%s" % (ctx.label.name, idx))
      ctx.actions.write(output=argfile, content=arg)
      argfile_inputs.append(argfile)
      flagfile_prefix = "@" if (idx % 2 == 0) else "--flagfile="
      argfile_arguments.append(flagfile_prefix + argfile.path)
      idx += 1
  else:
    # Generate the "@"-file containing the command-line args for the unit of work.
    argfile = ctx.actions.declare_file("%s_worker_input" % ctx.label.name)
    argfile_contents = "\n".join(["--output_file=" + output.path] + ctx.attr.args)
    ctx.actions.write(output=argfile, content=argfile_contents)
    argfile_inputs.append(argfile)
    argfile_arguments.append("@" + argfile.path)

  ctx.actions.run(
      inputs=argfile_inputs + ctx.files.srcs,
      outputs=[output],
      executable=worker,
      progress_message="Working on %s" % ctx.label.name,
      mnemonic="Work",
      execution_requirements={"supports-multiplex-workers": "1"},
      arguments=ctx.attr.worker_args + argfile_arguments,
  )

work = rule(
    implementation=_impl,
    attrs={
        "worker": attr.label(cfg="host", mandatory=True, allow_files=True, executable=True),
        "worker_args": attr.string_list(),
        "args": attr.string_list(),
        "srcs": attr.label_list(allow_files=True),
        "multiflagfiles": attr.bool(default=False),
    },
    outputs = {"out": "%{name}.out"},
)
EOF
  cat >BUILD <<EOF
load(":work.bzl", "work")

java_import(
  name = "worker_lib",
  jars = ["worker_lib.jar"],
)

java_binary(
  name = "worker",
  main_class = "com.google.devtools.build.lib.worker.ExampleWorkerMultiplexer",
  runtime_deps = [
    ":worker_lib",
  ],
  data = [
    ":worker_data.txt",
    ":worker_data_dir",
  ]
)
EOF
}

function test_example_worker_multiplexer() {
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

  bazel build  :hello_world &> $TEST_log \
    || fail "build failed"
  assert_equals "hello world" "$(cat $BINS/hello_world.out)"

  bazel build  :hello_world_uppercase &> $TEST_log \
    || fail "build failed"
  assert_equals "HELLO WORLD" "$(cat $BINS/hello_world_uppercase.out)"
}

function test_multiple_target_without_delay() {
  prepare_example_worker
  cat >>BUILD <<EOF
work(
  name = "hello_world_1",
  worker = ":worker",
  args = ["hello world 1"],
)

work(
  name = "hello_world_2",
  worker = ":worker",
  args = ["hello world 2"],
)

work(
  name = "hello_world_3",
  worker = ":worker",
  args = ["hello world 3"],
)
EOF

  bazel build  :hello_world_1 :hello_world_2 :hello_world_3 &> $TEST_log \
    || fail "build failed"
  assert_equals "hello world 1" "$(cat $BINS/hello_world_1.out)"
  assert_equals "hello world 2" "$(cat $BINS/hello_world_2.out)"
  assert_equals "hello world 3" "$(cat $BINS/hello_world_3.out)"
}
run_suite "Worker multiplexer integration tests"
