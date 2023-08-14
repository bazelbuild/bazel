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
# Test hermetic Linux sandbox
#


# Load test environment
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/../sandboxing_test_utils.sh \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }

cat >>$TEST_TMPDIR/bazelrc <<'EOF'
# Testing the sandboxed strategy requires using the sandboxed strategy. While it is the default,
# we want to make sure that this explicitly fails when the strategy is not available on the system
# running the test.
# The hermetic sandbox requires the Linux sandbox.
build --spawn_strategy=sandboxed
build --experimental_use_hermetic_linux_sandbox
build --sandbox_fake_username
EOF

# For the test to work we need to bind mount a couple of folders to
# get access to bash, ls, python etc. Depending on linux distribution
# these folders may vary. Mount all folders in the root directory '/'
# except the project directory, the directory containing the bazel
# workspace under test.
project_folder=`pwd | cut -d"/" -f 2`
for folder in /*/
do
  if [ -d "$folder" ] && [ "$folder" != "/$project_folder/" ]
  then
    if [[ -L $folder ]]
    then
      # Get resolved link
      linked_folder=`readlink -f $folder`
      echo "build --sandbox_add_mount_pair=/$linked_folder:$folder" >> $TEST_TMPDIR/bazelrc
    else
      echo "build --sandbox_add_mount_pair=$folder" >> $TEST_TMPDIR/bazelrc
    fi
  fi
done

function set_up {
  export BAZEL_GENFILES_DIR=$(bazel info bazel-genfiles 2>/dev/null)
  export BAZEL_BIN_DIR=$(bazel info bazel-bin 2>/dev/null)

  sed -i.bak '/sandbox_tmpfs_path/d' $TEST_TMPDIR/bazelrc

  mkdir -p examples/hermetic

  cat << 'EOF' > examples/hermetic/unknown_file.txt
text inside this file
EOF

  ABSOLUTE_PATH=$CURRENT_DIR/workspace/examples/hermetic/unknown_file.txt

  # In this case the ABSOLUTE_PATH will be expanded
  # and the absolute path will be written to script_absolute_path.sh
  cat << EOF > examples/hermetic/script_absolute_path.sh
#! /bin/sh
ls ${ABSOLUTE_PATH}
EOF

  chmod 777 examples/hermetic/script_absolute_path.sh

  cat << 'EOF' > examples/hermetic/script_symbolic_link.sh
#! /bin/sh
OUTSIDE_SANDBOX_DIR=$(dirname $(realpath $0))
cat $OUTSIDE_SANDBOX_DIR/unknown_file.txt
EOF

  chmod 777 examples/hermetic/script_symbolic_link.sh

  touch examples/hermetic/import_module.py

  cat << 'EOF' > examples/hermetic/py_module_test.py
import import_module
EOF

  cat << 'EOF' > examples/hermetic/BUILD

load(
  "test.bzl",
  "overwrite_via_symlink",
  "overwrite_file_from_declared_directory",
  "subdirectories_in_declared_directory",
  "other_artifacts",
)

overwrite_via_symlink(
  name = "overwrite_via_resolved_symlink",
  resolve_symlink = True
)

overwrite_via_symlink(
  name = "overwrite_via_unresolved_symlink",
  resolve_symlink = False
)

overwrite_file_from_declared_directory(
  name = "overwrite_file_from_declared_directory"
)

subdirectories_in_declared_directory(
  name = "subdirectories_in_declared_directory"
)

other_artifacts(
  name = "other_artifacts"
)

genrule(
  name = "absolute_path",
  srcs = ["script_absolute_path.sh"], # unknown_file.txt not referenced.
  outs = [ "absolute_path.txt" ],
  cmd = "./$(location :script_absolute_path.sh) > $@",
)

genrule(
  name = "symbolic_link",
  srcs = ["script_symbolic_link.sh"], # unknown_file.txt not referenced.
  outs = ["symbolic_link.txt"],
  cmd = "./$(location :script_symbolic_link.sh) > $@",
)

py_test(
  name = "py_module_test",
  srcs = ["py_module_test.py"],  # import_module.py not referenced.
  size = "small",
)

genrule(
  name = "input_file",
  outs = ["input_file.txt"],
  cmd = "echo original text input > $@",
)

genrule(
  name = "write_input_test",
  srcs = [":input_file"],
  outs = ["status.txt"],
  cmd = "(chmod 777 $(location :input_file) && \
         (echo overwrite text > $(location :input_file)) && \
         (echo success > $@)) || (echo fail > $@)",
)
EOF

  cat << 'EOF' > examples/hermetic/test.bzl

def _overwrite_via_symlink_impl(ctx):
    file = ctx.actions.declare_file(ctx.attr.name + ".file")
    if ctx.attr.resolve_symlink:
        symlink = ctx.actions.declare_file(ctx.attr.name + ".symlink")
    else:
        symlink = ctx.actions.declare_symlink(ctx.attr.name + ".symlink")

    ctx.actions.write(file, "")

    if ctx.attr.resolve_symlink:
        ctx.actions.symlink(
            output = symlink,
            target_file = file
        )
        # Symlink become resolved to RegularFileArtifactValue.
        needed_inputs = [symlink]
    else:
        ctx.actions.symlink(
            output = symlink,
            target_path = file.basename
        )
        # Symlink become UnresolvedSymlinkArtifactValue and would be
        # dangling unless also providing the actual file as input to sandbox.
        needed_inputs = [symlink, file]

    result_file = ctx.actions.declare_file(ctx.attr.name + ".result")

    # Try invalid write to the input file via the symlink
    ctx.actions.run_shell(
       command = "chmod u+w $1 && echo hello >> $1 && ls -lR > $2",
       arguments = [symlink.path, result_file.path],
       inputs = needed_inputs,
       outputs = [result_file],
    )

    return [DefaultInfo(files = depset([result_file]))]

overwrite_via_symlink = rule(
    attrs = {
        "resolve_symlink" : attr.bool(),
    },
    implementation = _overwrite_via_symlink_impl,
)


def _overwrite_file_from_declared_directory_impl(ctx):
    dir = ctx.actions.declare_directory(ctx.attr.name + ".dir")

    ctx.actions.run_shell(
        command = "mkdir -p $1/subdir && touch $1/subdir/file",
        arguments = [dir.path],
        outputs = [dir],
    )

    # Try invalid write to input file, with file as implicit input
    # from declared directory.
    result_file = ctx.actions.declare_file(ctx.attr.name + ".result")
    ctx.actions.run_shell(
       command = "chmod -R u+w $1 && echo hello >> $1/subdir/file && touch $2",
       arguments = [dir.path, result_file.path],
       inputs = [dir],
       outputs = [result_file],
    )

    return [DefaultInfo(files = depset([result_file]))]

overwrite_file_from_declared_directory = rule(
    implementation = _overwrite_file_from_declared_directory_impl,
)


def _subdirectories_in_declared_directory_impl(ctx):
    dir = ctx.actions.declare_directory(ctx.attr.name + ".dir")

    ctx.actions.run_shell(
        command = "mkdir -p %s/subdir1/subdir2" % dir.path,
        outputs = [dir],
    )

    result_file = ctx.actions.declare_file(ctx.attr.name + ".result")
    ctx.actions.run_shell(
        command = "ls -lRH %s > %s" % (dir.path, result_file.path),
        inputs = [dir],
        outputs = [result_file],
    )

    return [DefaultInfo(files = depset([result_file]))]

subdirectories_in_declared_directory = rule(
    implementation = _subdirectories_in_declared_directory_impl,
)


def _other_artifacts_impl(ctx):

    # Produce artifacts of other types

    regular_file_artifact = ctx.actions.declare_file(ctx.attr.name + ".regular_file_artifact")
    directory_artifact = ctx.actions.declare_file(ctx.attr.name + ".directory_artifact")
    tree_artifact = ctx.actions.declare_directory(ctx.attr.name + ".tree_artifact")
    unresolved_symlink_artifact = ctx.actions.declare_symlink(ctx.attr.name + ".unresolved_symlink_artifact")

    ctx.actions.run_shell(
        command = "touch %s && mkdir %s" % (regular_file_artifact.path, directory_artifact.path),
        outputs = [regular_file_artifact, tree_artifact, directory_artifact],
    )

    ctx.actions.symlink(
        output = unresolved_symlink_artifact,
        target_path="dangling"
    )

    # Test other artifact types as input to hermetic sandbox.

    all_artifacts = [regular_file_artifact,
                     directory_artifact,
                     tree_artifact,
                     unresolved_symlink_artifact]
    input_paths_string = " ".join([a.path for a in all_artifacts])
    result_file = ctx.actions.declare_file(ctx.attr.name + ".result")
    ctx.actions.run_shell(
        command = "ls -lR %s > %s" % (input_paths_string, result_file.path),
        inputs = all_artifacts,
        outputs = [result_file],
    )

    return [DefaultInfo(files = depset([result_file]))]

other_artifacts = rule(
    implementation = _other_artifacts_impl,
)
EOF
}

# Test that the build can't escape the sandbox via absolute path.
function test_absolute_path() {
  bazel build examples/hermetic:absolute_path &> $TEST_log \
    && fail "Fail due to non hermetic sandbox: examples/hermetic:absolute_path" || true
  expect_log "ls:.* '\?.*/examples/hermetic/unknown_file.txt'\?: No such file or directory"
}

# Test that the build can't escape the sandbox by resolving symbolic link.
function test_symbolic_link() {
  bazel build examples/hermetic:symbolic_link &> $TEST_log \
    && fail "Fail due to non hermetic sandbox: examples/hermetic:symbolic_link" || true
  expect_log "cat: \/execroot\/main\/examples\/hermetic\/unknown_file.txt: No such file or directory"
}

# Test that the sandbox discover if the bazel python rule miss dependencies.
function test_missing_python_deps() {
  bazel test examples/hermetic:py_module_test --test_output=all &> $TEST_TMPDIR/log \
    && fail "Fail due to non hermetic sandbox: examples/hermetic:py_module_test" || true

  expect_log "No module named '\?import_module'\?"
}

# Test that the intermediate corrupt input file gets re:evaluated
function test_writing_input_file() {
  # Write an input file, this should cause the hermetic sandbox to fail with an exception
  bazel build examples/hermetic:write_input_test &> $TEST_log  \
    && fail "Fail due to non hermetic sandbox: examples/hermetic:write_input_test" || true
  expect_log "input dependency .*examples/hermetic/input_file.txt was modified during execution."
  cat "${BAZEL_GENFILES_DIR}/examples/hermetic/input_file.txt" &> $TEST_log
  expect_log "overwrite text"

  # Build the input file again, this should not use the cache, but instead re:evaluate the file
  bazel build examples/hermetic:input_file &> $TEST_log \
    || fail "Fail due to non hermetic sandbox: examples/hermetic:input_file"
  [ -f "${BAZEL_GENFILES_DIR}/examples/hermetic/input_file.txt" ] \
    || fail "Genrule did not produce output: examples/hermetic:input_file"
  cat "${BAZEL_GENFILES_DIR}/examples/hermetic/input_file.txt" &> $TEST_log
  expect_log "original text input"
}

# Test that invalid write of input file is detected, when file is accessed via resolved symlink.
function test_overwrite_via_resolved_symlink() {
  bazel build examples/hermetic:overwrite_via_resolved_symlink &> $TEST_log  \
    && fail "Hermetic sandbox did not detect invalid write to input file"
  expect_log "input dependency .* was modified during execution."
}

# Test that invalid write of input file is detected, when file is accessed via unresolved symlink.
function test_overwrite_via_unresolved_symlink() {
  bazel build examples/hermetic:overwrite_via_unresolved_symlink &> $TEST_log  \
    && fail "Hermetic sandbox did not detect invalid write to input file"
  expect_log "input dependency .* was modified during execution."
}

# Test that invalid write of input file is detected, when file is found implicit via declared directory.
function test_overwrite_file_from_declared_directory() {
  bazel build examples/hermetic:overwrite_file_from_declared_directory &> $TEST_log  \
    && fail "Hermetic sandbox did not detect invalid write to input file"
  expect_log "input dependency .* was modified during execution."
}

# Test that the sandbox can handle deep directory trees from declared directory.
function test_subdirectories_in_declared_directory() {
  bazel build examples/hermetic:subdirectories_in_declared_directory &> $TEST_log
  cat bazel-bin/examples/hermetic/subdirectories_in_declared_directory.result
  assert_contains "dir/subdir1/subdir2" "bazel-bin/examples/hermetic/subdirectories_in_declared_directory.result"
}

# Test that the sandbox is able to handle various types of artifacts.
# Regression test for Issue #15340
function test_other_artifacts() {
  bazel build examples/hermetic:other_artifacts &> $TEST_log
  assert_contains ".regular_file_artifact" "bazel-bin/examples/hermetic/other_artifacts.result"
  assert_contains ".unresolved_symlink_artifact" "bazel-bin/examples/hermetic/other_artifacts.result"
  assert_contains ".directory_artifact" "bazel-bin/examples/hermetic/other_artifacts.result"
  assert_contains ".tree_artifact" "bazel-bin/examples/hermetic/other_artifacts.result"
}

# The test shouldn't fail if the environment doesn't support running it.
check_sandbox_allowed || exit 0
[ "$PLATFORM" != "darwin" ] || exit 0

run_suite "hermetic_sandbox"
