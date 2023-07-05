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
# Test the local_repository binding
#

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

source "$(rlocation "io_bazel/src/test/shell/bazel/remote_helpers.sh")" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

mock_rules_java_to_avoid_downloading

# Basic test.
function test_macro_local_repository() {
  create_new_workspace
  repo2=$new_workspace_dir

  mkdir -p carnivore
  cat > carnivore/BUILD <<'EOF'
genrule(
    name = "mongoose",
    cmd = "echo 'Tra-la!' | tee $@",
    outs = ["moogoose.txt"],
    visibility = ["//visibility:public"],
)
EOF

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load('//:test.bzl', 'macro')

macro('$repo2')
EOF

  # Empty package for the .bzl file
  echo -n >BUILD

  # Our macro
  cat >test.bzl <<EOF
def macro(path):
  print('bleh')
  native.local_repository(name='endangered', path=path)
  native.bind(name='mongoose', actual='@endangered//carnivore:mongoose')
EOF
  mkdir -p zoo
  cat > zoo/BUILD <<'EOF'
genrule(
    name = "ball-pit1",
    srcs = ["@endangered//carnivore:mongoose"],
    outs = ["ball-pit1.txt"],
    cmd = "cat $< >$@",
)

genrule(
    name = "ball-pit2",
    srcs = ["//external:mongoose"],
    outs = ["ball-pit2.txt"],
    cmd = "cat $< >$@",
)
EOF

  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_log "bleh"
  expect_log "Tra-la!"  # Invalidation
  cat bazel-bin/zoo/ball-pit1.txt >$TEST_log
  expect_log "Tra-la!"

  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la!"  # No invalidation

  bazel build //zoo:ball-pit2 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la!"  # No invalidation
  cat bazel-bin/zoo/ball-pit2.txt >$TEST_log
  expect_log "Tra-la!"

  # Test invalidation of the WORKSPACE file
  create_new_workspace
  repo2=$new_workspace_dir

  mkdir -p carnivore
  cat > carnivore/BUILD <<'EOF'
genrule(
    name = "mongoose",
    cmd = "echo 'Tra-la-la!' | tee $@",
    outs = ["moogoose.txt"],
    visibility = ["//visibility:public"],
)
EOF
  cd ${WORKSPACE_DIR}
  cat >test.bzl <<EOF
def macro(path):
  print('blah')
  native.local_repository(name='endangered', path='$repo2')
  native.bind(name='mongoose', actual='@endangered//carnivore:mongoose')
EOF
  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_log "blah"
  expect_log "Tra-la-la!"  # Invalidation
  cat bazel-bin/zoo/ball-pit1.txt >$TEST_log
  expect_log "Tra-la-la!"

  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la-la!"  # No invalidation

  bazel build //zoo:ball-pit2 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la-la!"  # No invalidation
  cat bazel-bin/zoo/ball-pit2.txt >$TEST_log
  expect_log "Tra-la-la!"
}

function test_load_from_symlink_to_outside_of_workspace() {
  OTHER=$TEST_TMPDIR/other

  cat >> $(create_workspace_with_default_repos WORKSPACE)<<EOF
load("//a/b:c.bzl", "c")
EOF

  mkdir -p $OTHER/a/b
  touch $OTHER/a/b/BUILD
  cat > $OTHER/a/b/c.bzl <<EOF
def c():
  pass
EOF

  touch BUILD
  ln -s $TEST_TMPDIR/other/a a
  bazel build //:BUILD || fail "Failed to build"
  rm -fr $TEST_TMPDIR/other
}

# Test load from repository.
function test_external_load_from_workspace() {
  create_new_workspace
  repo2=$new_workspace_dir

  mkdir -p carnivore
  cat > carnivore/BUILD <<'EOF'
genrule(
    name = "mongoose",
    cmd = "echo 'Tra-la-la!' | tee $@",
    outs = ["moogoose.txt"],
    visibility = ["//visibility:public"],
)
EOF

  create_new_workspace
  repo3=$new_workspace_dir
  # Our macro
  cat >WORKSPACE
  cat >test.bzl <<EOF
def macro(path):
  print('bleh')
  native.local_repository(name='endangered', path=path)
EOF
  cat >BUILD <<'EOF'
exports_files(["test.bzl"])
EOF

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
local_repository(name='proxy', path='$repo3')
load('@proxy//:test.bzl', 'macro')
macro('$repo2')
EOF

  bazel build @endangered//carnivore:mongoose >& $TEST_log \
    || fail "Failed to build"
  expect_log "bleh"
}

# Test loading a repository with a load statement in the WORKSPACE file
function test_load_repository_with_load() {
  create_new_workspace
  repo2=$new_workspace_dir

  echo "Tra-la!" > data.txt
  cat <<'EOF' >BUILD
exports_files(["data.txt"])
EOF

  cat <<'EOF' >ext.bzl
def macro():
  print('bleh')
EOF

  cat <<'EOF' >WORKSPACE
workspace(name = "foo")
load("//:ext.bzl", "macro")
macro()
EOF

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
local_repository(name='foo', path='$repo2')
load("@foo//:ext.bzl", "macro")
macro()
EOF

  cat > BUILD <<'EOF'
genrule(name = "foo", srcs=["@foo//:data.txt"], outs=["foo.txt"], cmd = "cat $< | tee $@")
EOF

  bazel build //:foo >& $TEST_log || fail "Failed to build"
  expect_log "bleh"
  expect_log "Tra-la!"
}

# Test cycle when loading a repository with a load statement in the WORKSPACE file that is not
# yet defined.
function test_cycle_load_repository() {
  create_new_workspace
  repo2=$new_workspace_dir

  echo "Tra-la!" > data.txt
  cat <<'EOF' >BUILD
exports_files(["data.txt"])
EOF

  cat <<'EOF' >ext.bzl
def macro():
  print('bleh')
EOF

  cat >WORKSPACE

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load("@foo//:ext.bzl", "macro")
macro()
local_repository(name='foo', path='$repo2')
EOF

  local exitCode=0
  bazel build @foo//:data.txt >& $TEST_log || exitCode=$?
  [ $exitCode != 0 ] || fail "building @foo//:data.txt succeed while expected failure"

  expect_not_log "PACKAGE"
  expect_log "Failed to load Starlark extension '@foo//:ext.bzl'"
}

function test_load_nonexistent_with_subworkspace() {
  mkdir ws2
  cat >ws2/WORKSPACE

  cat <<'EOF' >WORKSPACE
load("@does_not_exist//:random.bzl", "random")
EOF
  cat >BUILD

  # Test build //...
  bazel clean --expunge
  bazel build //... >& $TEST_log || exitCode=$?
  [ $exitCode != 0 ] || fail "building //... succeed while expected failure"

  expect_not_log "PACKAGE"
  expect_log "Failed to load Starlark extension '@does_not_exist//:random.bzl'"

  # Retest with query //...
  bazel clean --expunge
  bazel query //... >& $TEST_log || exitCode=$?
  [ $exitCode != 0 ] || fail "querying //... succeed while expected failure"

  expect_not_log "PACKAGE"
  expect_log "Failed to load Starlark extension '@does_not_exist//:random.bzl'"
}

function test_starlark_local_repository() {
  create_new_workspace
  repo2=$new_workspace_dir
  # Remove the WORKSPACE file in the symlinked repo, so our Starlark rule has to
  # create one.
  rm $repo2/WORKSPACE

  cat > BUILD <<'EOF'
genrule(name='bar', cmd='echo foo | tee $@', outs=['bar.txt'])
EOF

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load('//:test.bzl', 'repo')
repo(name='foo', path='$repo2')
EOF

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.symlink(repository_ctx.path(repository_ctx.attr.path), repository_ctx.path(""))

repo = repository_rule(
    implementation=_impl,
    local=True,
    attrs={"path": attr.string(mandatory=True)})
EOF
  # Need to be in a package
  cat > BUILD

  bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "foo"
  expect_not_log "Workspace name in .*/WORKSPACE (.*) does not match the name given in the repository's definition (@foo)"
  cat bazel-bin/external/foo/bar.txt >$TEST_log
  expect_log "foo"
}

function setup_starlark_repository() {
  create_new_workspace
  repo2=$new_workspace_dir

  cat > bar.txt
  echo "filegroup(name='bar', srcs=['bar.txt'])" > BUILD

  cd "${WORKSPACE_DIR}"
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load('//:test.bzl', 'repo')
repo(name = 'foo')
EOF
  # Need to be in a package
  cat > BUILD
}

function test_starlark_flags_affect_repository_rule() {
  setup_starlark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  print("In repo rule: ")
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))

repo = repository_rule(implementation=_impl, local=True)
EOF

  MARKER="<== Starlark flag test ==>"

  bazel build @foo//:bar >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log "In repo rule: " "Did not find repository rule print output"
  expect_not_log "$MARKER" \
      "Marker string '$MARKER' was seen even though \
      --internal_starlark_flag_test_canary wasn't passed"

  # Build with the special testing flag that appends a marker string to all
  # print() calls.
  bazel build @foo//:bar --internal_starlark_flag_test_canary >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log "In repo rule: $MARKER" \
      "Starlark flags are not propagating to repository rule implementation \
      function evaluation"
}

function test_starlark_repository_which_and_execute() {
  setup_starlark_repository

  echo "#!/bin/sh" > bin.sh
  echo "exit 0" >> bin.sh
  chmod +x bin.sh

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
  bash = repository_ctx.which("bash")
  if bash == None:
    fail("Bash not found!")
  bin = repository_ctx.which("bin.sh")
  if bin == None:
    fail("bin.sh not found!")
  result = repository_ctx.execute([bash, "--version"], 10, {"FOO": "BAR"})
  if result.return_code != 0:
    fail("Non-zero return code from bash: " + str(result.return_code))
  if result.stderr != "":
    fail("Non-empty error output: " + result.stderr)
  print(result.stdout)
repo = repository_rule(implementation=_impl, local=True)
EOF

  # Test we are using the client environment, not the server one
  bazel info &> /dev/null  # Start up the server.

  FOO="BAZ" PATH="${PATH}:${PWD}" bazel build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "version"
}

function test_starlark_repository_execute_stderr() {
  setup_starlark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
  result = repository_ctx.execute([str(repository_ctx.which("bash")), "-c", "echo erf >&2; exit 1"])
  if result.return_code != 1:
    fail("Incorrect return code from bash: %s != 1\n%s" % (result.return_code, result.stderr))
  if result.stdout != "":
    fail("Non-empty output: %s (stderr was %s)" % (result.stdout, result.stderr))
  print(result.stderr)
  repository_ctx.execute([str(repository_ctx.which("bash")), "-c", "echo shhhh >&2"], quiet = False)

repo = repository_rule(implementation=_impl, local=True)
EOF

  bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "erf"
  expect_log "shhhh"
}

function test_starlark_repository_execute_env_and_workdir() {
  setup_starlark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
  result = repository_ctx.execute(
    [str(repository_ctx.which("bash")), "-c", "echo PWD=\$PWD TOTO=\$TOTO"],
    1000000,
    { "TOTO": "titi" },
    working_directory = "$repo2")
  if result.return_code != 0:
    fail("Incorrect return code from bash: %s != 0\n%s" % (result.return_code, result.stderr))
  print(result.stdout)
repo = repository_rule(implementation=_impl, local=True)
EOF

  bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  if "$is_windows"; then
    repo2="$(cygpath $repo2)"
  fi
  expect_log "PWD=$repo2 TOTO=titi"
}

function test_starlark_repository_environ() {
  setup_starlark_repository

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  print(repository_ctx.os.environ["FOO"])
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
repo = repository_rule(implementation=_impl, local=False)
EOF

  # TODO(dmarting): We should seriously have something better to force a refetch...
  bazel clean --expunge
  FOO=BAR bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "BAR"

  FOO=BAR bazel clean --expunge >& $TEST_log
  FOO=BAR bazel info >& $TEST_log

  FOO=BAZ bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "BAZ"

  # Test that we don't re-run on server restart.
  FOO=BEZ bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_not_log "BEZ"
  bazel shutdown >& $TEST_log
  FOO=BEZ bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_not_log "BEZ"

  # Test that --action_env value is taken
  # TODO(dmarting): The current implemnentation cannot invalidate on environment
  # but the incoming change can declare environment dependency, once this is
  # done, maybe we should update this test to remove clean --expunge and use the
  # invalidation mechanism instead?
  bazel clean --expunge
  FOO=BAZ bazel build --action_env=FOO=BAZINGA @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "BAZINGA"

  bazel clean --expunge
  FOO=BAZ bazel build --action_env=FOO @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "BAZ"
  expect_not_log "BAZINGA"

  # Test modifying test.bzl invalidate the repository
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  print(repository_ctx.os.environ["BAR"])
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
repo = repository_rule(implementation=_impl, local=True)
EOF
  BAR=BEZ bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "BEZ"

  # Shutdown and modify again
  bazel shutdown
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  print(repository_ctx.os.environ["BAZ"])
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
repo = repository_rule(implementation=_impl, local=True)
EOF
  BAZ=BOZ bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "BOZ"
}

function write_environ_starlark() {
  local execution_file="$1"
  local environ="$2"

  cat >test.bzl <<EOF
load("//:environ.bzl", "environ")

def _impl(repository_ctx):
  # This might cause a function restart, do it first
  foo = environ(repository_ctx, "FOO")
  bar = environ(repository_ctx, "BAR")
  baz = environ(repository_ctx, "BAZ")
  repository_ctx.template("bar.txt", Label("//:bar.tpl"), {
      "%{FOO}": foo,
      "%{BAR}": bar,
      "%{BAZ}": baz}, False)

  exe_result = repository_ctx.execute(["cat", "${execution_file}"]);
  execution = int(exe_result.stdout.strip()) + 1
  repository_ctx.execute(["bash", "-c", "echo %s >${execution_file}" % execution])
  print("<%s> FOO=%s BAR=%s BAZ=%s" % (execution, foo, bar, baz))
  repository_ctx.file("BUILD", "filegroup(name='bar', srcs=['bar.txt'])")

repo = repository_rule(implementation=_impl, environ=[${environ}])
EOF
}

function setup_invalidation_test() {
  local startup_flag="${1-}"
  setup_starlark_repository

  # We use a counter to avoid other invalidation to hide repository
  # invalidation (e.g., --action_env will cause all action to re-run).
  local execution_file="${TEST_TMPDIR}/execution"

  # Our custom repository rule
  cat >environ.bzl <<EOF
def environ(r_ctx, var):
  return r_ctx.os.environ[var] if var in r_ctx.os.environ else "undefined"
EOF

  write_environ_starlark "${execution_file}" '"FOO", "BAR"'

  cat <<EOF >bar.tpl
FOO=%{FOO} BAR=%{BAR} BAZ=%{BAZ}
EOF

  bazel ${startup_flag} clean --expunge
  echo 0 >"${execution_file}"
  echo "${execution_file}"
}

# Test invalidation based on environment variable
function environ_invalidation_test_template() {
  local startup_flag="${1-}"
  local execution_file="$(setup_invalidation_test)"
  FOO=BAR bazel ${startup_flag} build @foo//:bar >& $TEST_log \
     || fail "Failed to build"
  expect_log "<1> FOO=BAR BAR=undefined BAZ=undefined"
  assert_equals 1 $(cat "${execution_file}")
  FOO=BAR bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 1 $(cat "${execution_file}")

  # Test that changing FOO is causing a refetch
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<2> FOO=BAZ BAR=undefined BAZ=undefined"
  assert_equals 2 $(cat "${execution_file}")
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 2 $(cat "${execution_file}")

  # Test that changing BAR is causing a refetch
  FOO=BAZ BAR=FOO bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<3> FOO=BAZ BAR=FOO BAZ=undefined"
  assert_equals 3 $(cat "${execution_file}")
  FOO=BAZ BAR=FOO bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 3 $(cat "${execution_file}")

  # Test that changing BAZ is not causing a refetch
  FOO=BAZ BAR=FOO BAZ=BAR bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 3 $(cat "${execution_file}")

  # Test more change in the environment
  FOO=BAZ BAR=FOO BEZ=BAR bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 3 $(cat "${execution_file}")

  # Test that removing BEZ is not causing a refetch
  FOO=BAZ BAR=FOO bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 3 $(cat "${execution_file}")

  # Test that removing BAR is causing a refetch
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<4> FOO=BAZ BAR=undefined BAZ=undefined"
  assert_equals 4 $(cat "${execution_file}")
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 4 $(cat "${execution_file}")

  # Now try to depends on more variables
  write_environ_starlark "${execution_file}" '"FOO", "BAR", "BAZ"'

  # The Starlark rule has changed, so a rebuild should happen
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<5> FOO=BAZ BAR=undefined BAZ=undefined"
  assert_equals 5 $(cat "${execution_file}")
  FOO=BAZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 5 $(cat "${execution_file}")

  # Now a change to BAZ should trigger a rebuild
  FOO=BAZ BAZ=BEZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<6> FOO=BAZ BAR=undefined BAZ=BEZ"
  assert_equals 6 $(cat "${execution_file}")
  FOO=BAZ BAZ=BEZ bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  assert_equals 6 $(cat "${execution_file}")
}

function environ_invalidation_action_env_test_template() {
  local startup_flag="${1-}"
  setup_starlark_repository

  # We use a counter to avoid other invalidation to hide repository
  # invalidation (e.g., --action_env will cause all action to re-run).
  local execution_file="$(setup_invalidation_test)"

  # Set to FOO=BAZ BAR=FOO
  FOO=BAZ BAR=FOO bazel ${startup_flag} build @foo//:bar >& $TEST_log \
      || fail "Failed to build"
  expect_log "<1> FOO=BAZ BAR=FOO BAZ=undefined"
  assert_equals 1 $(cat "${execution_file}")

  # Test with changing using --action_env
  bazel ${startup_flag} build \
      --action_env FOO=BAZ --action_env BAR=FOO  --action_env BEZ=BAR \
      @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 1 $(cat "${execution_file}")
  bazel ${startup_flag} build \
      --action_env FOO=BAZ --action_env BAR=FOO --action_env BAZ=BAR \
      @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 1 $(cat "${execution_file}")
  bazel ${startup_flag} build \
      --action_env FOO=BAR --action_env BAR=FOO --action_env BAZ=BAR \
      @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<2> FOO=BAR BAR=FOO BAZ=BAR"
  assert_equals 2 $(cat "${execution_file}")
}

function test_starlark_repository_environ_invalidation() {
  environ_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_starlark_repository_environ_invalidation_batch() {
  environ_invalidation_test_template --batch
}

function test_starlark_repository_environ_invalidation_action_env() {
  environ_invalidation_action_env_test_template
}

function test_starlark_repository_environ_invalidation_action_env_batch() {
  environ_invalidation_action_env_test_template --batch
}

# Test invalidation based on change to the bzl files
function bzl_invalidation_test_template() {
  local startup_flag="${1-}"
  local execution_file="$(setup_invalidation_test)"
  local flags="--action_env FOO=BAR --action_env BAR=BAZ --action_env BAZ=FOO"

  local bazel_build="bazel ${startup_flag} build ${flags}"

  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<1> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 1 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 1 $(cat "${execution_file}")

  # Changing the Starlark file cause a refetch
  cat <<EOF >>test.bzl

# Just add a comment
EOF
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<2> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 2 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 2 $(cat "${execution_file}")

  # But also changing the environ.bzl file does a refetch
  cat <<EOF >>environ.bzl

# Just add a comment
EOF
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<3> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 3 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 3 $(cat "${execution_file}")
}

function test_starlark_repository_bzl_invalidation() {
  bzl_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_starlark_repository_bzl_invalidation_batch() {
  bzl_invalidation_test_template --batch
}

# Test invalidation based on change to the bzl files
function file_invalidation_test_template() {
  local startup_flag="${1-}"
  local execution_file="$(setup_invalidation_test)"
  local flags="--action_env FOO=BAR --action_env BAR=BAZ --action_env BAZ=FOO"

  local bazel_build="bazel ${startup_flag} build ${flags}"

  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<1> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 1 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 1 $(cat "${execution_file}")

  # Changing the Starlark file cause a refetch
  cat <<EOF >>bar.tpl
Add more stuff
EOF
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<2> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 2 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 2 $(cat "${execution_file}")
}

function test_starlark_repository_file_invalidation() {
  file_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_starlark_repository_file_invalidation_batch() {
  file_invalidation_test_template --batch
}

# Test invalidation based on changes of the Starlark semantics
function starlark_invalidation_test_template() {
  local startup_flag="${1-}"
  local execution_file="$(setup_invalidation_test)"
  local flags="--action_env FOO=BAR --action_env BAR=BAZ --action_env BAZ=FOO"
  local bazel_build="bazel ${startup_flag} build ${flags}"

  ${bazel_build} --noincompatible_run_shell_command_string @foo//:bar \
    >& ${TEST_log} || fail "Expected success"
  expect_log "<1> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 1 $(cat "${execution_file}")

  echo; cat ${TEST_log}; echo

  ${bazel_build} --noincompatible_run_shell_command_string @foo//:bar \
    >& ${TEST_log} || fail "Expected success"
  assert_equals 1 $(cat "${execution_file}")

  echo; cat ${TEST_log}; echo

  # Changing the starlark semantics should invalidate once
  ${bazel_build} --incompatible_run_shell_command_string @foo//:bar \
    >& ${TEST_log} || fail "Expected success"
  expect_log "<2> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 2 $(cat "${execution_file}")
  ${bazel_build} --incompatible_run_shell_command_string @foo//:bar \
    >& ${TEST_log} || fail "Expected success"
  assert_equals 2 $(cat "${execution_file}")
}

function test_starlark_invalidation() {
    starlark_invalidation_test_template
}

function test_starlark_invalidation_batch() {
    starlark_invalidation_test_template --batch
}


function test_repo_env() {
  setup_starlark_repository

  cat > test.bzl <<'EOF'
def _impl(ctx):
  # Make a rule depending on the environment variable FOO,
  # properly recording its value. Also add a time stamp
  # to verify that the rule is rerun.
  ctx.execute(["bash", "-c", "echo FOO=$FOO > env.txt"])
  ctx.execute(["bash", "-c", "date +%s >> env.txt"])
  ctx.file("BUILD", 'exports_files(["env.txt"])')

repo = repository_rule(
  implementation = _impl,
  environ = ["FOO"],
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "repoenv",
  outs = ["repoenv.txt"],
  srcs = ["@foo//:env.txt"],
  cmd = "cp $< $@",
)

# Have a normal rule, unrelated to the external repository.
# To test if it was rerun, make it non-hermetic and record a
# time stamp.
genrule(
  name = "unrelated",
  outs = ["unrelated.txt"],
  cmd = "date +%s > $@",
)
EOF
  cat > .bazelrc <<EOF
common:foo --repo_env=FOO=foo
build:bar --repo_env=FOO=bar
common:qux --repo_env=FOO
EOF

  bazel build --config=foo //:repoenv //:unrelated
  cp `bazel info bazel-bin 2>/dev/null`/repoenv.txt repoenv1.txt
  cp `bazel info bazel-bin 2> /dev/null`/unrelated.txt unrelated1.txt
  echo; cat repoenv1.txt; echo; cat unrelated1.txt; echo

  grep -q 'FOO=foo' repoenv1.txt \
      || fail "Expected FOO to be visible to repo rules"

  sleep 2 # ensure any rerun will have a different time stamp

  FOO=CHANGED bazel build --config=foo //:repoenv //:unrelated
  # nothing should change, as actions don't see FOO and for repositories
  # the value is fixed by --repo_env
  cp `bazel info bazel-bin 2>/dev/null`/repoenv.txt repoenv2.txt
  cp `bazel info bazel-bin 2> /dev/null`/unrelated.txt unrelated2.txt
  echo; cat repoenv2.txt; echo; cat unrelated2.txt; echo

  diff repoenv1.txt repoenv2.txt \
      || fail "Expected repository to not change"
  diff unrelated1.txt unrelated2.txt \
      || fail "Expected unrelated action to not be rerun"

  bazel build --config=bar //:repoenv //:unrelated
  # The new config should be picked up, but the unrelated target should
  # not be rerun
  cp `bazel info bazel-bin 3>/dev/null`/repoenv.txt repoenv3.txt
  cp `bazel info bazel-bin 3> /dev/null`/unrelated.txt unrelated3.txt
  echo; cat repoenv3.txt; echo; cat unrelated3.txt; echo

  grep -q 'FOO=bar' repoenv3.txt \
      || fail "Expected FOO to be visible to repo rules"
  diff unrelated1.txt unrelated3.txt \
      || fail "Expected unrelated action to not be rerun"

  FOO=qux bazel build --config=qux //:repoenv //:unrelated
  # The new config should be picked up, but the unrelated target should
  # not be rerun
  cp `bazel info bazel-genfiles 3>/dev/null`/repoenv.txt repoenv4.txt
  cp `bazel info bazel-genfiles 3> /dev/null`/unrelated.txt unrelated4.txt
  echo; cat repoenv4.txt; echo; cat unrelated4.txt; echo

  grep -q 'FOO=qux' repoenv4.txt \
      || fail "Expected FOO to be visible to repo rules"
  diff unrelated1.txt unrelated4.txt \
      || fail "Expected unrelated action to not be rerun"
}

function test_repo_env_inverse() {
  # This test makes sure that a repository rule that has no dependencies on
  # environment variables does _not_ get refetched when --repo_env changes.
  setup_starlark_repository

  cat > test.bzl <<'EOF'
def _impl(ctx):
  # Record a time stamp to verify that the rule is not rerun.
  ctx.execute(["bash", "-c", "date +%s >> env.txt"])
  ctx.file("BUILD", 'exports_files(["env.txt"])')

repo = repository_rule(
  implementation = _impl,
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "repoenv",
  outs = ["repoenv.txt"],
  srcs = ["@foo//:env.txt"],
  cmd = "cp $< $@",
)
EOF
  cat > .bazelrc <<EOF
build:foo --repo_env=FOO=foo
build:bar --repo_env=FOO=bar
EOF

  bazel build --config=foo //:repoenv
  cp `bazel info bazel-bin 2>/dev/null`/repoenv.txt repoenv1.txt
  echo; cat repoenv1.txt; echo;

  sleep 2 # ensure any rerun will have a different time stamp

  bazel build --config=bar //:repoenv
  # The new config should not trigger a rerun of repoenv.
  cp `bazel info bazel-bin 2>/dev/null`/repoenv.txt repoenv2.txt
  echo; cat repoenv2.txt; echo;

  diff repoenv1.txt repoenv2.txt \
      || fail "Expected repository to not change"
}

function test_repo_env_invalidation() {
    # regression test for https://github.com/bazelbuild/bazel/issues/8869
    WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
    cd "${WRKDIR}"
    cat > WORKSPACE <<'EOF'
load("//:my_repository_rule.bzl", "my_repository_rule")

my_repository_rule(
    name = "my_repository_rule",
)
EOF
    cat > my_repository_rule.bzl <<'EOF'
def _my_repository_rule_impl(rctx):
    foo = rctx.os.environ.get("foo", default = "")

    print('foo = ' + foo)

    rctx.file("BUILD.bazel",
              "exports_files(['update_time'], visibility=['//visibility:public'])")
    rctx.execute(["bash", "-c", "date +%s > update_time"])

my_repository_rule = repository_rule(
    environ = ["foo"],
    implementation = _my_repository_rule_impl,
)
EOF
    cat > BUILD <<'EOF'
genrule(
  name = "repotime",
  outs = ["repotime.txt"],
  srcs = ["@my_repository_rule//:update_time"],
  cmd = "cp $< $@",
)
EOF

    bazel build //:repotime
    cp `bazel info bazel-bin 2>/dev/null`/repotime.txt time1.txt

    sleep 2;
    bazel build --repo_env=foo=bar //:repotime
    cp `bazel info bazel-bin 2>/dev/null`/repotime.txt time2.txt
    diff time1.txt time2.txt && fail "Expected repo to be refetched" || :

    bazel shutdown
    sleep 2;

    bazel build --repo_env=foo=bar //:repotime
    cp `bazel info bazel-bin 2>/dev/null`/repotime.txt time3.txt
    diff time2.txt time3.txt || fail "Expected repo to not be refetched"
}

function test_starlark_repository_executable_flag() {
  if "$is_windows"; then
    # There is no executable flag on Windows.
    echo "Skipping test_starlark_repository_executable_flag on Windows"
    return
  fi
  setup_starlark_repository

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.file("test.sh", "exit 0")
  repository_ctx.file("BUILD", "sh_binary(name='bar',srcs=['test.sh'])", False)
  repository_ctx.template("test2", Label("//:bar"), {}, False)
  repository_ctx.template("test2.sh", Label("//:bar"), {}, True)
repo = repository_rule(implementation=_impl, local=True)
EOF
  cat >bar

  bazel run @foo//:bar >& $TEST_log || fail "Execution of @foo//:bar failed"
  output_base=$(bazel info output_base)
  test -x "${output_base}/external/foo/test.sh" || fail "test.sh is not executable"
  test -x "${output_base}/external/foo/test2.sh" || fail "test2.sh is not executable"
  test ! -x "${output_base}/external/foo/BUILD" || fail "BUILD is executable"
  test ! -x "${output_base}/external/foo/test2" || fail "test2 is executable"
}

function test_starlark_repository_download() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local download_with_sha256="${server_dir}/download_with_sha256.txt"
  local download_executable_file="${server_dir}/download_executable_file.sh"
  echo "This is a file" > "${download_with_sha256}"
  echo "echo 'I am executable'" > "${download_executable_file}"
  file_sha256="$(sha256sum "${download_with_sha256}" | head -c 64)"
  file_exec_sha256="$(sha256sum "${download_executable_file}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  setup_starlark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download(
    "http://localhost:${fileserver_port}/download_with_sha256.txt",
    "download_with_sha256.txt", "${file_sha256}")
  repository_ctx.download(
    "http://localhost:${fileserver_port}/download_executable_file.sh",
    "download_executable_file.sh", executable=True, sha256="$file_exec_sha256")
  repository_ctx.file("BUILD")  # necessary directories should already created by download function
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  # Test download
  test -e "${output_base}/external/foo/download_with_sha256.txt" \
    || fail "download_with_sha256.txt is not downloaded"
  test -e "${output_base}/external/foo/download_executable_file.sh" \
    || fail "download_executable_file.sh is not downloaded"
  # Test download
  diff "${output_base}/external/foo/download_with_sha256.txt" \
    "${download_with_sha256}" >/dev/null \
    || fail "download_with_sha256.txt is not downloaded successfully"
  diff "${output_base}/external/foo/download_executable_file.sh" \
    "${download_executable_file}" >/dev/null \
    || fail "download_executable_file.sh is not downloaded successfully"

  # No executable flag for file on Windows
  if "$is_windows"; then
    return
  fi

  # Test executable
  test ! -x "${output_base}/external/foo/download_with_sha256.txt" \
    || fail "download_with_sha256.txt is executable"
  test -x "${output_base}/external/foo/download_executable_file.sh" \
    || fail "download_executable_file.sh is not executable"
}

function test_starlark_repository_context_downloads_return_struct() {
   # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local download_with_sha256="${server_dir}/download_with_sha256.txt"
  local download_no_sha256="${server_dir}/download_no_sha256.txt"
  local compressed_with_sha256="${server_dir}/compressed_with_sha256.txt"
  local compressed_no_sha256="${server_dir}/compressed_no_sha256.txt"
  echo "This is one file" > "${download_no_sha256}"
  echo "This is another file" > "${download_with_sha256}"
  echo "Compressed file with sha" > "${compressed_with_sha256}"
  echo "Compressed file no sha" > "${compressed_no_sha256}"
  zip "${compressed_with_sha256}".zip "${compressed_with_sha256}"
  zip "${compressed_no_sha256}".zip "${compressed_no_sha256}"

  provided_sha256="$(sha256sum "${download_with_sha256}" | head -c 64)"
  not_provided_sha256="$(sha256sum "${download_no_sha256}" | head -c 64)"
  compressed_provided_sha256="$(sha256sum "${compressed_with_sha256}.zip" | head -c 64)"
  compressed_not_provided_sha256="$(sha256sum "${compressed_no_sha256}.zip" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  # On Windows, a file url should be file:///C:/foo/bar,
  # we need to add one more slash at the beginning.
  if "$is_windows"; then
    server_dir="/${server_dir}"
  fi

  setup_starlark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  no_sha_return = repository_ctx.download(
    url = "file://${server_dir}/download_no_sha256.txt",
    output = "download_no_sha256.txt")
  with_sha_return = repository_ctx.download(
    url = "http://localhost:${fileserver_port}/download_with_sha256.txt",
    output = "download_with_sha256.txt",
    sha256 = "${provided_sha256}")
  compressed_no_sha_return = repository_ctx.download_and_extract(
    url = "file://${server_dir}/compressed_no_sha256.txt.zip",
    output = "compressed_no_sha256.txt.zip")
  compressed_with_sha_return = repository_ctx.download_and_extract(
      url = "http://localhost:${fileserver_port}/compressed_with_sha256.txt.zip",
      output = "compressed_with_sha256.txt.zip",
      sha256 = "${compressed_provided_sha256}")

  file_content = "no_sha_return " + no_sha_return.sha256 + "\n"
  file_content += "with_sha_return " + with_sha_return.sha256 + "\n"
  file_content += "compressed_no_sha_return " + compressed_no_sha_return.sha256 + "\n"
  file_content += "compressed_with_sha_return " + compressed_with_sha_return.sha256
  repository_ctx.file("returned_shas.txt", content = file_content, executable = False)
  repository_ctx.file("BUILD")  # necessary directories should already created by download function
repo = repository_rule(implementation = _impl, local = False)
EOF

  # This test case explicitly verifies that a checksum is returned, even if
  # none was provided by the call to download_and_extract. So we do have to
  # allow a download without provided checksum, even though it is plain http;
  # nevertheless, localhost is pretty safe against man-in-the-middle attacks.
  bazel build @foo//:all \
        >& $TEST_log && shutdown_server || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  grep "no_sha_return $not_provided_sha256" $output_base/external/foo/returned_shas.txt \
      || fail "expected calculated sha256 $not_provided_sha256"
  grep "with_sha_return $provided_sha256" $output_base/external/foo/returned_shas.txt \
      || fail "expected provided sha256 $provided_sha256"
  grep "compressed_with_sha_return $compressed_provided_sha256" $output_base/external/foo/returned_shas.txt \
      || fail "expected provided sha256 $compressed_provided_sha256"
  grep "compressed_no_sha_return $compressed_not_provided_sha256" $output_base/external/foo/returned_shas.txt \
      || fail "expected compressed calculated sha256 $compressed_not_provided_sha256"
}

function test_starlark_repository_download_args() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local download_1="${server_dir}/download_1.txt"
  local download_2="${server_dir}/download_2.txt"
  echo "The file's content" > "${download_1}"
  echo "The file's content" > "${download_2}"
  file_sha256="$(sha256sum "${download_1}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  create_new_workspace
  repo2=$new_workspace_dir

  cat > bar.txt
  echo "filegroup(name='bar', srcs=['bar.txt'])" > BUILD

  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load('//:test.bzl', 'repo')
repo(name = 'foo',
     urls = [
       "http://localhost:${fileserver_port}/download_1.txt",
       "http://localhost:${fileserver_port}/download_2.txt",
     ],
     sha256 = "${file_sha256}",
     output = "whatever.txt"
)
EOF

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.file("BUILD")
  repository_ctx.download(
    repository_ctx.attr.urls,
    sha256 = repository_ctx.attr.sha256,
    output=repository_ctx.attr.output,
  )

repo = repository_rule(implementation=_impl,
      local=False,
      attrs = {
        "urls" : attr.string_list(),
        "output" : attr.string(),
        "sha256" : attr.string(),
     }
)
EOF

  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  # Test download
  test -e "${output_base}/external/foo/whatever.txt" \
    || fail "whatever.txt is not downloaded"
}


function test_starlark_repository_download_and_extract() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local file_prefix="${server_dir}/download_and_extract"

  pushd ${TEST_TMPDIR}
  echo "This is one file" > server_dir/download_and_extract1.txt
  echo "This is another file" > server_dir/download_and_extract2.txt
  echo "This is a third file" > server_dir/download_and_extract3.txt
  tar -zcvf server_dir/download_and_extract1.tar.gz server_dir/download_and_extract1.txt
  zip server_dir/download_and_extract2.zip server_dir/download_and_extract2.txt
  zip server_dir/download_and_extract3.zip server_dir/download_and_extract3.txt
  file1_sha256="$(sha256sum server_dir/download_and_extract1.tar.gz | head -c 64)"
  file2_sha256="$(sha256sum server_dir/download_and_extract2.zip | head -c 64)"
  file3_sha256="$(sha256sum server_dir/download_and_extract3.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  setup_starlark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.file("BUILD")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract1.tar.gz", "", sha256="${file1_sha256}")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract2.zip", "", "${file2_sha256}")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract1.tar.gz", "some/path", sha256="${file1_sha256}")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract3.zip", ".", "${file3_sha256}", "", "")
  repository_ctx.download_and_extract(
    url = ["http://localhost:${fileserver_port}/download_and_extract3.zip"],
    output = "other/path",
    sha256 = "${file3_sha256}"
  )
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel clean --expunge_async >& $TEST_log || fail "bazel clean failed"
  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  # Test cleanup
  test -e "${output_base}/external/foo/server_dir/download_and_extract1.tar.gz" \
    && fail "temp file was not deleted successfully" || true
  test -e "${output_base}/external/foo/server_dir/download_and_extract2.zip" \
    && fail "temp file was not deleted successfully" || true
  test -e "${output_base}/external/foo/server_dir/download_and_extract3.zip" \
    && fail "temp file was not deleted successfully" || true
  # Test download_and_extract
  diff "${output_base}/external/foo/server_dir/download_and_extract1.txt" \
    "${file_prefix}1.txt" >/dev/null \
    || fail "download_and_extract1.tar.gz was not extracted successfully"
  diff "${output_base}/external/foo/some/path/server_dir/download_and_extract1.txt" \
    "${file_prefix}1.txt" >/dev/null \
    || fail "download_and_extract1.tar.gz was not extracted successfully in some/path"
  diff "${output_base}/external/foo/server_dir/download_and_extract2.txt" \
    "${file_prefix}2.txt" >/dev/null \
    || fail "download_and_extract2.zip was not extracted successfully"
  diff "${output_base}/external/foo/server_dir/download_and_extract3.txt" \
    "${file_prefix}3.txt" >/dev/null \
    || fail "download_and_extract3.zip was not extracted successfully"
  diff "${output_base}/external/foo/other/path/server_dir/download_and_extract3.txt" \
    "${file_prefix}3.txt" >/dev/null \
    || fail "download_and_extract3.tar.gz was not extracted successfully"
}

# Test native.bazel_version
function test_bazel_version() {
  create_new_workspace
  repo2=$new_workspace_dir

  cat > BUILD <<'EOF'
genrule(
    name = "test",
    cmd = "echo 'Tra-la!' | tee $@",
    outs = ["test.txt"],
    visibility = ["//visibility:public"],
)
EOF

  cd ${WORKSPACE_DIR}
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load('//:test.bzl', 'macro')

macro('$repo2')
EOF

  # Empty package for the .bzl file
  echo -n >BUILD

  # Our macro
  cat >test.bzl <<EOF
def macro(path):
  print(native.bazel_version)
  native.local_repository(name='test', path=path)
EOF

  local version="$(bazel info release)"
  # On release, Bazel binary get stamped, else we might run with an unstamped version.
  if [ "$version" == "development version" ]; then
    version=""
  else
    version="${version#* }"
  fi
  bazel build @test//:test >& $TEST_log || fail "Failed to build"
  expect_log ": ${version}."
}


# Test native.existing_rule(s), regression test for #1277
function test_existing_rule() {
  create_new_workspace
  setup_skylib_support
  repo2=$new_workspace_dir

  cat > BUILD

  cat >> WORKSPACE <<EOF
local_repository(name = 'existing', path='$repo2')
load('//:test.bzl', 'macro')

macro()
EOF

  # Empty package for the .bzl file
  echo -n >BUILD

  # Our macro
  cat >test.bzl <<EOF
def test(s):
  print("%s = %s,%s" % (s,
                        native.existing_rule(s) != None,
                        s in native.existing_rules()))
def macro():
  test("existing")
  test("non_existing")
EOF

  bazel query //... >& $TEST_log || fail "Failed to build"
  expect_log "existing = True,True"
  expect_log "non_existing = False,False"
}

function test_configure_like_repos() {
  cat > repos.bzl <<'EOF'
def _impl(ctx):
  print("Executing %s" % (ctx.attr.name,))
  ref = ctx.path(ctx.attr.reference)
  # Here we explicitly copy a file where we constructed the name
  # completely outside any build interfaces, so it is not registered
  # as a dependency of the external repository.
  ctx.execute(["cp", "%s.shadow" % (ref,), ctx.path("it.txt")])
  ctx.file("BUILD", "exports_files(['it.txt'])")

source = repository_rule(
 implementation = _impl,
 attrs = {"reference" : attr.label()},
)

configure = repository_rule(
 implementation = _impl,
 attrs = {"reference" : attr.label()},
 configure = True,
)

EOF
  cat > WORKSPACE <<'EOF'
load("//:repos.bzl", "configure", "source")

configure(name="configure", reference="@//:reference.txt")
source(name="source", reference="@//:reference.txt")
EOF
  cat > BUILD <<'EOF'
[ genrule(
    name = name,
    srcs = ["@%s//:it.txt" % (name,)],
    outs = ["%s.txt" % (name,)],
    cmd = "cp $< $@",
  ) for name in ["source", "configure"] ]
EOF
  echo "Just to get the path" > reference.txt
  echo "initial" > reference.txt.shadow

  bazel build //:source //:configure
  grep 'initial' `bazel info bazel-bin`/source.txt \
       || fail '//:source not generated properly'
  grep 'initial' `bazel info bazel-bin`/configure.txt \
       || fail '//:configure not generated properly'

  echo "new value" > reference.txt.shadow
  bazel sync --configure --experimental_repository_resolved_file=resolved.bzl \
        2>&1 || fail "Expected sync --configure to succeed"
  grep -q 'name.*configure' resolved.bzl \
      || fail "Expected 'configure' to be synced"
  grep -q 'name.*source' resolved.bzl \
      && fail "Expected 'source' not to be synced" || :

  bazel build //:source //:configure
  grep -q 'initial' `bazel info bazel-bin`/source.txt \
       || fail '//:source did not keep its old value'
  grep -q 'new value' `bazel info bazel-bin`/configure.txt \
       || fail '//:configure not synced properly'
}


function test_timeout_tunable() {
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<'EOF'
load("//:repo.bzl", "with_timeout")

with_timeout(name="maytimeout")
EOF
  touch BUILD
  cat > repo.bzl <<'EOF'
def _impl(ctx):
  st =ctx.execute(["bash", "-c", "sleep 10 && echo Hello world > data.txt"],
                  timeout=1)
  if st.return_code:
    fail("Command did not succeed")
  ctx.file("BUILD", "exports_files(['data.txt'])")

with_timeout = repository_rule(attrs = {}, implementation = _impl)
EOF
  bazel sync && fail "expected timeout" || :

  bazel sync --experimental_scale_timeouts=100 \
      || fail "expected success now the timeout is scaled"

  bazel build @maytimeout//... \
      || fail "expected success after successful sync"
}

function test_sync_only() {
  # Set up two repositories that count how often they are fetched
  cat >environ.bzl <<'EOF'
def environ(r_ctx, var):
  return r_ctx.os.environ[var] if var in r_ctx.os.environ else "undefined"
EOF
  cat <<'EOF' >bar.tpl
FOO=%{FOO} BAR=%{BAR} BAZ=%{BAZ}
EOF
  write_environ_starlark "${TEST_TMPDIR}/executionFOO" ""
  mv test.bzl testfoo.bzl
  write_environ_starlark "${TEST_TMPDIR}/executionBAR" ""
  mv test.bzl testbar.bzl
  cat > WORKSPACE <<'EOF'
load("//:testfoo.bzl", foorepo="repo")
load("//:testbar.bzl", barrepo="repo")
foorepo(name="foo")
barrepo(name="bar")
EOF
  touch BUILD
  bazel clean --expunge
  echo 0 > "${TEST_TMPDIR}/executionFOO"
  echo 0 > "${TEST_TMPDIR}/executionBAR"

  # Normal sync should hit both repositories
  echo; echo bazel sync; echo
  bazel sync
  assert_equals 1 $(cat "${TEST_TMPDIR}/executionFOO")
  assert_equals 1 $(cat "${TEST_TMPDIR}/executionBAR")

  # Only foo
  echo; echo bazel sync --only foo; echo
  bazel sync --only foo
  assert_equals 2 $(cat "${TEST_TMPDIR}/executionFOO")
  assert_equals 1 $(cat "${TEST_TMPDIR}/executionBAR")

  # Only bar
  echo; echo bazel sync --only bar; echo
  bazel sync --only bar
  assert_equals 2 $(cat "${TEST_TMPDIR}/executionFOO")
  assert_equals 2 $(cat "${TEST_TMPDIR}/executionBAR")

  # Only bar
  echo; echo bazel sync --only bar; echo
  bazel sync --only bar
  assert_equals 2 $(cat "${TEST_TMPDIR}/executionFOO")
  assert_equals 3 $(cat "${TEST_TMPDIR}/executionBAR")
}

function test_download_failure_message() {
  # Regression test for #7850
  # Verify that the for a failed download, it is clearly indicated
  # what was attempted to download and how it fails.
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  outs = ["it.txt"],
  srcs = ["@ext_foo//:data.txt"],
  cmd = "cp $< $@",
)
EOF
  cat > repo.bzl <<'EOF'
def _impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  ctx.symlink(ctx.attr.data, "data.txt")

trivial_repo = repository_rule(
  implementation = _impl,
  attrs = { "data" : attr.label() },
)
EOF
  cat > root.bzl <<'EOF'
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def root_cause():
  http_archive(
    name = "this_is_the_root_cause",
    urls = ["http://does.not.exist.example.com/some/file.tar"],
    sha256 = "aba1fcb7781eb26c854d13446a4b3e8a906cc03676371bbb69eb4430926f5969",
  )

EOF
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<'EOF'
load("//:root.bzl", "root_cause")
load("//:repo.bzl", "trivial_repo")

root_cause()

trivial_repo(
  name = "ext_baz",
  data = "@this_is_the_root_cause//:data.txt",
)
trivial_repo(
  name = "ext_bar",
  data = "@ext_baz//:data.txt",
)

trivial_repo(
  name = "ext_foo",
  data = "@ext_bar//:data.txt",
)
EOF

  bazel build //:it > "${TEST_log}" 2>&1 \
      && fail "Expected failure" || :

  # Extract the first error message printed
  #
  # ERROR: An error occurred during the fetch of repository 'this_is_the_root_cause':
  #    Traceback (most recent call last):
  # 	File ".../http.bzl", line 111, column 45, in _http_archive_impl
  # 		download_info = ctx.download_and_extract(
  # Error in download_and_extract: java.io.IOException: Error downloading \
  #   [http://does.not.exist.example.com/some/file.tar] to ...file.tar: \
  #   Unknown host: does.not.exist.example.com
  awk '/^ERROR/ {on=1} on {print} /^Error/ {exit}' < "${TEST_log}" > firsterror.log
  echo; echo "first error message which should focus on the root cause";
  echo "=========="; cat firsterror.log; echo "=========="
  # We expect it to contain the root cause, and the failure ...
  grep -q 'this_is_the_root_cause' firsterror.log \
      || fail "Root-cause repository not mentioned"
  grep -q '[uU]nknown host.*does.not.exist.example.com' firsterror.log \
      || fail "Failure reason not mentioned"
  # ...but not be cluttered with information not related to the root cause
  grep 'ext_foo' firsterror.log && fail "unrelated repo mentioned" || :
  grep 'ext_bar' firsterror.log && fail "unrelated repo mentioned" || :
  grep 'ext_baz' firsterror.log && fail "unrelated repo mentioned" || :
  grep '//:it' firsterror.log && fail "unrelated target mentioned" || :

  # Verify that the same is true, if the error is caused by a fail statement.
  cat > root.bzl <<'EOF'
def _impl(ctx):
  fail("Here be dragons")

repo = repository_rule(implementation=_impl, attrs = {})

def root_cause():
  repo(name = "this_is_the_root_cause")
EOF
  bazel build //:it > "${TEST_log}" 2>&1 \
      && fail "Expected failure" || :

  # Extract the first error message printed (see previous awk command).
  awk '/^ERROR/ {on=1} on {print} /^Error/ {exit}' < "${TEST_log}" > firsterror.log
  echo "=========="; cat firsterror.log; echo "=========="
  grep -q 'this_is_the_root_cause' firsterror.log \
      || fail "Root-cause repository not mentioned"
  grep -q 'Here be dragons' firsterror.log \
      || fail "fail error message not shown"
  grep 'ext_foo' firsterror.log && fail "unrelated repo mentioned" || :
  grep 'ext_bar' firsterror.log && fail "unrelated repo mentioned" || :
  grep 'ext_baz' firsterror.log && fail "unrelated repo mentioned" || :
  grep '//:it' firsterror.log && fail "unrelated target mentioned" || :
}

function test_circular_load_error_message() {
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<'EOF'
load("//:a.bzl", "total")
EOF
  touch BUILD
  cat > a.bzl <<'EOF'
load("//:b.bzl", "b")

a = 10

total = a + b
EOF
  cat > b.bzl <<'EOF'
load("//:a.bzl", "a")

b = 20

difference = b - a
EOF

  bazel build //... > "${TEST_log}" 2>&1 && fail "Expected failure" || :

  expect_not_log "[iI]nternal [eE]rror"
  expect_not_log "IllegalStateException"
  expect_log "//:a.bzl"
  expect_log "//:b.bzl"
}

function test_ciruclar_load_error_with_path_message() {
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<'EOF'
load("//:x.bzl", "x")
EOF
  touch BUILD
  cat > x.bzl <<'EOF'
load("//:y.bzl", "y")
x = y
EOF
  cat > y.bzl <<'EOF'
load("//:a.bzl", "total")

y = total
EOF
  cat > a.bzl <<'EOF'
load("//:b.bzl", "b")

a = 10

total = a + b
EOF
  cat > b.bzl <<'EOF'
load("//:a.bzl", "a")

b = 20

difference = b - a
EOF

  bazel build //... > "${TEST_log}" 2>&1 && fail "Expected failure" || :

  expect_not_log "[iI]nternal [eE]rror"
  expect_not_log "IllegalStateException"
  expect_log "WORKSPACE"
  expect_log "//:x.bzl"
  expect_log "//:y.bzl"
  expect_log "//:a.bzl"
  expect_log "//:b.bzl"
}

# Common setup logic for test_auth_*.
# Receives as arguments the username and password for basic authentication.
# If no arguments are provided, no authentication is used.
function setup_auth() {
  if [[ $# -ne 0 && $# -ne 2 ]]; then
    fail "setup_auth expects exactly zero or two arguments"
  fi

  mkdir -p x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256="$(sha256sum x.tar | head -c 64)"
  serve_file_auth x.tar

  if [[ $# -eq 2 ]]; then
    local -r username=$1
    local -r password=$2
    cat > auth_attrs.bzl <<EOF
AUTH_ATTRS = {
  "type": "basic",
  "login" : "$username",
  "password" : "$password",
}
EOF
  else
    cat > auth_attrs.bzl <<'EOF'
AUTH_ATTRS = None
EOF
  fi

  cat > auth.bzl <<'EOF'
load("//:auth_attrs.bzl", "AUTH_ATTRS")
def _impl(ctx):
  ctx.download_and_extract(
    url = ctx.attr.url,
    sha256 = ctx.attr.sha256,
    auth = { ctx.attr.url: AUTH_ATTRS } if AUTH_ATTRS else {},
  )

maybe_with_auth = repository_rule(
  implementation = _impl,
  attrs = {
    "url" : attr.string(),
    "sha256" : attr.string(),
  }
)
EOF

  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load("//:auth.bzl", "maybe_with_auth")
maybe_with_auth(
  name = "ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  sha256 = "$sha256",
)
EOF

  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
}

function test_auth_from_starlark() {
  setup_auth foo bar

  bazel build //:it \
      || fail "Expected success when downloading repo with basic auth"
}

function test_auth_from_credential_helper() {
  if "$is_windows"; then
    # Skip on Windows: credential helper is a Python script.
    return
  fi

  setup_credential_helper

  setup_auth # no auth

  bazel build //:it \
      && fail "Expected failure when downloading repo without credential helper"

  bazel build --credential_helper="${TEST_TMPDIR}/credhelper" //:it \
      || fail "Expected success when downloading repo with credential helper"

  expect_credential_helper_calls 1

  bazel build --credential_helper="${TEST_TMPDIR}/credhelper" //:it \
      || fail "Expected success when downloading repo with credential helper"

  expect_credential_helper_calls 1 # expect credentials to have been cached
}

function test_auth_from_credential_helper_overrides_starlark() {
  if "$is_windows"; then
    # Skip on Windows: credential helper is a Python script.
    return
  fi

  setup_credential_helper

  setup_auth baduser badpass

  bazel build --credential_helper="${TEST_TMPDIR}/credhelper" //:it \
      || fail "Expected success when downloading repo with credential helper overriding basic auth"
}

function test_netrc_reading() {
  # Write a badly formatted, but correct, .netrc file
  cat > .netrc <<'EOF'
machine ftp.example.com
macdef init
cd pub
mget *
quit

# this is a comment
machine example.com login
myusername password mysecret default
login anonymous password myusername@example.com
EOF
  # We expect that `read_netrc` can parse this file...
  cat > def.bzl <<'EOF'
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "read_netrc")
def _impl(ctx):
  rc = read_netrc(ctx, ctx.attr.path)
  ctx.file("data.bzl", "netrc = %s" % (rc,))
  ctx.file("BUILD", "")
  ctx.file("WORKSPACE", "")

netrcrepo = repository_rule(
  implementation = _impl,
  attrs = {"path": attr.string()},
)
EOF

  netrc_dir="$(pwd)"
  if "$is_windows"; then
    netrc_dir="$(cygpath -m ${netrc_dir})"
  fi

  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load("//:def.bzl", "netrcrepo")

netrcrepo(name = "netrc", path="${netrc_dir}/.netrc")
EOF
  # ...and that from the parse result, we can read off the
  # credentials for example.com.
  cat > BUILD <<'EOF'
load("@netrc//:data.bzl", "netrc")

[genrule(
  name = name,
  outs = [ "%s.txt" % (name,)],
  cmd = "echo %s > $@" % (netrc["example.com"][name],),
) for name in ["login", "password"]]
EOF
  bazel build //:login //:password
  grep 'myusername' `bazel info bazel-bin`/login.txt \
       || fail "Username not parsed correctly"
  grep 'mysecret' `bazel info bazel-bin`/password.txt \
       || fail "Password not parsed correctly"

  # Also check the precise value of parsed file
  cat > expected.bzl <<'EOF'
expected = {
  "ftp.example.com" : { "macdef init" : "cd pub\nmget *\nquit\n" },
  "example.com" : { "login" : "myusername",
                    "password" : "mysecret",
                  },
  "" : { "login": "anonymous",
          "password" : "myusername@example.com" },
}
EOF
  cat > verify.bzl <<'EOF'
load("@netrc//:data.bzl", "netrc")
load("//:expected.bzl", "expected")

def check_equal_expected():
  print("Parsed value:   %s" % (netrc,))
  print("Expected value: %s" % (expected,))
  if netrc == expected:
    return "OK"
  else:
    return "BAD"
EOF
  cat > BUILD <<'EOF'
load ("//:verify.bzl", "check_equal_expected")
genrule(
  name = "check_expected",
  outs = ["check_expected.txt"],
  cmd = "echo %s > $@" % (check_equal_expected(),)
)
EOF
  bazel build //:check_expected
  grep 'OK' `bazel info bazel-bin`/check_expected.txt \
       || fail "Parsed dict not equal to expected value"
}

function test_use_netrc() {
    # Test the starlark utility function use_netrc.
  cat > .netrc <<'EOF'
machine foo.example.org
login foousername
password foopass

machine bar.example.org
login barusername
password passbar

# following lines mix tabs and spaces
machine	  oauthlife.com
	password	TOKEN
EOF
  # Read a given .netrc file and combine it with a list of URL,
  # and write the obtained authentication dictionary to disk; this
  # is not the intended way of using, but makes testing easy.
  cat > def.bzl <<'EOF'
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "read_netrc", "use_netrc")
def _impl(ctx):
  rc = read_netrc(ctx, ctx.attr.path)
  auth = use_netrc(rc, ctx.attr.urls, {"oauthlife.com": "Bearer <password>",})
  ctx.file("data.bzl", "auth = %s" % (auth,))
  ctx.file("BUILD", "")
  ctx.file("WORKSPACE", "")

authrepo = repository_rule(
  implementation = _impl,
  attrs = {"path": attr.string(),
           "urls": attr.string_list()
  },
)
EOF

  netrc_dir="$(pwd)"
  if "$is_windows"; then
    netrc_dir="$(cygpath -m ${netrc_dir})"
  fi

  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
load("//:def.bzl", "authrepo")

authrepo(
  name = "auth",
  path="${netrc_dir}/.netrc",
  urls = [
    "http://example.org/public/null.tar",
    "https://foo.example.org/file1.tar",
    "https://foo.example.org:8080/file2.tar",
    "https://bar.example.org/file3.tar",
    "https://evil.com/bar.example.org/file4.tar",
    "https://oauthlife.com/fizz/buzz/file5.tar",
  ],
)
EOF
  # Here dicts give us the correct notion of equality, so we can simply
  # compare against the expected value.
  cat > expected.bzl <<'EOF'
expected = {
    "https://foo.example.org/file1.tar" : {
      "type" : "basic",
      "login": "foousername",
      "password" : "foopass",
    },
    "https://foo.example.org:8080/file2.tar" : {
      "type" : "basic",
      "login": "foousername",
      "password" : "foopass",
    },
    "https://bar.example.org/file3.tar" : {
      "type" : "basic",
      "login": "barusername",
      "password" : "passbar",
    },
    "https://oauthlife.com/fizz/buzz/file5.tar": {
      "type" : "pattern",
      "pattern" : "Bearer <password>",
      "password" : "TOKEN",
    },
}
EOF
  cat > verify.bzl <<'EOF'
load("@auth//:data.bzl", "auth")
load("//:expected.bzl", "expected")

def check_equal_expected():
  print("Computed value: %s" % (auth,))
  print("Expected value: %s" % (expected,))
  if auth == expected:
    return "OK"
  else:
    return "BAD"
EOF
  cat > BUILD <<'EOF'
load ("//:verify.bzl", "check_equal_expected")
genrule(
  name = "check_expected",
  outs = ["check_expected.txt"],
  cmd = "echo %s > $@" % (check_equal_expected(),)
)
EOF
  bazel build //:check_expected
  grep 'OK' `bazel info bazel-bin`/check_expected.txt \
       || fail "Authentication merged incorrectly"
}

function test_disallow_unverified_http() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256="$(sha256sum x.tar | head -c 64)"
  serve_file x.tar
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:it > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log 'plain http.*missing checksum'

  # After adding a good checksum, we expect success
  ed WORKSPACE <<EOF
/url
a
sha256 = "$sha256",
.
w
q
EOF
  bazel build //:it || fail "Expected success one the checksum is given"

}

function tear_down() {
  shutdown_server
  if [ -d "${TEST_TMPDIR}/server_dir" ]; then
    rm -fr "${TEST_TMPDIR}/server_dir"
  fi
  true
}

function test_http_archive_netrc() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file_auth x.tar
  netrc_dir="$(pwd)"
  if "$is_windows"; then
    netrc_dir="$(cygpath -m ${netrc_dir})"
  fi
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  netrc = "${netrc_dir}/.netrc",
  sha256="$sha256",
)
EOF
  cat > .netrc <<'EOF'
machine 127.0.0.1
login foo
password bar
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:it \
      || fail "Expected success despite needing a file behind basic auth"
}

function test_http_archive_auth_patterns() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file_auth x.tar
  netrc_dir="$(pwd)"
  if "$is_windows"; then
    netrc_dir="$(cygpath -m ${netrc_dir})"
  fi
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  netrc = "${netrc_dir}/.netrc",
  sha256="$sha256",
  auth_patterns = {
    "127.0.0.1": "Bearer <password>"
  }
)
EOF
  cat > .netrc <<'EOF'
machine 127.0.0.1
password TOKEN
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:it \
      || fail "Expected success despite needing a file behind bearer auth"
}

function test_http_archive_implicit_netrc() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file_auth x.tar

  export HOME=`pwd`
  if "$is_windows"; then
    export USERPROFILE="$(cygpath -m ${HOME})"
  fi
  cat > .netrc <<'EOF'
machine 127.0.0.1
login foo
password bar
EOF

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  sha256="$sha256",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:it \
      || fail "Expected success despite needing a file behind basic auth"
}

function test_http_archive_credential_helper() {
  if "$is_windows"; then
    # Skip on Windows: credential helper is a Python script.
    return
  fi

  setup_credential_helper

  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file_auth x.tar
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  sha256="$sha256",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build --credential_helper="${TEST_TMPDIR}/credhelper" //:it \
      || fail "Expected success despite needing a file behind credential helper"
}

function test_http_archive_credential_helper_overrides_netrc() {
  if "$is_windows"; then
    # Skip on Windows: credential helper is a Python script.
    return
  fi

  setup_credential_helper

  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file_auth x.tar

  export HOME=`pwd`
  if "$is_windows"; then
    export USERPROFILE="$(cygpath -m ${HOME})"
  fi
  cat > .netrc <<'EOF'
machine 127.0.0.1
login badusername
password badpassword
EOF

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  sha256="$sha256",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build --credential_helper="${TEST_TMPDIR}/credhelper" //:it \
      || fail "Expected success despite needing a file behind credential helper"
}

function test_disable_download_should_prevent_downloading() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)
  serve_file x.tar

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1:$nc_port/x.tar",
  sha256="$sha256",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build --repository_disable_download //:it > "${TEST_log}" 2>&1 \
      && fail "Expected failure" || :
  expect_log "Failed to download repository @ext: download is disabled"
}

function test_disable_download_should_allow_distdir() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  tar cvf x.tar x
  sha256=$(sha256sum x.tar | head -c 64)

  mkdir main
  cp x.tar main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  url = "http://127.0.0.1/x.tar",
  sha256="$sha256",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//x:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build --distdir="." --repository_disable_download //:it || fail "Failed to build"
}

function test_disable_download_should_allow_local_repository() {
  mkdir x
  echo 'exports_files(["file.txt"])' > x/BUILD
  echo 'Hello World' > x/file.txt
  touch x/WORKSPACE

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
local_repository(
  name="ext",
  path="../x",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//:file.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel build --repository_disable_download //:it || fail "Failed to build"
}

function test_no_restarts_fetching_with_worker_thread() {
  setup_starlark_repository

  echo foo > file1
  echo bar > file2

  cat >test.bzl <<EOF
def _impl(rctx):
  print("hello world!")
  print(rctx.read(Label("//:file1")))
  print(rctx.read(Label("//:file2")))
  rctx.file("BUILD", "filegroup(name='bar')")

repo = repository_rule(implementation=_impl, local=True)
EOF

  # no worker thread, restarts twice
  bazel build @foo//:bar --experimental_worker_for_repo_fetching=off >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log_n "hello world!" 3

  # platform worker thread, never restarts
  bazel shutdown
  bazel build @foo//:bar --experimental_worker_for_repo_fetching=platform >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log_n "hello world!" 1
}

run_suite "local repository tests"
