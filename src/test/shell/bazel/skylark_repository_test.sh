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
  cat > WORKSPACE <<EOF
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
  cat bazel-genfiles/zoo/ball-pit1.txt >$TEST_log
  expect_log "Tra-la!"

  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la!"  # No invalidation

  bazel build //zoo:ball-pit2 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la!"  # No invalidation
  cat bazel-genfiles/zoo/ball-pit2.txt >$TEST_log
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
  cat bazel-genfiles/zoo/ball-pit1.txt >$TEST_log
  expect_log "Tra-la-la!"

  bazel build //zoo:ball-pit1 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la-la!"  # No invalidation

  bazel build //zoo:ball-pit2 >& $TEST_log || fail "Failed to build"
  expect_not_log "Tra-la-la!"  # No invalidation
  cat bazel-genfiles/zoo/ball-pit2.txt >$TEST_log
  expect_log "Tra-la-la!"
}

function test_load_from_symlink_to_outside_of_workspace() {
  OTHER=$TEST_TMPDIR/other

  cat > WORKSPACE<<EOF
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
  cat > WORKSPACE <<EOF
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
  cat > WORKSPACE <<EOF
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
  cat > WORKSPACE <<EOF
load("@foo//:ext.bzl", "macro")
macro()
local_repository(name='foo', path='$repo2')
EOF

  local exitCode=0
  bazel build @foo//:data.txt >& $TEST_log || exitCode=$?
  [ $exitCode != 0 ] || fail "building @foo//:data.txt succeed while expected failure"

  expect_not_log "PACKAGE"
  expect_log "Failed to load Starlark extension '@foo//:ext.bzl'"
  expect_log "repository 'foo' was defined too late in your WORKSPACE file"
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
  expect_log "repository 'does_not_exist' was defined too late in your WORKSPACE file"

  # Retest with query //...
  bazel clean --expunge
  bazel query //... >& $TEST_log || exitCode=$?
  [ $exitCode != 0 ] || fail "querying //... succeed while expected failure"

  expect_not_log "PACKAGE"
  expect_log "Failed to load Starlark extension '@does_not_exist//:random.bzl'"
  expect_log "repository 'does_not_exist' was defined too late in your WORKSPACE file"
}

function test_skylark_local_repository() {
  create_new_workspace
  repo2=$new_workspace_dir
  # Remove the WORKSPACE file in the symlinked repo, so our skylark rule has to
  # create one.
  rm $repo2/WORKSPACE

  cat > BUILD <<'EOF'
genrule(name='bar', cmd='echo foo | tee $@', outs=['bar.txt'])
EOF

  cd ${WORKSPACE_DIR}
  cat > WORKSPACE <<EOF
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
  cat bazel-genfiles/external/foo/bar.txt >$TEST_log
  expect_log "foo"
}

function setup_skylark_repository() {
  create_new_workspace
  repo2=$new_workspace_dir

  cat > bar.txt
  echo "filegroup(name='bar', srcs=['bar.txt'])" > BUILD

  cd "${WORKSPACE_DIR}"
  cat > WORKSPACE <<EOF
load('//:test.bzl', 'repo')
repo(name = 'foo')
EOF
  # Need to be in a package
  cat > BUILD
}

function test_skylark_flags_affect_repository_rule() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  print("In repo rule: ")
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))

repo = repository_rule(implementation=_impl, local=True)
EOF

  MARKER="<== skylark flag test ==>"

  bazel build @foo//:bar >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log "In repo rule: " "Did not find repository rule print output"
  expect_not_log "$MARKER" \
      "Marker string '$MARKER' was seen even though \
      --internal_skylark_flag_test_canary wasn't passed"

  # Build with the special testing flag that appends a marker string to all
  # print() calls.
  bazel build @foo//:bar --internal_skylark_flag_test_canary >& $TEST_log \
    || fail "Expected build to succeed"
  expect_log "In repo rule: $MARKER" \
      "Starlark flags are not propagating to repository rule implementation \
      function evaluation"
}

function test_skylark_repository_which_and_execute() {
  setup_skylark_repository

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

function test_skylark_repository_execute_stderr() {
  setup_skylark_repository

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

function test_skylark_repository_execute_env_and_workdir() {
  setup_skylark_repository

  cat >test.bzl <<EOF
def _impl(repository_ctx):
  # Symlink so a repository is created
  repository_ctx.symlink(repository_ctx.path("$repo2"), repository_ctx.path(""))
  result = repository_ctx.execute(
    [str(repository_ctx.which("bash")), "-c", "echo PWD=\$PWD TOTO=\$TOTO"],
    1000000,
    { "TOTO": "titi" })
  if result.return_code != 0:
    fail("Incorrect return code from bash: %s != 0\n%s" % (result.return_code, result.stderr))
  print(result.stdout)
repo = repository_rule(implementation=_impl, local=True)
EOF

  bazel build @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "PWD=$repo2 TOTO=titi"
}

function test_skylark_repository_environ() {
  setup_skylark_repository

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

function write_environ_skylark() {
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
  setup_skylark_repository

  # We use a counter to avoid other invalidation to hide repository
  # invalidation (e.g., --action_env will cause all action to re-run).
  local execution_file="${TEST_TMPDIR}/execution"

  # Our custom repository rule
  cat >environ.bzl <<EOF
def environ(r_ctx, var):
  return r_ctx.os.environ[var] if var in r_ctx.os.environ else "undefined"
EOF

  write_environ_skylark "${execution_file}" '"FOO", "BAR"'

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
  write_environ_skylark "${execution_file}" '"FOO", "BAR", "BAZ"'

  # The skylark rule has changed, so a rebuild should happen
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
  setup_skylark_repository

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

function test_skylark_repository_environ_invalidation() {
  environ_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_skylark_repository_environ_invalidation_batch() {
  environ_invalidation_test_template --batch
}

function test_skylark_repository_environ_invalidation_action_env() {
  environ_invalidation_action_env_test_template
}

function test_skylark_repository_environ_invalidation_action_env_batch() {
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

  # Changing the skylark file cause a refetch
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

function test_skylark_repository_bzl_invalidation() {
  bzl_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_skylark_repository_bzl_invalidation_batch() {
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

  # Changing the skylark file cause a refetch
  cat <<EOF >>bar.tpl
Add more stuff
EOF
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  expect_log "<2> FOO=BAR BAR=BAZ BAZ=FOO"
  assert_equals 2 $(cat "${execution_file}")
  ${bazel_build} @foo//:bar >& $TEST_log || fail "Failed to build"
  assert_equals 2 $(cat "${execution_file}")
}

function test_skylark_repository_file_invalidation() {
  file_invalidation_test_template
}

# Same test as previous but with server restart between each invocation
function test_skylark_repository_file_invalidation_batch() {
  file_invalidation_test_template --batch
}

function test_skylark_repository_executable_flag() {
  setup_skylark_repository

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

function test_skylark_repository_download() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local download_with_sha256="${server_dir}/download_with_sha256.txt"
  local download_no_sha256="${server_dir}/download_no_sha256.txt"
  local download_executable_file="${server_dir}/download_executable_file.sh"
  echo "This is one file" > "${download_no_sha256}"
  echo "This is another file" > "${download_with_sha256}"
  echo "echo 'I am executable'" > "${download_executable_file}"
  file_sha256="$(sha256sum "${download_with_sha256}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  setup_skylark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.download(
    "http://localhost:${fileserver_port}/download_no_sha256.txt",
    "download_no_sha256.txt")
  repository_ctx.download(
    "http://localhost:${fileserver_port}/download_with_sha256.txt",
    "download_with_sha256.txt", "${file_sha256}")
  repository_ctx.download(
    "http://localhost:${fileserver_port}/download_executable_file.sh",
    "download_executable_file.sh", executable=True)
  repository_ctx.file("BUILD")  # necessary directories should already created by download function
repo = repository_rule(implementation=_impl, local=False)
EOF

  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  # Test download
  test -e "${output_base}/external/foo/download_no_sha256.txt" \
    || fail "download_no_sha256.txt is not downloaded"
  test -e "${output_base}/external/foo/download_with_sha256.txt" \
    || fail "download_with_sha256.txt is not downloaded"
  test -e "${output_base}/external/foo/download_executable_file.sh" \
    || fail "download_executable_file.sh is not downloaded"
  # Test download
  diff "${output_base}/external/foo/download_no_sha256.txt" \
    "${download_no_sha256}" >/dev/null \
    || fail "download_no_sha256.txt is not downloaded successfully"
  diff "${output_base}/external/foo/download_with_sha256.txt" \
    "${download_with_sha256}" >/dev/null \
    || fail "download_with_sha256.txt is not downloaded successfully"
  diff "${output_base}/external/foo/download_executable_file.sh" \
    "${download_executable_file}" >/dev/null \
    || fail "download_executable_file.sh is not downloaded successfully"
  # Test executable
  test ! -x "${output_base}/external/foo/download_no_sha256.txt" \
    || fail "download_no_sha256.txt is executable"
  test ! -x "${output_base}/external/foo/download_with_sha256.txt" \
    || fail "download_with_sha256.txt is executable"
  test -x "${output_base}/external/foo/download_executable_file.sh" \
    || fail "download_executable_file.sh is not executable"
}

function test_skylark_repository_context_downloads_return_struct() {
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

  setup_skylark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  no_sha_return = repository_ctx.download(
    url = "http://localhost:${fileserver_port}/download_no_sha256.txt",
    output = "download_no_sha256.txt")
  with_sha_return = repository_ctx.download(
    url = "http://localhost:${fileserver_port}/download_with_sha256.txt",
    output = "download_with_sha256.txt",
    sha256 = "${provided_sha256}")
  compressed_no_sha_return = repository_ctx.download_and_extract(
    url = "http://localhost:${fileserver_port}/compressed_no_sha256.txt.zip",
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

  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

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

function test_skylark_repository_download_args() {
  # Prepare HTTP server with Python
  local server_dir="${TEST_TMPDIR}/server_dir"
  mkdir -p "${server_dir}"
  local download_with_sha256="${server_dir}/download_with_sha256.txt"
  local download_no_sha256="${server_dir}/download_no_sha256.txt"
  local download_executable_file="${server_dir}/download_executable_file.sh"
  echo "This is one file" > "${download_no_sha256}"
  echo "This is another file" > "${download_with_sha256}"
  echo "echo 'I am executable'" > "${download_executable_file}"
  file_sha256="$(sha256sum "${download_with_sha256}" | head -c 64)"

  # Start HTTP server with Python
  startup_server "${server_dir}"

  create_new_workspace
  repo2=$new_workspace_dir

  cat > bar.txt
  echo "filegroup(name='bar', srcs=['bar.txt'])" > BUILD

  cat > WORKSPACE <<EOF
load('//:test.bzl', 'repo')
repo(name = 'foo',
     urls = [
       "http://localhost:${fileserver_port}/download_no_sha256.txt",
       "http://localhost:${fileserver_port}/download_with_sha256.txt",
     ],
     output = "whatever.txt"
)
EOF

  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.file("BUILD")
  repository_ctx.download(repository_ctx.attr.urls, output=repository_ctx.attr.output)

repo = repository_rule(implementation=_impl,
      local=False,
      attrs = { "urls" : attr.string_list(), "output" : attr.string() }
)
EOF

  bazel build @foo//:all >& $TEST_log && shutdown_server \
    || fail "Execution of @foo//:all failed"

  output_base="$(bazel info output_base)"
  # Test download
  test -e "${output_base}/external/foo/whatever.txt" \
    || fail "whatever.txt is not downloaded"
}


function test_skylark_repository_download_and_extract() {
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
  file_sha256="$(sha256sum server_dir/download_and_extract3.zip | head -c 64)"
  popd

  # Start HTTP server with Python
  startup_server "${server_dir}"

  setup_skylark_repository
  # Our custom repository rule
  cat >test.bzl <<EOF
def _impl(repository_ctx):
  repository_ctx.file("BUILD")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract1.tar.gz", "")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract2.zip", "", "")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract1.tar.gz", "some/path")
  repository_ctx.download_and_extract(
    "http://localhost:${fileserver_port}/download_and_extract3.zip", ".", "${file_sha256}", "", "")
  repository_ctx.download_and_extract(
    url = ["http://localhost:${fileserver_port}/download_and_extract3.zip"],
    output = "other/path",
    sha256 = "${file_sha256}"
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
  cat > WORKSPACE <<EOF
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


function test_timeout_tunable() {
  cat > WORKSPACE <<'EOF'
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

function tear_down() {
  shutdown_server
  if [ -d "${TEST_TMPDIR}/server_dir" ]; then
    rm -fr "${TEST_TMPDIR}/server_dir"
  fi
  true
}

run_suite "local repository tests"
