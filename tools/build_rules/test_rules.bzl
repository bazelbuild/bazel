"""Utilities for testing bazel."""
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

### First, trivial tests that either always pass, always fail,
### or sometimes pass depending on a trivial computation.

def success_target(ctx, msg):
    """Return a success for an analysis test.

    The test rule must have an executable output.

    Args:
      ctx: the Bazel rule context
      msg: an informative message to display

    Returns:
      a suitable rule implementation struct(),
      with actions that always succeed at execution time.
    """
    exe = ctx.outputs.executable
    dat = ctx.new_file(ctx.genfiles_dir, exe, ".dat")
    ctx.actions.write(
        output = dat,
        content = msg,
    )
    ctx.actions.write(
        output = exe,
        content = "cat " + dat.path + " ; echo",
        is_executable = True,
    )
    return struct(runfiles = ctx.runfiles([exe, dat]))

def _successful_test_impl(ctx):
    return success_target(ctx, ctx.attr.msg)

successful_test = rule(
    attrs = {"msg": attr.string(mandatory = True)},
    executable = True,
    test = True,
    implementation = _successful_test_impl,
)

def failure_target(ctx, msg):
    """Return a failure for an analysis test.

    The test rule must have an executable output.

    Args:
      ctx: the Bazel rule context
      msg: an informative message to display

    Returns:
      a suitable rule implementation struct(),
      with actions that always fail at execution time.
    """

    ### fail(msg) ### <--- This would fail at analysis time.
    exe = ctx.outputs.executable
    dat = ctx.new_file(ctx.genfiles_dir, exe, ".dat")
    ctx.file_action(
        output = dat,
        content = msg,
    )
    ctx.file_action(
        output = exe,
        content = "(cat " + dat.short_path + " ; echo ) >&2 ; exit 1",
        executable = True,
    )
    return struct(runfiles = ctx.runfiles([exe, dat]))

def _failed_test_impl(ctx):
    return failure_target(ctx, ctx.attr.msg)

failed_test = rule(
    attrs = {"msg": attr.string(mandatory = True)},
    executable = True,
    test = True,
    implementation = _failed_test_impl,
)

### Second, general purpose utilities

def assert_(condition, string = "assertion failed", *args):
    """Trivial assertion mechanism.

    Args:
      condition: a generalized boolean expected to be true
      string: a format string for the error message should the assertion fail
      *args: format arguments for the error message should the assertion fail

    Returns:
      None.

    Raises:
      an error if the condition isn't true.
    """

    if not condition:
        fail(string % args)

def strip_prefix(prefix, string):
    assert_(
        string.startswith(prefix),
        "%s does not start with %s",
        string,
        prefix,
    )
    return string[len(prefix):len(string)]

def expectation_description(expect = None, expect_failure = None):
    """Turn expectation of result or error into a string."""
    if expect_failure:
        return "failure " + str(expect_failure)
    else:
        return "result " + repr(expect)

def check_results(result, failure, expect, expect_failure):
    """See if actual computation results match expectations.

    Args:
      result: the result returned by the test if it ran to completion
      failure: the failure message caught while testing, if any
      expect: the expected result for a successful test, if no failure expected
      expect_failure: the expected failure message for the test, if any

    Returns:
      a pair (tuple) of a boolean (true if success) and a message (string).
    """
    wanted = expectation_description(expect, expect_failure)
    found = expectation_description(result, failure)
    if wanted == found:
        return (True, "successfully computed " + wanted)
    else:
        return (False, "expect " + wanted + " but found " + found)

def load_results(
        name,
        result = None,
        failure = None,
        expect = None,
        expect_failure = None):
    """issue load-time results of a test.

    Args:
      name: the name of the Bazel rule at load time.
      result: the result returned by the test if it ran to completion
      failure: the failure message caught while testing, if any
      expect: the expected result for a successful test, if no failure expected
      expect_failure: the expected failure message for the test, if any

    Returns:
      None, after issuing a rule that will succeed at execution time if
      expectations were met.
    """
    (is_success, msg) = check_results(result, failure, expect, expect_failure)
    this_test = successful_test if is_success else failed_test
    return this_test(name = name, msg = msg)

def analysis_results(
        ctx,
        result = None,
        failure = None,
        expect = None,
        expect_failure = None):
    """issue analysis-time results of a test.

    Args:
      ctx: the Bazel rule context
      result: the result returned by the test if it ran to completion
      failure: the failure message caught while testing, if any
      expect: the expected result for a successful test, if no failure expected
      expect_failure: the expected failure message for the test, if any

    Returns:
      a suitable rule implementation struct(),
      with actions that succeed at execution time if expectation were met,
      or fail at execution time if they didn't.
    """
    (is_success, msg) = check_results(result, failure, expect, expect_failure)
    this_test = success_target if is_success else failure_target
    return this_test(ctx, msg)

### Simple tests

def _rule_test_impl(ctx):
    """check that a rule generates the desired outputs and providers."""
    rule_ = ctx.attr.rule
    rule_name = str(rule_.label)
    exe = ctx.outputs.executable
    if ctx.attr.generates:
        # Generate the proper prefix to remove from generated files.
        prefix_parts = []

        if rule_.label.workspace_root:
            # Create a prefix that is correctly relative to the output of this rule.
            prefix_parts = ["..", strip_prefix("external/", rule_.label.workspace_root)]

        if rule_.label.package:
            prefix_parts.append(rule_.label.package)

        prefix = "/".join(prefix_parts)

        if prefix:
            # If the prefix isn't empty, it needs a trailing slash.
            prefix = prefix + "/"

        # TODO(bazel-team): Use set() instead of sorted() once
        # set comparison is implemented.
        # TODO(bazel-team): Use a better way to determine if two paths refer to
        # the same file.
        generates = sorted(ctx.attr.generates)
        generated = sorted([
            strip_prefix(prefix, f.short_path)
            for f in rule_.files.to_list()
        ])
        if generates != generated:
            fail("rule %s generates %s not %s" %
                 (rule_name, repr(generated), repr(generates)))
    provides = ctx.attr.provides
    if provides:
        files = []
        commands = []
        for k in provides.keys():
            if hasattr(rule_, k):
                v = repr(getattr(rule_, k))
            else:
                fail(("rule %s doesn't provide attribute %s. " +
                      "Its list of attributes is: %s") %
                     (rule_name, k, dir(rule_)))
            file_ = ctx.new_file(ctx.genfiles_dir, exe, "." + k)
            files += [file_]
            regexp = provides[k]
            commands += [
                "if ! grep %s %s ; then echo 'bad %s:' ; cat %s ; echo ; exit 1 ; fi" %
                (repr(regexp), file_.short_path, k, file_.short_path),
            ]
            ctx.file_action(output = file_, content = v)
        script = "\n".join(commands + ["true"])
        ctx.file_action(output = exe, content = script, executable = True)
        return struct(runfiles = ctx.runfiles([exe] + files))
    else:
        return success_target(ctx, "success")

rule_test = rule(
    attrs = {
        "rule": attr.label(mandatory = True),
        "generates": attr.string_list(),
        "provides": attr.string_dict(),
    },
    executable = True,
    test = True,
    implementation = _rule_test_impl,
)

def _file_test_impl(ctx):
    """check that a file has a given content."""
    script1 = ctx.actions.declare_file(ctx.label.name + "-gen.bash")

    # Since file_test is used from the @bazel_tools repo but also tested in the main Bazel repo, we
    # cannot create a sh_binary for the script below and use it from file_test, because depending on
    # the sh_binary would require knowing which depot the file_test is instantiated from.
    #
    # In other words, if there were a @bazel_tools//tools/build_rules:file_test_helper rule (the
    # sh_binary), then file_test could reference it as:
    #   - @bazel_tools//tools/build_rules:file_test_helper, which would work when using a Bazel
    #     version that already contains this target, but not with Bazel 0.17.2 (latest release as of
    #     the writing of this comment), or
    #   - @io_bazel//tools/build_rules:file_test_helper, which would only work if the @io_bazel repo
    #     is defined (either in Bazel's own source tree, or if the current project imports Bazel's
    #     tree)
    #   - //tools/build_rules:file_test_helper, which would only work if the current source tree
    #     contains this target, which is unlikely.
    # Considering that all 3 options are wrong, we resort to writing a script file on-the-fly.
    ctx.actions.write(script1, is_executable = True, content = """#!/bin/bash
set -euo pipefail
declare -r OUT="$1"
declare -r INPUT="$2"
declare -r IS_WINDOWS="$3"
declare -r CONTENT="$4"
declare -r REGEXP="$5"
declare -r MATCHES="$6"

if [[ ( -n "${CONTENT:-}" && -n "${REGEXP:-}" ) || ( -z "${CONTENT:-}" && -z "${REGEXP:-}" ) ]]; then
  echo >&2 "ERROR: expected either 'content' or 'regexp'"
  exit 1
elif [[ -n "${CONTENT:-}" && ( -n "${MATCHES:-}" && "$MATCHES" != "-1" ) ]]; then
  echo >&2 "ERROR: cannot specify 'matches' together with 'content'"
  exit 1
elif [[ ! ( -z "${MATCHES:-}" || "$MATCHES" = 0 || "$MATCHES" =~ ^-?[1-9][0-9]*$ ) ]]; then
  echo >&2 "ERROR: 'matches' must be an integer"
  exit 1
elif [[ ! -e "${INPUT:-/dev/null/does-not-exist}" ]]; then
  echo >&2 "ERROR: input file must exist"
  exit 1
else
  if [[ -n "${CONTENT:-}" ]]; then
    declare -r GOLDEN_FILE="$(mktemp)"
    declare -r ACTUAL_FILE="$(mktemp)"
    # Normalize line endings in both files.
    echo -e -n "$CONTENT" | sed 's,\\r\\n,\\n,g' > "$GOLDEN_FILE"
    sed 's,\\r\\n,\\n,g' "$INPUT" > "$ACTUAL_FILE"
    if ! diff -u "$GOLDEN_FILE" "$ACTUAL_FILE" ; then
      echo >&2 "ERROR: file did not have expected content"
      exit 1
    fi
  else
    if [[ -n "${MATCHES:-}" && $MATCHES -gt -1 ]]; then
      if [[ "$MATCHES" != $(grep -c "$REGEXP" "$INPUT") ]]; then
        echo >&2 "ERROR: file did not contain expected regexp $MATCHES times"
        exit 1
      fi
    else
      if ! grep "$REGEXP" "$INPUT"; then
        echo >&2 "ERROR: file did not contain expected regexp"
        exit 1
      fi
    fi
  fi

  # Write a platform-specific script that is the actual test.
  # The test script is a dummy, always-passing test. However if this script got to this point, then
  # the actual assertions succeeded.
  # The test script has an embedded timestamp, for the purpose of correct cache hit reporting. If it
  # didn't, Bazel would always report file_test to be cached, because its only input (the test
  # script) would never change. However, Bazel would still rebuild the action that generates the
  # test script, i.e. it would perform the actual assertions, but it would look like it didn't.
  if [[ "${IS_WINDOWS:-}" = "yes" ]]; then
    echo -e "@rem $(date +"%s.%N")\\n@echo PASSED" > "$OUT"
  else
    echo -e "#!/bin/sh\\n# $(date +"%s.%N")\\necho PASSED" > "$OUT"
  fi
  chmod +x "$OUT"
fi
""")

    is_windows = bool(ctx.attr.is_windows)
    script2 = ctx.actions.declare_file(ctx.label.name + (".bat" if is_windows else ".bash"))

    # TODO(laszlocsomor): once https://github.com/bazelbuild/bazel/issues/6391 is fixed, change the
    # "command" to only contain script1's path, and pass arguments with the "arguments" attribute.
    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [script2],
        tools = [script1],
        command = " ".join([
            script1.path,
            script2.path,
            ctx.file.src.path,
            "yes" if is_windows else "no",
            repr(ctx.attr.content),
            repr(ctx.attr.regexp),
            ctx.attr.matches if ctx.attr.matches > -1 else repr(""),
        ]),
    )

    return [DefaultInfo(executable = script2)]

_file_test = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "is_windows": attr.bool(mandatory = True),
        "content": attr.string(default = ""),
        "regexp": attr.string(default = ""),
        "matches": attr.int(default = -1),
    },
    executable = True,
    test = True,
    implementation = _file_test_impl,
)

def file_test(name, file, content = None, regexp = None, matches = None, **kwargs):
    _file_test(
        name = name,
        src = file,
        content = content,
        regexp = regexp,
        matches = matches or -1,
        is_windows = select({
            "@bazel_tools//src/conditions:windows": True,
            "//conditions:default": False,
        }),
        **kwargs
    )
