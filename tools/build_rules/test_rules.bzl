# Copyright 2015 Google Inc. All rights reserved.
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

# This is a quick and dirty rule to make Bazel compile itself.  It
# only supports Java.

"""
test_rules.bzl: Utilities for testing bazel
"""

### First, trivial tests that either always pass, always fail,
### or pass depending on a trivial computation

def success_target(ctx, msg):
  """Return a success for an analysis test.

The test rule must have an executable output.
"""
  exe = ctx.outputs.executable
  dat = ctx.new_file(ctx.data_configuration.genfiles_dir, exe, ".dat")
  ctx.file_action(
      output = dat,
      content = msg)
  ctx.file_action(
      output = exe,
      content = "cat " + dat.path + " ; echo",
      executable = True)
  return struct(runfiles=ctx.runfiles([exe, dat]))

def _successful_test_impl(ctx):
  return success_target(ctx, ctx.attr.msg)

successful_test = rule(
    implementation=_successful_test_impl,
    attrs={"msg": attr.string(mandatory=True)},
    test=True, executable=True)


def failure_target(ctx, msg):
  """Return a failure for an analysis test.

The test rule must have an executable output.
"""
  ### fail(msg) ### <--- This would fail at analysis time.
  exe = ctx.outputs.executable
  dat = ctx.new_file(ctx.data_configuration.genfiles_dir, exe, ".dat")
  ctx.file_action(
      output = dat,
      content = msg)
  ctx.file_action(
      output = exe,
      content = "(cat " + dat.short_path + " ; echo ) >&2 ; exit 1",
      executable = True)
  return struct(runfiles=ctx.runfiles([exe, dat]))

def _failed_test_impl(ctx):
  return failure_target(ctx, ctx.attr.msg)

failed_test = rule(
    implementation=_failed_test_impl,
    attrs={"msg": attr.string(mandatory=True)},
    test=True, executable=True)


### Second, general purpose utilities

def assert_(condition, string="assertion failed", *args):
  """Trivial assertion mechanism.

  Not quite compatible with python assert(): it takes % arguments."""

  if not condition:
    fail(string % args)


def strip_prefix(prefix, string):
  assert_(string.startswith(prefix),
    "%s does not start with %s", string, prefix)
  return string[len(prefix)+1:len(string)]


def expectation_description(expect=None, expect_failure=None):
  """Turn expectation of result or error into a string"""
  if expect_failure:
    return "failure " + str(expect_failure)
  else:
    return "result " + repr(expect)


def check_results(result, failure_message, expect, expect_failure):
  """See if actual computation matches expectation

  return: a pair (tuple) of a boolean (true if success)
  and a message (string)"""

  wanted = expectation_description(expect, expect_failure)
  found = expectation_description(result, failure_message)
  if wanted == found:
    return (True, "successfully computed " + wanted)
  else:
    return (False, "expect " + wanted + " but found " + found)


def load_results(name, result=None, failure=None, expect=None, expect_failure=None):
  """issue load-time success or failure of a test of given name based on results."""
  (is_success, msg) = check_results(result, failure, expect, expect_failure)
  this_test = successful_test if is_success else failed_test
  return this_test(name=name, msg=msg)


def analysis_results(ctx, result=None, failure=None, expect=None, expect_failure=None):
  """issue analysis-time success or failure of a test of given ctx based on results."""
  (is_success, msg) = check_results(result, failure, expect, expect_failure)
  this_test = success_target if is_success else failure_target
  return this_test(ctx, msg)



### Simple tests

def _rule_test_impl(ctx):
  """check that a rule generates the desired outputs and providers"""
  rule_ = ctx.attr.rule
  rule_name = str(rule_.label)
  if hasattr(ctx, "generates"):
    prefix = ctx.label.package + "/"
    generates = ctx.attr.generates
    generated = [strip_prefix(prefix, f.short_path) for f in rule.files]
    if not generates == generated:
      fail("rule %s generates %s not %s"
        % (rule_name, repr(generated), repr(generates)))
  if hasattr(ctx, "provides"):
    # TODO(bazel-team): implement this!
    fail("provides not implemented yet!")
  return success_target(ctx, "success")


rule_test = rule(
    implementation=_rule_test_impl,
    attrs={
        "rule": attr.label(mandatory=True),
        "generates": attr.string_list(),
        "provides": attr.string_dict()},
    test=True, executable=True)


def _file_test_impl(ctx):
  """check that a file has a given content"""
  exe = ctx.outputs.executable
  file = ctx.file.file
  content = ctx.attr.content
  regexp = ctx.attr.regexp
  matches = ctx.attr.matches
  if (content == "") == (regexp == ""):
    fail("Must specify one and only one of content or regexp")
  if content != "" and matches != -1:
    fail("matches only makes sense with regexp")
  if content != "":
    dat = ctx.new_file(ctx.data_configuration.genfiles_dir, exe, ".dat")
    ctx.file_action(
      output=dat,
      content=content)
    ctx.file_action(
      output = exe,
      content = "diff -u %s %s" % (dat.short_path, file.short_path),
      executable = True)
    return struct(runfiles=ctx.runfiles([exe, dat, file]))
  if matches != -1:
    script = "[ %s == $(grep -c %s %s) ]" % (matches, repr(regexp), file.path)
  else:
    script = "grep %s %s" % (repr(regexp), file.path)
  ctx.file_action(
    output = exe,
    content = script,
    executable = True)
  return struct(runfiles=ctx.runfiles([exe, file]))

file_test = rule(
    implementation=_file_test_impl,
    attrs={
      "file": attr.label(mandatory=True, allow_files=True, single_file=True),
      "content": attr.string(default=""),
      "regexp": attr.string(default=""),
      "matches": attr.int(default=-1)},
    test=True, executable=True)

